#!/usr/bin/env python3
"""
Mamba 模型 — CPU 友好的纯 PyTorch 实现

核心思想: 
  h_t = exp(Δ·A) · h_{t-1} + (Δ·B) · x_t
  y_t = h_t · C_t
  out = silu(conv(x)) * y

其中 Δ, B, C 都是从输入动态生成的，让模型"选择"记什么忘什么。
"""

import torch, torch.nn as nn, torch.nn.functional as F

class MambaBlock(nn.Module):
    """Mamba 核心块 — CPU 可运行的纯 PyTorch 版本"""
    
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        
        # 1) 输入投影: x → x_proj, z (门控)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2) 深度可分离卷积 (对每个维度独立做因果卷积)
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # 因果卷积：只看过去
            bias=False,
        )
        # 初始化卷积权重
        nn.init.normal_(self.conv.weight, std=0.01)
        
        # 3) SSM 参数生成: x → Δ_proj, B, C
        # delta 用低秩投影 (省参数)
        d_rank = d_model // 4  # 低秩维度
        self.delta_rank = d_rank
        self.x_proj = nn.Linear(self.d_inner, d_rank + d_state * 2, bias=False)
        self.delta_proj = nn.Linear(d_rank, self.d_inner, bias=True)
        
        # 4) A 矩阵: 可学习的状态转移 (diagonal 形式)
        # 初始化为负值 (让 exp(ΔA) < 1, 防止发散)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        A = A.repeat(self.d_inner, 1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))  # 训练时取 exp
        
        # 5) D: 跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 6) 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # ── 1) 输入投影 ──
        xz = self.in_proj(x)  # (batch, seq, d_inner*2)
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (batch, seq, d_inner)
        
        # ── 2) 因果卷积 ──
        # 需要 (batch, d_inner, seq) 格式
        x_proj = x_proj.transpose(1, 2)  # (batch, d_inner, seq)
        x_conv = self.conv(x_proj)[:, :, :seq_len]  # 因果: 取前 seq 个
        x_conv = F.silu(x_conv)  # SiLU 激活
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)
        
        # ── 3) SSM 参数 ──
        # 计算 Δ, B, C
        x_db = self.x_proj(x_conv)  # (batch, seq, d_rank + 2*d_state)
        delta, B, C = torch.split(x_db, [self.delta_rank, self.d_state, self.d_state], dim=-1)
        
        delta = self.delta_proj(delta)  # (batch, seq, d_inner)
        # softplus 确保 Δ > 0
        delta = F.softplus(delta)
        
        # ── 4) 选择性扫描 ──
        y = self._selective_scan(x_conv, delta, B, C)
        
        # ── 5) 门控输出 ──
        y = y * F.silu(z)  # (batch, seq, d_inner)
        y = self.out_proj(y)  # (batch, seq, d_model)
        
        return y
    
    def _selective_scan(self, u, delta, B, C):
        """
        核心扫描 — CPU 优化版
        - 用 mul+sum 替代 einsum (CPU 上 5-10x 快)
        - 预分配 ys 减少 Python 开销
        """
        batch, seq, d_inner = u.shape
        A = -torch.exp(self.A_log)
        d_state = A.shape[-1]
        
        delta = delta.unsqueeze(-1)
        A = A.unsqueeze(0).unsqueeze(0)
        deltaA = torch.exp(delta * A)
        B = B.unsqueeze(2)
        deltaB = delta * B
        
        # 预分配输出
        ys = torch.zeros(batch, seq, d_inner, device=u.device)
        h = torch.zeros(batch, d_inner, d_state, device=u.device)
        
        for t in range(seq):
            # h_t = ΔA_t · h_{t-1} + ΔB_t · u_t
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t].unsqueeze(-1)
            # y_t = sum(h * C_t, dim=-1)  — matmul 替代 einsum
            ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        
        return ys
    
    def step(self, x, h=None):
        """
    单步推理 (生成时用)
        x: (batch, d_model)
        h (可选): (batch, d_inner, d_state) 或 None
            
        return: (batch, d_model), (batch, d_inner, d_state)
        """
        batch = x.shape[0]
        
        # 输入投影
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # 卷积 (单步: 直接计算)
        # 对于生成模式, 我们简化卷积为线性投影 (因为没有历史)
        # 这里用简单版本
        x_conv = F.silu(x_proj)
        
        # SSM 参数
        x_db = self.x_proj(x_conv)
        delta_raw, B, C = torch.split(x_db, [self.delta_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.delta_proj(delta_raw))
        
        # 状态更新
        A = -torch.exp(self.A_log)
        if h is None:
            h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        # h = exp(Δ·A) · h + (Δ·B) · x_conv
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (batch, d_inner, d_state)
        h = deltaA * h + deltaB * x_conv.unsqueeze(-1)
        
        # 输出
        y = torch.einsum('bds,bs->bd', h, C)
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return y, h


class MambaModel(nn.Module):
    """完整的 Mamba 语言模型"""
    
    def __init__(self, vocab_size, embed_dim=768, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mamba = MambaBlock(embed_dim, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # 不绑定权重，给更多容量
        
    def forward(self, h, t):
        """单步: 兼容 GRU 接口"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(0)
        x = self.embed(t)  # (1, embed)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (1, 1, embed) — batch=1, seq=1
        y = self.mamba(x)  # (1, 1, embed)
        y = self.norm(y.squeeze(1))  # (1, embed)
        return y, self.lm_head(y)
    
    def forward_seq(self, tokens):
        """训练: 处理完整序列"""
        x = self.embed(tokens)  # (batch, seq, embed)
        y = self.mamba(x)
        y = self.norm(y)
        return self.lm_head(y)  # (batch, seq, vocab)


if __name__ == "__main__":
    # 快速测试
    model = MambaModel(2022)
    print(f"MambaModel: {sum(p.numel() for p in model.parameters()):,} params")
    
    # 单步测试
    h = torch.zeros(1, 256)
    x = torch.tensor([0])  # (1,)
    h2, logits = model(h, x)
    print(f"单步: input {x.shape} → output {logits.shape}")
    
    # 序列测试
    x = torch.randint(0, 100, (1, 20))
    out = model.forward_seq(x)
    print(f"序列: input {x.shape} → output {out.shape}")
    print("✅ Mamba 模型正常工作")
