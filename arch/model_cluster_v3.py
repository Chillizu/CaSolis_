"""
ModelCluster v3 — 共享专家 + 负载均衡 + 动态扩展接口

架构升级:
  ┌─ Shared Expert (stable base)
  ├─ Routed Experts (specialized, configurable count)
  ├─ Router with load-balancing (aux loss)
  ├─ World Model (MLP, not Linear)
  └─ add_expert() — 动态扩展接口
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, random
from arch.mamba_model import MambaBlock

V = 2022
TV = V + 19
PAD = TV - 1


class SharedExpert(nn.Module):
    """共享专家 — 所有输入都经过它（提供通用知识底座）"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.mamba(x))


class RoutedExpert(nn.Module):
    """可路由专家 — 被路由器选中的才激活"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.mamba(x))


class RouterV3(nn.Module):
    """路由器 v3 — 带温度 + 探索 + 负载均衡输出"""
    def __init__(self, d_model=1024, n_routed=4):
        super().__init__()
        self.n_routed = n_routed
        shared_dim = d_model  # 共享专家的输出维度
        routed_dim = d_model   # 路由专家的输出维度

        self.net = nn.Sequential(
            nn.Linear(shared_dim + routed_dim * n_routed + n_routed, 256),
            nn.ReLU(),
            nn.Linear(256, n_routed),
        )

        # 每个专家的偏置（初始不同 → 帮助路由分化）
        self.expert_bias = nn.Parameter(torch.randn(n_routed) * 0.1)

        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, shared_h, routed_h, last_selection=None, explore=False):
        """
        shared_h: (B, d) — 共享专家输出
        routed_h: (B, K, d) — 各路由专家输出
        last_selection: (B, K) — one-hot 上次选择
        """
        B, K, d = routed_h.shape
        flat = torch.cat([shared_h] + [routed_h[:, k] for k in range(K)], dim=-1)
        if last_selection is None:
            last_selection = torch.zeros(B, K, device=shared_h.device)
        flat = torch.cat([flat, last_selection], dim=-1)

        logits = self.net(flat) + self.expert_bias.unsqueeze(0)

        if explore:
            noise = torch.randn_like(logits) * 0.5
            logits = logits + noise

        # 训练时始终加少量噪声（鼓励探索，防止坍缩）
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        return F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)


class WorldModelV3(nn.Module):
    """世界模型 v3 — MLP 替代 Linear"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, h):
        return self.net(h)


class ModelClusterV3(nn.Module):
    """ModelCluster v3 — 共享 + 路由 + 可扩展"""

    def __init__(self, d_model=1024, n_routed=4):
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed

        # 共享嵌入
        self.embed = nn.Embedding(TV, d_model)

        # 共享专家（始终激活）
        self.shared_expert = SharedExpert(d_model)

        # 路由专家（可扩展）
        self.routed_experts = nn.ModuleList([
            RoutedExpert(d_model) for _ in range(n_routed)
        ])

        # 路由器
        self.router = RouterV3(d_model, n_routed)

        # 世界模型（MLP）
        self.world_model = WorldModelV3(d_model)

        # 输出头
        self.lm_head = nn.Linear(d_model, V)
        self.confidence_head = nn.Linear(d_model, 1)

        # 使用统计（用于负载均衡）
        self.register_buffer("expert_usage", torch.zeros(n_routed))
        self.register_buffer("total_steps", torch.zeros(1))

    def add_expert(self, from_checkpoint=None):
        """
        动态扩展：加一个新路由专家。

        from_checkpoint: 如果提供 checkpoint 路径，从已有权重初始化
                         否则随机初始化
        """
        new_expert = RoutedExpert(self.d_model)
        if from_checkpoint:
            sd = torch.load(from_checkpoint, map_location="cpu", weights_only=True)
            # 只加载 mamba + norm 权重
            for name, param in new_expert.named_parameters():
                ck_key = f"mamba.{name}" if not name.startswith("norm") else name
                if ck_key in sd and param.shape == sd[ck_key].shape:
                    param.data.copy_(sd[ck_key])

        self.routed_experts.append(new_expert)
        self.n_routed += 1

        # 更新路由器输出维度
        old_router = self.router
        self.router = RouterV3(self.d_model, self.n_routed)
        # 从旧路由器复制权重（保持已有知识）
        with torch.no_grad():
            for old_p, new_p in zip(old_router.parameters(), self.router.parameters()):
                if old_p.shape == new_p.shape:
                    new_p.copy_(old_p)
            # 新专家的偏置略高（鼓励它被使用）
            if hasattr(self.router, 'expert_bias'):
                self.router.expert_bias.data[-1] = 0.3

        # 重置使用统计
        self.expert_usage = torch.zeros(self.n_routed)
        self.total_steps = torch.zeros(1)

        print(f"  ➕ 新增专家 E{self.n_routed-1}, 共 {self.n_routed} 个路由专家", flush=True)
        return self.n_routed - 1

    def init_from_checkpoint(self, ckpt_path):
        """从已训练的 Mamba checkpoint 初始化"""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # 共享嵌入
        if "embed.weight" in sd:
            self.embed.weight.data.copy_(sd["embed.weight"])

        # 共享专家
        ms = self.shared_expert.state_dict()
        for ck_key, val in sd.items():
            key = ck_key.replace("mamba.", "") if ck_key.startswith("mamba.") else ck_key
            if key in ms and ms[key].shape == val.shape:
                ms[key].copy_(val)
        self.shared_expert.load_state_dict(ms, strict=False)

        # 路由专家（从同一 checkpoint 初始化）
        for expert in self.routed_experts:
            es = expert.state_dict()
            for ck_key, val in sd.items():
                key = ck_key.replace("mamba.", "") if ck_key.startswith("mamba.") else ck_key
                if key in es and es[key].shape == val.shape:
                    es[key].copy_(val)
            expert.load_state_dict(es, strict=False)

        # 输出头
        self_params = dict(self.named_parameters())
        for key in ["lm_head.weight", "lm_head.bias"]:
            if key in sd and key in self_params:
                self_params[key].data.copy_(sd[key])

        print(f"  ✅ 从 {ckpt_path} 加载权重", flush=True)

    def forward(self, tokens, explore_rate=0.0):
        """
        训练前向。

        tokens: (B, S)
        Returns: loss, metrics (dict), usage (list)
        """
        B, S = tokens.shape
        x = self.embed(tokens)  # (B, S, d)

        # 共享专家前向
        shared_out = self.shared_expert(x)  # (B, S, d)

        # 各路由专家前向
        routed_out = torch.stack(
            [expert(x) for expert in self.routed_experts],
            dim=1
        )  # (B, K, S, d)

        # 逐 step
        total_loss = torch.tensor(0.0)
        total_lm_loss = 0.0
        total_world_loss = 0.0
        total_load_loss = 0.0
        n_steps = 0
        expert_counts = torch.zeros(self.n_routed)
        last_onehot = torch.zeros(B, self.n_routed)

        for t in range(S - 1):
            h_shared = shared_out[:, t]       # (B, d)
            h_routed = routed_out[:, :, t]    # (B, K, d)
            target = tokens[:, t + 1]

            # 路由
            explore = random.random() < explore_rate
            route = self.router(h_shared, h_routed, last_onehot, explore=explore)

            # 用路由权重混合所有专家的输出 → LM 预测
            # 混合策略：共享专家 + 路由加权
            mixed_h = h_shared + torch.sum(
                route.unsqueeze(-1) * h_routed, dim=1
            )  # (B, d)

            logits = self.lm_head(mixed_h)  # (B, V)
            valid = target < V
            if valid.sum() > 0:
                lm_loss = F.cross_entropy(logits[valid], target[valid])
            else:
                lm_loss = torch.tensor(0.0)

            # 世界模型 loss（预测下一个 token 的嵌入）
            pred_emb = self.world_model(mixed_h)
            next_target = target.clamp(0, V - 1)
            next_emb = self.embed(next_target)
            world_loss = 0.3 * F.mse_loss(pred_emb, next_emb.detach())

            # Load balancing loss（均匀分布 + epsilon 防 log(0)）
            expert_probs = route.mean(dim=0) + 1e-6
            expert_probs = expert_probs / expert_probs.sum()
            uniform = torch.full_like(expert_probs, 1.0 / self.n_routed)
            load_loss = 0.5 * torch.sum(
                expert_probs * (expert_probs / uniform).log()
            )

            step_loss = lm_loss + world_loss + load_loss
            total_loss = total_loss + step_loss
            total_lm_loss += lm_loss.item()
            total_world_loss += world_loss.item() if isinstance(world_loss, torch.Tensor) else 0
            total_load_loss += load_loss.item()

            # 统计
            choices = route.argmax(dim=1)
            for b in range(B):
                expert_counts[choices[b]] += 1

            n_steps += 1
            last_onehot = route.detach()

        avg_loss = total_loss / n_steps
        usage = (expert_counts / (n_steps * B)).tolist()

        # 更新使用统计
        self.expert_usage = self.expert_usage * 0.99 + torch.tensor(usage) * 0.01 * B * n_steps
        self.total_steps += n_steps * B

        return avg_loss, {
            "lm": total_lm_loss / n_steps,
            "world": total_world_loss / n_steps,
            "load": total_load_loss / n_steps,
        }, usage

    @torch.no_grad()
    def generate(self, seed_ids, n=60, temp=0.85, explore=False):
        """自回归生成"""
        out = list(seed_ids)
        self.eval()
        last_onehot = torch.zeros(1, self.n_routed)

        for _ in range(n):
            x = self.embed(torch.tensor([[out[-1]]]).long())

            h_shared = self.shared_expert(x)[:, 0]
            h_routed = torch.stack([e(x)[:, 0] for e in self.routed_experts], dim=1)

            route = self.router(h_shared, h_routed, last_onehot, explore=explore)

            mixed_h = h_shared + torch.sum(route.unsqueeze(-1) * h_routed, dim=1)

            logits = self.lm_head(mixed_h)
            lp = F.softmax(logits / temp, dim=-1)
            nt = torch.multinomial(lp.squeeze(0), 1).item()
            if nt >= V:
                break
            out.append(nt)
            last_onehot = route

        self.train()
        return out


def test():
    print("🧪 ModelCluster v3 — Shared + Routed + Expandable", flush=True)

    model = ModelClusterV3(d_model=1024, n_routed=4)
    total = sum(p.numel() for p in model.parameters())
    print(f"   总参数: {total:,} ({total/1e6:.1f}M)", flush=True)

    shared_p = sum(p.numel() for p in model.shared_expert.parameters())
    routed_p = sum(p.numel() for p in model.routed_experts[0].parameters())
    print(f"   共享专家: {shared_p:,}, 每个路由专家: {routed_p:,}", flush=True)

    # 尝试加载 checkpoint
    import os
    ckpt = "checkpoints/big-mamba/model_best.pt"
    if os.path.exists(ckpt):
        model.init_from_checkpoint(ckpt)

    # 前向
    B, S = 4, 32
    tokens = torch.randint(0, 100, (B, S))
    loss, metrics, usage = model(tokens, explore_rate=0.1)
    print(f"\n📊 前向: LM={metrics['lm']:.3f} W={metrics['world']:.3f} L={metrics['load']:.3f}", flush=True)
    print(f"   使用率: {' '.join(f'E{k}:{u:.0%}' for k,u in enumerate(usage))}", flush=True)
    loss.backward()
    print(f"   反向: ✅", flush=True)

    # 测试动态加专家
    print(f"\n  动态加专家测试...", flush=True)
    idx = model.add_expert(ckpt)
    print(f"   新增 E{idx}, 现在有 {model.n_routed} 个路由专家", flush=True)

    loss2, metrics2, usage2 = model(tokens, explore_rate=0.1)
    print(f"   加专家后前向: LM={metrics2['lm']:.3f}", flush=True)
    print(f"   使用率: {' '.join(f'E{k}:{u:.0%}' for k,u in enumerate(usage2))}", flush=True)

    print("\n✅ ModelCluster v3 OK")


if __name__ == "__main__":
    test()
