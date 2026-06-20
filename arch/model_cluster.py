"""
ModelCluster — 多专家协作架构

不多专家共同处理一个任务，路由器选择最合适的专家。

架构:
  Shared Embed
       ↓
    Router → 选专家
       ↓
  4 Experts (Mamba)
       ↓
  Shared Heads (lm_head, world_head, confidence)

每个专家是完整 MambaBlock，从已训 checkpoint 初始化。
路由器是轻量 MLP，从头训。
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, random
from arch.mamba_model import MambaBlock

V = 2022      # 词表大小
TV = V + 19   # 总词表（含特殊 token）
PAD = TV - 1  # padding


class Expert(nn.Module):
    """单个专家 = MambaBlock + LayerNorm"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.mamba(x))


class Router(nn.Module):
    """
    路由器 — 核心决策组件。

    输入: token embedding + 语境特征
    输出: 各专家的选择概率

    很小 (~100K 参数)，但决了定整个模型的行为。
    """
    def __init__(self, d_model=1024, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        # 语境特征：上一次选了谁 (4) + 当前 token 的部分信息
        self.feat_proj = nn.Linear(d_model + n_experts, 128)
        self.gate = nn.Linear(128, n_experts)

    def forward(self, x, last_expert=None, explore=False):
        """
        x: (B, d_model) — token 嵌入
        last_expert: (B, n_experts) — one-hot 上一步选的专家
        explore: bool — 探索模式

        Returns: (B, n_experts) 选择概率
        """
        B = x.shape[0]
        if last_expert is None:
            last_expert = torch.zeros(B, self.n_experts, device=x.device)

        inp = torch.cat([x, last_expert], dim=-1)
        h = torch.relu(self.feat_proj(inp))

        logits = self.gate(h)  # (B, n_experts)

        if explore:
            # 探索：加噪声到 logits
            noise = torch.randn_like(logits) * 0.3 * (1 + explore)
            logits = logits + noise

        return F.softmax(logits / 0.5, dim=-1)


class ModelCluster(nn.Module):
    """
    完整模型。

    使用规则:
      forward(tokens) — 训练（所有专家前向，路由器选最优）
      generate(seed) — 推理（只激活一个专家）
    """
    def __init__(self, d_model=1024, n_experts=4):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts

        # 共享嵌入
        self.embed = nn.Embedding(TV, d_model)

        # 专家集群
        self.experts = nn.ModuleList([
            Expert(d_model) for _ in range(n_experts)
        ])

        # 路由器
        self.router = Router(d_model, n_experts)

        # 共享输出头
        self.lm_head = nn.Linear(d_model, V)
        self.world_head = nn.Linear(d_model, d_model)
        self.confidence_head = nn.Linear(d_model, 1)

        # 记录上一步选的专家 (推理时用)
        self.last_expert = None

    def init_from_checkpoint(self, ckpt_path):
        """从训好的 MambaWorld checkpoint 加载权重"""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # 共享嵌入
        if "embed.weight" in sd:
            self.embed.weight.data.copy_(sd["embed.weight"])

        # 共享头
        self_params = dict(self.named_parameters())
        for key in ["lm_head.weight", "lm_head.bias", "world_head.weight", "world_head.bias"]:
            if key in sd and key in self_params:
                self_params[key].data.copy_(sd[key])

        # 各专家加载核心 Mamba + Norm 权重
        # checkpoint key 有 "mamba." 前缀，expert state_dict 没有
        loaded_count = 0
        for k_idx, expert in enumerate(self.experts):
            ms = expert.state_dict()
            for ck_key, ck_val in sd.items():
                if ck_key.startswith("mamba."):
                    # 去掉 "mamba." 前缀匹配 expert 的 key
                    expert_key = ck_key[6:]  # "mamba.in_proj" → "in_proj"
                    if expert_key in ms and ms[expert_key].shape == ck_val.shape:
                        ms[expert_key].copy_(ck_val)
                        loaded_count += 1
                elif ck_key.startswith("norm."):
                    if ck_key in ms and ms[ck_key].shape == ck_val.shape:
                        ms[ck_key].copy_(ck_val)
                        loaded_count += 1
            expert.load_state_dict(ms, strict=False)

        print(f"  ✅ Loaded: embed + {loaded_count} expert params + shared heads", flush=True)

    def forward(self, tokens, explore_rate=0.0):
        """
        训练前向。

        流程:
          1. Embed 输入
          2. 路由选择最优专家
          3. 所有专家前向（为了算「谁最该被选」）
          4. 损失 = LM + World + Confidence + LoadBalance

        tokens: (B, S)
        """
        B, S = tokens.shape
        d = self.d_model

        # Embed
        x = self.embed(tokens)  # (B, S, d)

        # 各专家处理
        expert_out = []
        for expert in self.experts:
            expert_out.append(expert(x))  # (B, S, d)

        expert_out = torch.stack(expert_out, dim=1)  # (B, K, S, d)

        # 路由器（逐 step）
        last_expert = torch.zeros(B, self.n_experts, device=tokens.device)
        total_loss = 0.0
        total_lm = 0.0
        total_world = 0.0
        total_conf = 0.0
        total_load = 0.0
        expert_counts = torch.zeros(self.n_experts)
        n_steps = 0

        for t in range(S - 1):
            # 当前 step 的隐藏状态
            h_t = expert_out[:, :, t]  # (B, K, d) 每个专家在 t 位置的输出

            # 路由
            explore = random.random() < explore_rate
            route = self.router(x[:, t], last_expert if t == 0 else route.detach(),
                                explore=explore)  # (B, K)

            # 目标 token
            target = tokens[:, t + 1]

            # 用所有专家的 logits 算加权 LM loss
            lm_loss_t = 0.0
            best_expert = None
            best_loss = float('inf')

            for k in range(self.n_experts):
                logits_k = self.lm_head(h_t[:, k])  # (B, V)
                valid = target < V
                if valid.sum() > 0:
                    loss_k = F.cross_entropy(logits_k[valid], target[valid])
                    weight = route[:, k].detach().mean()

                    # Track which expert is best
                    if loss_k.item() < best_loss:
                        best_loss = loss_k.item()
                        best_expert = k

                    lm_loss_t = lm_loss_t + weight * loss_k

            # 世界模型 loss（只看 E2）
            world_loss_t = torch.tensor(0.0)
            if self.n_experts >= 3:
                h_last = expert_out[:, 2, t]  # E2 的隐藏状态
                pred = self.world_head(h_last)
                # 预测下一个位置的嵌入
                next_emb = self.embed(tokens[:, t + 1].clamp(0, V-1))
                world_loss_t = 0.3 * F.mse_loss(pred, next_emb.detach())

            # 自信度 loss（E3）
            conf_loss_t = torch.tensor(0.0)
            if self.n_experts >= 4:
                h_last = expert_out[:, 3, t]
                conf = self.confidence_head(h_last).squeeze(-1)  # (B,)
                # 高置信度 = 对该专家的 LM loss 低
                selected_idx = route.argmax(dim=1).detach()  # (B,)
                target_conf = torch.ones(B, device=tokens.device)
                for b in range(B):
                    k = selected_idx[b].item()
                    # 用专家的 loss 作为置信度目标
                    if valid.sum() > 0 and k < self.n_experts:
                        with torch.no_grad():
                            logits_k = self.lm_head(expert_out[b, k, t].unsqueeze(0))
                            lk = F.cross_entropy(logits_k, target[b:b+1].clamp(0, V-1))
                            target_conf[b] = torch.exp(-lk)  # 低 loss → 高自信
                conf_loss_t = F.mse_loss(conf, target_conf)

            # Load balancing loss
            expert_choice = route.argmax(dim=1)
            for b in range(B):
                expert_counts[expert_choice[b]] += 1

            # 均匀分布的目标
            target_dist = torch.full((self.n_experts,), 1.0 / self.n_experts)
            actual_dist = expert_counts / (expert_counts.sum() + 1e-8)
            load_balance_loss = -torch.sum(target_dist * torch.log(actual_dist + 1e-8))

            loss_step = lm_loss_t + world_loss_t + conf_loss_t + 0.01 * load_balance_loss
            total_loss = total_loss + loss_step
            total_lm = total_lm + lm_loss_t.item()
            total_world = total_world + (world_loss_t.item() if isinstance(world_loss_t, torch.Tensor) else 0)
            total_conf = total_conf + (conf_loss_t.item() if isinstance(conf_loss_t, torch.Tensor) else 0)
            total_load = total_load + load_balance_loss.item()
            n_steps += 1

            # 更新 last_expert 用于下一步路由
            route_detach = route.detach()
            if t == 0:
                last_expert = route_detach

        avg_loss = total_loss / n_steps
        avg_lm = total_lm / n_steps
        avg_world = total_world / n_steps
        avg_conf = total_conf / n_steps
        avg_load = total_load / n_steps
        usage = (expert_counts / (n_steps * B)).tolist()

        return avg_loss, {
            "lm": avg_lm,
            "world": avg_world,
            "conf": avg_conf,
            "load_balance": avg_load,
        }, usage

    @torch.no_grad()
    def generate(self, seed_ids, n=60, temp=0.85, explore=False):
        """
        自回归生成。

        每次 step 只激活路由器选的专家。
        可以探索（选非主专家）或利用（选最优专家）。
        """
        out = list(seed_ids)
        B = 1
        self.eval()

        h_states = [None] * self.n_experts  # 各专家的隐藏状态
        last_onehot = torch.zeros(1, self.n_experts)

        for step in range(n):
            x = self.embed(torch.tensor([[out[-1]]]).long())  # (1, 1, d)

            # 所有专家前向
            h_t = []
            for k, expert in enumerate(self.experts):
                h_k = expert(x)  # (1, 1, d)
                h_t.append(h_k)

            h_t = torch.stack([h[:, 0] for h in h_t], dim=1)  # (1, K, d)

            # 路由
            route = self.router(x[:, 0], last_onehot, explore=explore)
            k = route[0].argmax().item()

            # 用选中专家的输出
            logits = self.lm_head(h_t[0, k])
            lp = F.softmax(logits / temp, dim=-1)
            nt = torch.multinomial(lp, 1).item()
            if nt >= V:
                break
            out.append(nt)

            # 更新 last_expert
            last_onehot = torch.zeros(1, self.n_experts)
            last_onehot[0, k] = 1.0

        self.train()
        return out

    def generate_with_plan(self, seed_ids, n=60, temp=0.85, explore=False):
        """
        带规划的生成：用世界专家（E2）规划，命令专家（E0）执行。
        """
        out = list(seed_ids)
        B = 1
        self.eval()

        last_onehot = torch.zeros(1, self.n_experts)

        for step in range(n):
            x = self.embed(torch.tensor([[out[-1]]]).long())

            # 所有专家
            h_t = []
            for k, expert in enumerate(self.experts):
                h_k = expert(x)
                h_t.append(h_k[:, 0])

            h_t = torch.stack(h_t, dim=1)

            route = self.router(x[:, 0], last_onehot, explore=explore)

            if explore and random.random() < 0.3:
                k = torch.multinomial(route[0], 1).item()
            else:
                k = route[0].argmax().item()

            logits = self.lm_head(h_t[0, k])
            lp = F.softmax(logits / temp, dim=-1)
            nt = torch.multinomial(lp, 1).item()
            if nt >= V:
                break
            out.append(nt)

            last_onehot = torch.zeros(1, self.n_experts)
            last_onehot[0, k] = 1.0

        self.train()
        return out


def test():
    print("🧪 ModelCluster — Multi-Expert Architecture Test", flush=True)

    model = ModelCluster(d_model=1024, n_experts=4)
    total = sum(p.numel() for p in model.parameters())
    print(f"   总参数: {total:,} ({total/1e6:.1f}M)", flush=True)

    per_expert = sum(p.numel() for p in model.experts[0].parameters())
    print(f"   每个专家: {per_expert:,} ({per_expert/1e6:.1f}M)", flush=True)
    print(f"   路由器: {sum(p.numel() for p in model.router.parameters()):,}", flush=True)
    print(f"   共享头: {sum(p.numel() for p in model.lm_head.parameters()) + sum(p.numel() for p in model.world_head.parameters()):,}", flush=True)

    # 尝试加载 checkpoint
    import os
    ckpt = "checkpoints/big-mamba/model_best.pt"
    if os.path.exists(ckpt):
        print("\n📂 加载 checkpoint...", flush=True)
        model.init_from_checkpoint(ckpt)
    else:
        print("\n⚠️  无 checkpoint，随机初始化", flush=True)

    # 前向测试
    B, S = 4, 32
    tokens = torch.randint(0, min(100, V), (B, S))

    loss, metrics, usage = model(tokens, explore_rate=0.1)
    print(f"\n📊 前向结果:", flush=True)
    print(f"   LM loss: {metrics['lm']:.4f}", flush=True)
    print(f"   World loss: {metrics['world']:.4f}", flush=True)
    print(f"   Conf loss: {metrics['conf']:.4f}", flush=True)
    print(f"   Load balance: {metrics['load_balance']:.4f}", flush=True)
    print(f"   专家使用率: {[f'{u:.1%}' for u in usage]}", flush=True)

    loss.backward()
    print(f"\n   Backward: ✅", flush=True)
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   梯度范数: {grad_norm:.2f}", flush=True)

    # 生成测试
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
    print("\n✨ 生成测试:", flush=True)
    seed_ids = tok.encode("[THOUGHT] check\n[CMD] ").ids
    for i in range(3):
        gen = model.generate(seed_ids, 40, 0.85, explore=True)
        txt = tok.decode(gen)
        print(f"   第 {i+1} 次: {repr(txt[:60])}", flush=True)

    print("\n✅ ModelCluster OK", flush=True)


if __name__ == "__main__":
    test()
