"""
SlotMind — K-slot 工作记忆架构

不是另一个 LLM 变体，是完全不同的范式：

核心思想：
  ▸ K 个独立思维槽（slot），每个跟踪一条思路
  ▸ 每个槽有自己的隐藏状态和动力学
  ▸ 槽之间通过跨槽注意力通信
  ▸ 路由门控选择：这次用哪个槽的输出？
  ▸ 创造机制：噪音注入 + 非最优选择

对比传统 LLM:
  LLM:      h_t = f(h_{t-1}, x_t)          ← 一条路
  SlotMind: h_k = f_k(h_k, x)              ← K 条路并行
            输出 = Router([h_0, h_1, ..., h_K])  ← 选最优
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, random

# ─── 常量 ─────────────────────────────────────
V = 2022       # 词表大小（标准 token）
TV = V + 19    # 总词表（含特殊 token）
PAD = TV - 1   # padding token

# ─── 槽单元 ───────────────────────────────────
class SlotCell(nn.Module):
    """
    一个思维槽。
    每个槽有自己的状态 h_slot ∈ R^d。
    输入: x_in  (共享输入)
          h_slot (当前状态)
    输出: h_slot' (新状态)

    每个槽学不同的动力学参数（不是共享的！）
    """
    def __init__(self, d_input, d_slot):
        super().__init__()
        self.d_slot = d_slot
        # 槽特有的状态更新
        self.gate = nn.Linear(d_input + d_slot, d_slot)
        self.update = nn.Linear(d_input + d_slot, d_slot)

    def forward(self, x, h):
        """x: (B, d_input), h: (B, d_slot) → h': (B, d_slot)"""
        inp = torch.cat([x, h], dim=-1)
        g = torch.sigmoid(self.gate(inp))
        u = torch.tanh(self.update(inp))
        return h + g * u  # 残差更新


# ─── 路由门控 ─────────────────────────────────
class Router(nn.Module):
    """
    路由门控 — 核心决策机制。

    看 K 个槽的状态 + 当前输入
    决定：
      1. 哪个槽的输出去生成 token
      2. 这次探索还是利用

    创造力来自：有时不选「最优」槽，选「有趣」的
    """
    def __init__(self, d_input, d_slot, K):
        super().__init__()
        self.proj = nn.Linear(d_input + K * d_slot, K)
        self.temp = nn.Parameter(torch.ones(1))  # 可学习温度

    def forward(self, x, slots, explore=False):
        """
        x: (B, d_input)
        slots: (B, K, d_slot)
        Returns: (B, K) slot 选择概率
        """
        B = x.shape[0]
        flat = torch.cat([x] + [slots[:, k] for k in range(slots.shape[1])], dim=-1)
        logits = self.proj(flat)

        if explore:
            # 探索模式：加噪声到 logits
            noise = torch.randn_like(logits) * 0.5
            logits = logits + noise

        return F.softmax(logits / self.temp.clamp(min=0.1), dim=-1)


# ─── 跨槽注意力 ──────────────────────────────
class CrossSlotAttention(nn.Module):
    """
    槽之间互相看。
    每个槽从其他槽获取信息——「思路之间的对话」
    """
    def __init__(self, d_slot, K):
        super().__init__()
        self.q_proj = nn.Linear(d_slot, d_slot)
        self.k_proj = nn.Linear(d_slot, d_slot)
        self.v_proj = nn.Linear(d_slot, d_slot)
        self.out_proj = nn.Linear(d_slot, d_slot)

    def forward(self, slots):
        """
        slots: (B, K, d_slot)
        Returns: (B, K, d_slot) 通信后的槽
        """
        B, K, d = slots.shape
        Q = self.q_proj(slots)  # (B, K, d)
        K_proj = self.k_proj(slots)
        V = self.v_proj(slots)

        # 点积注意力
        attn = torch.bmm(Q, K_proj.transpose(1, 2)) / (d ** 0.5)
        attn = F.softmax(attn, dim=-1)  # (B, K, K)

        out = torch.bmm(attn, V)       # (B, K, d)
        out = self.out_proj(out)
        return out


# ─── SlotMind 主模型 ──────────────────────────
class SlotMind(nn.Module):
    """
    完整架构：

    Input → Embed → [K Slots] → CrossAttn → Router → Output

    K 个槽并行跟踪不同「思路」。
    路由门控决定哪个槽的输出用于生成 token。
    创造机制：噪音注入 + epsilon-贪婪槽选择。
    """
    def __init__(self, d_slot=384, K=4, noise_slot=1):
        """
        d_slot: 每个槽的隐藏维度
        K: 槽的数量
        noise_slot: 哪个槽是「噪声槽」（高创造性）
        """
        super().__init__()
        self.K = K
        self.d_slot = d_slot
        self.noise_slot = noise_slot

        self.embed = nn.Embedding(TV, d_slot)

        # K 个独立槽，每个有自己的动力学
        self.slots = nn.ModuleList([
            SlotCell(d_slot, d_slot) for _ in range(K)
        ])

        # 跨槽通信
        self.cross_attn = CrossSlotAttention(d_slot, K)

        # 路由门控
        self.router = Router(d_slot, d_slot, K)

        # 输出头（每个槽有自己的，不共享）
        self.out_heads = nn.ModuleList([
            nn.Linear(d_slot, V) for _ in range(K)
        ])

        # 噪声槽的噪音注入强度
        self.noise_scale = nn.Parameter(torch.tensor(0.3))

        # 槽初始状态
        self.slot_init = nn.Parameter(torch.randn(K, d_slot) * 0.02)

    def forward(self, tokens, explore_rate=0.0):
        """
        单步训练。

        tokens: (B, S)
        explore_rate: 在训练中探索的概率

        Returns:
          lm_loss, world_loss, slot_usage, stats
        """
        B, S = tokens.shape
        d = self.d_slot

        # 初始化 K 个槽
        h = self.slot_init.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        total_loss = 0.0
        total_usage = torch.zeros(self.K)
        n_steps = 0

        for t in range(S - 1):
            # 1. Embed 当前 token
            x = self.embed(tokens[:, t])  # (B, d)

            # 2. 更新每个槽
            h_new = []
            for k, slot_cell in enumerate(self.slots):
                h_k = h[:, k]  # (B, d)

                # 噪声槽：注入随机噪声
                if k == self.noise_slot:
                    noise = torch.randn_like(h_k) * self.noise_scale.sigmoid()
                    h_k = h_k + noise * 0.1

                h_k_new = slot_cell(x, h_k)
                h_new.append(h_k_new)

            h_new = torch.stack(h_new, dim=1)  # (B, K, d)

            # 3. 跨槽通信
            h_new = self.cross_attn(h_new) + h_new  # 残差

            # 4. 路由门控：选择槽
            explore = random.random() < explore_rate
            route = self.router(x, h_new, explore=explore)  # (B, K)

            # 对于训练，我们软路由（加权平均所有槽）
            target_token = tokens[:, t + 1]

            loss_t = 0.0
            for k in range(self.K):
                logits_k = self.out_heads[k](h_new[:, k])  # (B, V)
                # 只对 < V 的 token 算 loss
                valid = target_token < V
                if valid.sum() > 0:
                    loss_k = F.cross_entropy(
                        logits_k[valid],
                        target_token[valid]
                    )
                    # 加权：路由权重越高，这个 loss 越重要
                    weight = route[:, k].detach().mean()
                    loss_t = loss_t + weight * loss_k

                total_usage[k] += route[:, k].detach().mean().item()

            total_loss = total_loss + loss_t
            n_steps += 1

            # 更新槽状态（下一步用）
            h = h_new

        lm_loss = total_loss / max(n_steps, 1)
        slot_usage = (total_usage / max(n_steps, 1)).tolist()
        usage_entropy = sum(-u * math.log(u + 1e-8) for u in slot_usage if u > 0)

        return lm_loss, slot_usage, usage_entropy

    @torch.no_grad()
    def generate(self, seed_ids, n=60, temp=0.85, explore=True):
        """
        自回归生成。

        关键：每次 step，路由器选择「最优」或「探索」槽。
        每次生成走不同的思考路径 → 每次不一样。
        """
        out = list(seed_ids)
        B = 1
        d = self.d_slot
        self.eval()

        # 初始化槽
        h = self.slot_init.unsqueeze(0).expand(B, -1, -1)

        for step in range(n):
            # 取最后一个 token
            x = self.embed(torch.tensor([[out[-1]]]).long())  # (1, 1, d)

            # 更新所有槽
            h_new = []
            for k, slot_cell in enumerate(self.slots):
                h_k = h[:, k]
                # 噪声槽
                if k == self.noise_slot and explore:
                    noise = torch.randn_like(h_k) * self.noise_scale.sigmoid()
                    h_k = h_k + noise * 0.1
                h_k_new = slot_cell(x[:, 0], h_k)
                h_new.append(h_k_new)

            h_new = torch.stack(h_new, dim=1)
            h_new = self.cross_attn(h_new) + h_new

            # 路由
            route = self.router(x[:, 0], h_new, explore=explore)  # (1, K)

            if explore:
                # 有时选非最大概率的槽
                if random.random() < 0.3:
                    # 按路由分布采样
                    k = torch.multinomial(route[0], 1).item()
                else:
                    k = route[0].argmax().item()
            else:
                k = route[0].argmax().item()

            # 用选中的槽生成
            logits = self.out_heads[k](h_new[0, k])  # (V,)
            lp = F.softmax(logits / temp, dim=-1)
            nt = torch.multinomial(lp, 1).item()
            if nt >= V:
                break
            out.append(nt)

            h = h_new

        self.train()
        return out


def test():
    print("🧪 SlotMind — K-slot Working Memory", flush=True)
    model = SlotMind(d_slot=384, K=4, noise_slot=1)
    total = sum(p.numel() for p in model.parameters())
    print(f"   参数: {total:,} ({total/1e6:.1f}M)", flush=True)

    B, S = 4, 32
    tokens = torch.randint(10, min(100, V), (B, S))

    loss, usage, ent = model(tokens, explore_rate=0.1)
    print(f"   Forward: LM={loss.item():.4f}", flush=True)
    print(f"   槽使用率: {[f'{u:.2%}' for u in usage]}", flush=True)
    print(f"   槽熵: {ent:.3f}", flush=True)
    loss.backward()
    print(f"   Backward: ✅", flush=True)
    print(f"   梯度范数: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.2f}", flush=True)

    # 生成测试
    print("\n   ✨ 生成测试:", flush=True)
    seed = torch.randint(10, 100, (1, 5)).tolist()[0]
    for i in range(3):
        gen = model.generate(seed, n=40, explore=True)
        text = repr(gen[:20])
        print(f"   第 {i+1} 次: {text}", flush=True)

    print("✅ SlotMind OK", flush=True)


if __name__ == "__main__":
    test()
