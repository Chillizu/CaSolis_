"""
WorldModel V5.1 — 随机隐变量 RSSM Lite

V5 基础上增加:
- 随机隐变量 z_t (16 categorical × 16 classes = 256 states)
- Prior (闭眼预测) vs Posterior (睁眼后验)
- KL 散度 + free bits + KL balancing
- 预测误差 = KL 散度 → 更稳定的惊奇度信号

参数量: ~410K (vs V5 ~370K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """Straight-through categorical sampling"""
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    hard = F.one_hot(dist.sample(), num_classes=probs.size(-1)).float()
    return hard + probs - probs.detach()


def _kl_divergence(post_logits: torch.Tensor,
                   prior_logits: torch.Tensor,
                   free_bits: float = 0.5) -> torch.Tensor:
    """
    KL(Prior || Posterior) with free bits
    
    Returns: KL per latent element, floored at free_bits
    """
    post_probs = F.softmax(post_logits, dim=-1)
    prior_probs = F.softmax(prior_logits, dim=-1)
    kl = (post_probs * (post_probs.log() - prior_probs.log())).sum(dim=-1)
    # Free bits: floor control, not zero
    kl = kl.clamp(min=free_bits)
    return kl  # (B,)


class StochasticLatent(nn.Module):
    """
    RSSM 随机隐变量:
    - Prior: 从 h_t 预测 z_t 分布 (闭眼)
    - Posterior: 从 h_t + state 预测 z_t 分布 (睁眼)
    - 16 个 categorical × 16 classes = 256 种隐状态
    """
    def __init__(self, hidden_dim: int = 160,
                 state_dim: int = 384,
                 n_categories: int = 16,
                 n_classes: int = 16):
        super().__init__()
        self.n_categories = n_categories
        self.n_classes = n_classes
        latent_dim = n_categories * n_classes  # 256

        # Prior: h_t → logits (闭眼猜测)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Posterior: h_t + state → logits (睁眼修正)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Latent projection: z_t → conditioning for decoder
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, h_t: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            h_t: (B, hidden_dim) — GRU 隐状态
            state: (B, state_dim) — 实际 state (训练时提供, 推理时 None)
        Returns:
            dict: prior_logits, post_logits (or None), z, latent_emb
        """
        B = h_t.size(0)
        prior_logits = self.prior_net(h_t)  # (B, 256)
        prior_logits = prior_logits.view(B, self.n_categories, self.n_classes)

        post_logits = None
        z = None
        if state is not None:
            # Training mode: use posterior
            post_in = torch.cat([h_t, state], dim=-1)
            post_logits = self.posterior_net(post_in)  # (B, 256)
            post_logits = post_logits.view(B, self.n_categories, self.n_classes)
            z = _sample_categorical(post_logits)  # (B, 16, 16)
        else:
            # Inference mode: use prior
            z = _sample_categorical(prior_logits)  # (B, 16, 16)

        # Flatten z for decoder
        z_flat = z.view(B, -1)  # (B, 256)
        latent_emb = self.latent_proj(z_flat)  # (B, hidden_dim)
        return {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "z": z,
            "latent_emb": latent_emb,
        }


class RSSMCore(nn.Module):
    """
    RSSM Lite Core: 确定性 GRU + 随机隐变量
    
    架构:
    state + action → Prelude → GRU → h_t
    h_t → Prior → z_logits (闭眼)
    h_t + state → Posterior → z_logits (睁眼, 训练用)
    h_t + z_t → Decoder → next_state, reward, cont
    """
    def __init__(self,
                 state_dim: int = 384,
                 action_dim: int = 32,
                 cond_dim: int = 80,
                 hidden_dim: int = 160,
                 n_categories: int = 16,
                 n_classes: int = 16,
                 n_fact_categories: int = 20):
        super().__init__()
        # Prelude: state + action → conditioning signal
        self.prelude = nn.Sequential(
            nn.Linear(state_dim + action_dim, cond_dim * 2),
            nn.ReLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # State projection for GRU input
        self.state_proj = nn.Linear(state_dim, cond_dim)

        # GRU recurrent core
        self.gru = nn.GRUCell(cond_dim * 2, hidden_dim)

        # Stochastic latent (16×16)
        self.stochastic = StochasticLatent(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            n_categories=n_categories,
            n_classes=n_classes,
        )

        # Prediction heads — input = h_t + latent_emb
        dec_in_dim = hidden_dim + hidden_dim  # 320
        self.next_state_head = nn.Sequential(
            nn.Linear(dec_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.reward_head = nn.Linear(dec_in_dim, 1)
        self.continue_head = nn.Linear(dec_in_dim, 1)
        # P18 Phase 3: Fact prediction head — "which categories get new facts?"
        self.fact_head = nn.Linear(dec_in_dim, n_fact_categories)

    def forward(self, state: torch.Tensor, action_emb: torch.Tensor,
                hidden: Optional[torch.Tensor] = None,
                state_target: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            state: (B, 384)
            action_emb: (B, 32)
            hidden: (B, 160) or None
            state_target: (B, 384) — actual next_state for posterior (training)
        Returns:
            dict with keys: next_state, reward, cont, hidden,
                            prior_logits, post_logits, z, kl, latent_emb
        """
        B = state.size(0)

        # Conditioning signal
        cond_input = torch.cat([state, action_emb], dim=-1)  # (B, 416)
        e = self.prelude(cond_input)  # (B, 80)

        # Encode state for GRU
        s_enc = self.state_proj(state)  # (B, 80)

        # GRU input
        gru_input = torch.cat([s_enc, e], dim=-1)  # (B, 160)

        if hidden is None:
            hidden = torch.zeros(B, self.gru.hidden_size, device=state.device)

        hidden = self.gru(gru_input, hidden)  # (B, 160)

        # Stochastic latent
        latent = self.stochastic(hidden, state=state_target)
        z = latent["z"]
        latent_emb = latent["latent_emb"]

        # Decoder input: h_t + z_t
        dec_in = torch.cat([hidden, latent_emb], dim=-1)  # (B, 320)

        # KL divergence (only when posterior is available)
        kl = None
        if latent["post_logits"] is not None:
            kl = _kl_divergence(
                latent["post_logits"], latent["prior_logits"],
                free_bits=0.5,
            )
            # KL balancing: α=0.8
            kl_loss = 0.8 * kl.mean() + 0.2 * kl.detach().mean()
        else:
            kl_loss = torch.tensor(0.0, device=state.device)

        return {
            "next_state": self.next_state_head(dec_in),
            "reward": self.reward_head(dec_in),
            "cont": torch.sigmoid(self.continue_head(dec_in)),
            "fact_pred": torch.sigmoid(self.fact_head(dec_in)),
            "hidden": hidden,
            "prior_logits": latent["prior_logits"],
            "post_logits": latent["post_logits"],
            "z": z,
            "kl_per_element": kl,
            "kl_loss": kl_loss,
            "latent_emb": latent_emb,
        }


class ActionEncoderV51(nn.Module):
    """动作编码: 3 类 meta intent → 32-dim (与 V5 相同)"""
    def __init__(self, intent_dim: int = 3, param_dim: int = 29):
        super().__init__()
        self.intent_embed = nn.Embedding(intent_dim, 16)
        self.param_proj = nn.Linear(param_dim, 16)
        self.param_bias = nn.Parameter(torch.zeros(param_dim))

    def forward(self, intent: torch.Tensor,
                state_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        i = self.intent_embed(intent)  # (B, 16)
        p = torch.sigmoid(self.param_proj(self.param_bias.unsqueeze(0)))
        return torch.cat([i, p.expand(intent.size(0), -1)], dim=-1)  # (B, 32)


class WorldModelV51:
    """
    WorldModel V5.1 — RSSM Lite
    
    用法:
    - step(): 单步预测 (推理, 只用 prior)
    - train_step(): 训练 (posterior + prior + KL)
    
    """
    def __init__(self,
                 state_dim: int = 384,
                 intent_dim: int = 3,
                 hidden_dim: int = 160,
                 n_categories: int = 16,
                 n_classes: int = 16,
                 n_fact_categories: int = 20,
                 lr: float = 1e-3):
        self.action_encoder = ActionEncoderV51()
        self.core = RSSMCore(
            state_dim=state_dim,
            action_dim=32,
            cond_dim=80,
            hidden_dim=hidden_dim,
            n_categories=n_categories,
            n_classes=n_classes,
            n_fact_categories=n_fact_categories,
        )
        self.optimizer = torch.optim.Adam(
            list(self.action_encoder.parameters()) +
            list(self.core.parameters()),
            lr=lr,
        )
        self.hidden = None
        self.train_losses: list[float] = []
        self.kl_scale = 0.1  # β for KL loss
        self._param_count = self._count_params()
        self.last_kl_value = 0.0

    def _count_params(self) -> int:
        total = 0
        for p in list(self.action_encoder.parameters()) + list(self.core.parameters()):
            total += p.numel()
        return total

    def encode_action(self, intent: int,
                      state_emb: torch.Tensor) -> torch.Tensor:
        intent_t = torch.tensor([intent], dtype=torch.long)
        return self.action_encoder(intent_t, state_emb)

    def step(self, state_emb: torch.Tensor,
             intent: int,
             reset_hidden: bool = False) -> dict:
        """
        单步推理 — 只用 prior (不观测 state_target)
        
        Returns:
            dict: next_state, reward, cont, hidden, kl_per_element (None), z
        """
        if reset_hidden:
            self.hidden = None
        s = state_emb.unsqueeze(0)
        a = self.encode_action(intent, s)
        result = self.core(s, a, self.hidden, state_target=None)
        self.hidden = result["hidden"].detach()
        return {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == 1 else v
                for k, v in result.items() if k != "prior_logits" and k != "post_logits"}

    def train_step(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   next_states: torch.Tensor,
                   rewards: torch.Tensor,
                   continues: torch.Tensor,
                   fact_targets: Optional[torch.Tensor] = None,
                   chunk_size: int = 32) -> float:
        T = states.size(0)
        total_loss = 0.0
        n_chunks = max(1, T // chunk_size)

        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, T)
            hidden = None
            pred_s, pred_r, pred_c, pred_f = [], [], [], []
            kl_vals = []

            for t in range(start, end):
                s = states[t:t+1].detach()
                a = actions[t:t+1].detach()
                ns = next_states[t:t+1].detach()
                r = self.core(s, a, hidden, state_target=ns)
                hidden = r["hidden"].detach()
                pred_s.append(r["next_state"])
                pred_r.append(r["reward"])
                pred_c.append(r["cont"])
                pred_f.append(r["fact_pred"])
                kl_vals.append(r["kl_loss"])

            ps = torch.cat(pred_s, dim=0)
            pr = torch.cat(pred_r, dim=0)
            pc = torch.cat(pred_c, dim=0)
            pf = torch.cat(pred_f, dim=0)

            l_s = F.mse_loss(ps, next_states[start:end])
            l_r = F.mse_loss(pr, rewards[start:end])
            l_c = F.binary_cross_entropy(pc.squeeze(-1),
                                          continues[start:end].squeeze(-1))
            l_f = 0.0
            if fact_targets is not None:
                ft = fact_targets[start:end]
                if ft.size(-1) == pf.size(-1):
                    l_f = F.binary_cross_entropy(pf, ft)
            l_kl = sum(kl_vals) / max(len(kl_vals), 1)

            loss = l_s + 0.1 * l_r + 0.1 * l_c + self.kl_scale * l_kl + 0.05 * l_f

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.action_encoder.parameters()) +
                list(self.core.parameters()), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_chunks
        self.train_losses.append(avg_loss)
        self.last_kl_value = l_kl.item() if isinstance(l_kl, torch.Tensor) else l_kl
        return avg_loss
    def save(self, path: str):
        torch.save({
            "action_encoder": self.action_encoder.state_dict(),
            "core": self.core.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "kl_scale": self.kl_scale,
            "hidden": self.hidden,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.action_encoder.load_state_dict(data["action_encoder"])
        self.core.load_state_dict(data["core"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.train_losses = data.get("train_losses", [])
        self.kl_scale = data.get("kl_scale", 0.1)
        self.hidden = data.get("hidden", None)
