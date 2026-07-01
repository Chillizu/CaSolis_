"""
WorldModel V5 — 确定性 GRU 状态转移模型

基于 LoopWM 的设计理念, 去掉 RSSM 的 stochastic latent (确定环境无用):
- Prelude: 编码 state + action → conditioning signal
- GRU: 处理时序依赖
- 三个输出头: next_state, reward, continue
- CounterfactualHead (optional): 反事实模拟

参数量: ~370K (vs V4 128K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ActionEncoder(nn.Module):
    """
    动作编码: 3 类 meta intent + 参数 → 32-dim
    """
    def __init__(self, intent_dim: int = 3, param_dim: int = 29):
        super().__init__()
        self.intent_embed = nn.Embedding(intent_dim, intent_dim * 4)  # 3→12
        self.intent_out = nn.Linear(intent_dim * 4, intent_dim)
        self.param_proj = nn.Linear(384, param_dim)  # 对 MiniLM 文本编码投影
        self.param_bias = nn.Parameter(torch.zeros(param_dim))

    def forward(self, intent: torch.Tensor,
                state_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            intent: (B,) long tensor, 0=OBSERVE 1=CREATE 2=TRY
            state_emb: (B, 384) optional, 用于参数字典编码
        Returns:
            action_emb: (B, 32)
        """
        # Intent embedding
        i = self.intent_embed(intent)  # (B, 12)
        i = self.intent_out(i)  # (B, 3)

        # Param embedding (fallback to bias if no state)
        if state_emb is not None:
            p = F.relu(self.param_proj(state_emb))  # (B, 29)
        else:
            p = self.param_bias.expand(intent.size(0), -1)  # (B, 29)

        return torch.cat([i, p], dim=-1)  # (B, 32)


class WorldModelV5Core(nn.Module):
    """
    世界模型核心: prelude + GRU + 预测头

    Param count: ~370K
    """
    def __init__(self,
                 state_dim: int = 384,
                 action_dim: int = 32,
                 cond_dim: int = 80,
                 hidden_dim: int = 160):
        super().__init__()
        # Prelude: state + action → conditioning signal
        self.prelude = nn.Sequential(
            nn.Linear(state_dim + action_dim, cond_dim * 2),  # 416→160
            nn.ReLU(),
            nn.Linear(cond_dim * 2, cond_dim),  # 160→80
        )

        # State projection for GRU input
        self.state_proj = nn.Linear(state_dim, cond_dim)  # 384→80

        # GRU recurrent core
        self.gru = nn.GRUCell(cond_dim * 2, hidden_dim)  # 160→160

        # Prediction heads
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),  # 160→384
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.continue_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_emb: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            state: (B, 384) — MiniLM embedding of current state text
            action_emb: (B, 32) — encoded action
            hidden: (B, 192) or None — previous GRU hidden state
        Returns:
            dict with keys: next_state, reward, cont, hidden
        """
        B = state.size(0)

        # Conditioning signal
        cond_input = torch.cat([state, action_emb], dim=-1)  # (B, 416)
        e = self.prelude(cond_input)  # (B, 128)

        # Encode state for GRU
        s_enc = self.state_proj(state)  # (B, 128)

        # GRU input: state encoding + conditioning
        gru_input = torch.cat([s_enc, e], dim=-1)  # (B, 256)

        # Initialize hidden if needed
        if hidden is None:
            hidden = torch.zeros(B, self.gru.hidden_size,
                                 device=state.device)

        # GRU step
        hidden = self.gru(gru_input, hidden)  # (B, 192)

        # Predictions
        return {
            "next_state": self.next_state_head(hidden),  # (B, 384)
            "reward": self.reward_head(hidden),  # (B, 1)
            "cont": torch.sigmoid(self.continue_head(hidden)),  # (B, 1)
            "hidden": hidden,
        }


class WorldModelV5:
    """
    WorldModel V5 封装: 简化动作编码 + 训练 + 推理

    用法:
        wm = WorldModelV5()
        result = wm.step(state_emb, intent, params)
        wm.train_step(states, actions, next_states, rewards, continues)
    """

    def __init__(self,
                 state_dim: int = 384,
                 action_dim: int = 32,
                 cond_dim: int = 80,
                 hidden_dim: int = 160,
                 lr: float = 1e-3):
        self.action_encoder = ActionEncoder()
        self.core = WorldModelV5Core(
            state_dim=state_dim,
            action_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )
        self.optimizer = torch.optim.Adam(
            list(self.action_encoder.parameters()) +
            list(self.core.parameters()),
            lr=lr,
        )
        self.hidden: Optional[torch.Tensor] = None
        self.train_losses: list[float] = []
        self._param_count = self._count_params()

    def _count_params(self) -> int:
        total = 0
        for p in list(self.action_encoder.parameters()) + \
                 list(self.core.parameters()):
            total += p.numel()
        return total

    def encode_action(self, intent: int,
                      state_emb: torch.Tensor) -> torch.Tensor:
        """单步动作编码"""
        intent_t = torch.tensor([intent], dtype=torch.long)
        return self.action_encoder(intent_t, state_emb)

    def step(self, state_emb: torch.Tensor,
             intent: int,
             reset_hidden: bool = False) -> dict:
        """
        单步预测

        Args:
            state_emb: (384,) — 当前状态嵌入
            intent: 0/1/2 — OBSERVE/CREATE/TRY
            reset_hidden: 是否重置 GRU 隐藏状态
        Returns:
            dict with next_state, reward, cont
        """
        if reset_hidden:
            self.hidden = None

        s = state_emb.unsqueeze(0)  # (1, 384)
        a = self.encode_action(intent, s)  # (1, 32)
        result = self.core(s, a, self.hidden)
        self.hidden = result["hidden"]
        return {k: v.squeeze(0) for k, v in result.items()}

    def train_step(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   next_states: torch.Tensor,
                   rewards: torch.Tensor,
                   continues: torch.Tensor,
                   chunk_size: int = 32) -> float:
        """训练一步, 截断 BPTT"""
        T = states.size(0)
        total_loss = 0.0
        n_chunks = max(1, T // chunk_size)

        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, T)
            hidden = None
            pred_s, pred_r, pred_c = [], [], []

            for t in range(start, end):
                s = states[t:t+1].detach()
                a = actions[t:t+1].detach()
                r = self.core(s, a, hidden)
                hidden = r["hidden"].detach()
                pred_s.append(r["next_state"])
                pred_r.append(r["reward"])
                pred_c.append(r["cont"])

            ps = torch.cat(pred_s, dim=0)
            pr = torch.cat(pred_r, dim=0)
            pc = torch.cat(pred_c, dim=0)

            l_s = F.mse_loss(ps, next_states[start:end])
            l_r = F.mse_loss(pr, rewards[start:end])
            l_c = F.binary_cross_entropy(pc.squeeze(-1),
                                          continues[start:end].squeeze(-1))
            loss = l_s + 0.1 * l_r + 0.1 * l_c

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.action_encoder.parameters()) +
                list(self.core.parameters()), 5.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_chunks
        self.train_losses.append(avg_loss)
        return avg_loss

    def save(self, path: str):
        torch.save({
            "action_encoder": self.action_encoder.state_dict(),
            "core": self.core.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "losses": self.train_losses,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.action_encoder.load_state_dict(data["action_encoder"])
        self.core.load_state_dict(data["core"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.train_losses = data.get("losses", [])
