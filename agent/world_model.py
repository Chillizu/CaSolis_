"""
世界模型 — 基于 MiniLM 嵌入空间的输出预测器

给定 (state_embedding, intent) → 预测 (output_embedding)
预测误差 MSE = 好奇心信号

比 RND 更强: 不仅知道"这个状态见过吗", 还知道"这个命令的输出和预期一样吗?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModelNet(nn.Module):
    """
    世界模型网络

    输入: state_emb (384) + intent_onehot (N) = 384+N
    输出: predicted_output_emb (384)
    """

    def __init__(self, embed_dim: int = 384, n_intents: int = 12,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = embed_dim + n_intents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, state_emb: torch.Tensor, intent_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, intent_onehot], dim=-1)
        return self.net(x)


class WorldModel:
    """
    世界模型封装

    Usage:
      wm = WorldModel()
      curiosity = wm.compute_curiosity(state_emb, intent_idx, output_emb)
      wm.update(state_emb, intent_idx, output_emb)
      wm.train_step(buffer_sample)
    """

    def __init__(self, embed_dim: int = 384, n_intents: int = 12,
                 lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.n_intents = n_intents
        self.predictor = WorldModelNet(embed_dim, n_intents=n_intents).to(device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(), lr=lr, weight_decay=1e-4
        )
        self.running_errors: list[float] = []
        self.max_error = 1.0

    def _intent_to_onehot(self, intent_idx: int) -> torch.Tensor:
        onehot = torch.zeros(1, self.n_intents, device=self.device)
        onehot[0, intent_idx] = 1.0
        return onehot

    @torch.no_grad()
    def compute_curiosity(self, state_emb: torch.Tensor,
                          intent_idx: int,
                          output_emb: torch.Tensor) -> float:
        """
        计算好奇心: 世界模型预测输出 vs 实际输出的误差

        Args:
            state_emb: (384,) MiniLM 状态嵌入
            intent_idx: 意图索引 (0-8)
            output_emb: (384,) MiniLM 输出嵌入

        Returns:
            curiosity: 0~1 归一化的预测误差
        """
        self.predictor.eval()

        # 确保是 2D
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        if output_emb.dim() == 1:
            output_emb = output_emb.unsqueeze(0)

        intent_onehot = self._intent_to_onehot(intent_idx)
        pred_output = self.predictor(state_emb, intent_onehot)

        error = F.mse_loss(pred_output, output_emb, reduction="mean").item()

        # 自适应归一化
        self.running_errors.append(error)
        if len(self.running_errors) > 100:
            self.running_errors.pop(0)
            self.max_error = max(self.running_errors) + 1e-8

        curiosity = min(error / self.max_error, 1.0)
        return curiosity

    def update(self, state_emb: torch.Tensor, intent_idx: int,
               output_emb: torch.Tensor) -> float:
        """单步更新世界模型 (训练 predictor)"""
        self.predictor.train()
        self.optimizer.zero_grad()

        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        if output_emb.dim() == 1:
            output_emb = output_emb.unsqueeze(0)

        intent_onehot = self._intent_to_onehot(intent_idx)
        pred_output = self.predictor(state_emb, intent_onehot)
        loss = F.mse_loss(pred_output, output_emb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        self.predictor.eval()
        return loss.item()

    def train_on_buffer(self, buffer_samples: list) -> float:
        """从缓冲区采样训练世界模型"""
        if not buffer_samples:
            return 0.0

        # 准备数据
        state_embs = []
        intent_idxs = []
        output_embs = []

        for sample in buffer_samples:
            s_emb = sample.get("state_emb")
            i_idx = sample.get("intent_id")
            o_emb = sample.get("output_emb")
            if s_emb is not None and i_idx is not None and o_emb is not None:
                state_embs.append(s_emb)
                intent_idxs.append(i_idx)
                output_embs.append(o_emb)

        if not state_embs:
            return 0.0

        s = torch.stack(state_embs).to(self.device)
        o = torch.stack(output_embs).to(self.device)
        i_onehot = torch.zeros(len(intent_idxs), self.n_intents, device=self.device)
        for idx, val in enumerate(intent_idxs):
            i_onehot[idx, val] = 1.0

        self.predictor.train()
        self.optimizer.zero_grad()

        pred = self.predictor(s, i_onehot)
        loss = F.mse_loss(pred, o)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        self.predictor.eval()
        return loss.item()

    def stats(self) -> dict:
        return {
            "running_errors_avg": (sum(self.running_errors) / len(self.running_errors)
                                   if self.running_errors else 0),
            "max_error": self.max_error,
            "n_samples": len(self.running_errors),
        }
