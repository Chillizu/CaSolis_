"""
世界模型 V2 — 输出属性预测器

给定 (state_embedding, intent) → 预测 (success, length, error_flag)
预测误差 = 好奇心信号 (0~1)

V1 的问题: 预测 384-dim MiniLM 输出嵌入, 所有输出嵌入塌缩到相近空间
V2 改进: 预测离散/可量化的输出属性, 产生有意义的好奇心
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelNet(nn.Module):
    """
    世界模型网络 V2

    输入: state_emb (384) + intent_onehot (N) = 384+N
    输出: [exit_code(2), length_bucket(3), has_error(2)]
          = 7 dims (用 softmax + sigmoid, 非回归)
    """

    def __init__(self, embed_dim: int = 384, n_intents: int = 14,
                 hidden_dim: int = 128):
        super().__init__()
        input_dim = embed_dim + n_intents
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Head 1: exit code (成功 vs 失败)
        self.exit_head = nn.Linear(hidden_dim, 2)
        # Head 2: output length bucket (短/中/长)
        self.length_head = nn.Linear(hidden_dim, 3)
        # Head 3: error keyword flag (含错误信息 vs 无)
        self.error_head = nn.Linear(hidden_dim, 2)

    def forward(self, state_emb: torch.Tensor, intent_onehot: torch.Tensor):
        x = torch.cat([state_emb, intent_onehot], dim=-1)
        h = self.shared(x)
        logits_exit = self.exit_head(h)
        logits_len = self.length_head(h)
        logits_err = self.error_head(h)
        return logits_exit, logits_len, logits_err


class WorldModel:
    """
    世界模型 V2

    好奇心 = 预测错误率 (而非 MSE)
    当世界模型预测 "exit=0, 短输出, 无错误" 但实际得到 "exit=1, 错误信息"
    → 高好奇心
    """

    def __init__(self, embed_dim: int = 384, n_intents: int = 14,
                 lr: float = 3e-4, device: str = "cpu"):
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

    def _extract_output_features(self, output_text: str, exit_code: int) -> torch.Tensor:
        """从原始输出提取分类目标: [exit_class, length_class, error_class]"""
        # exit_code: 0=成功, 1=失败
        exit_cls = 0 if exit_code == 0 else 1

        # length: 0=短(<100), 1=中(100-1000), 2=长(>1000)
        n = len(output_text)
        length_cls = 0 if n < 100 else (1 if n < 1000 else 2)

        # error keywords
        lower = output_text.lower()
        has_error = 1 if any(kw in lower for kw in ("not found", "error", "denied", "no such file", "not installed")) else 0

        return torch.tensor([exit_cls, length_cls, has_error], device=self.device)

    def _compute_prediction_error(self, logits: tuple, targets: torch.Tensor) -> float:
        """预测误差 = 3个分类头的平均交叉熵损失"""
        logits_exit, logits_len, logits_err = logits
        loss = (
            F.cross_entropy(logits_exit, targets[0:1])
            + F.cross_entropy(logits_len, targets[1:2])
            + F.cross_entropy(logits_err, targets[2:3])
        ) / 3.0
        return loss.item()

    @torch.no_grad()
    def compute_curiosity(self, state_emb: torch.Tensor,
                          intent_idx: int,
                          output_text: str = "",
                          exit_code: int = 0) -> float:
        """
        计算好奇心: 世界模型预测 vs 实际输出属性的误差

        Returns: 0~1 归一化的预测误差
        """
        self.predictor.eval()

        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)

        intent_onehot = self._intent_to_onehot(intent_idx)
        logits = self.predictor(state_emb, intent_onehot)
        targets = self._extract_output_features(output_text, exit_code)

        error = self._compute_prediction_error(logits, targets)

        # 自适应归一化
        self.running_errors.append(error)
        if len(self.running_errors) > 50:
            self.running_errors.pop(0)
            self.max_error = max(self.running_errors) + 1e-8

        curiosity = min(error / self.max_error, 1.0)
        return curiosity

    def update(self, state_emb: torch.Tensor, intent_idx: int,
               output_text: str, exit_code: int) -> float:
        """单步更新世界模型"""
        self.predictor.train()
        self.optimizer.zero_grad()

        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)

        intent_onehot = self._intent_to_onehot(intent_idx)
        logits_exit, logits_len, logits_err = self.predictor(state_emb, intent_onehot)
        targets = self._extract_output_features(output_text, exit_code)

        loss = (
            F.cross_entropy(logits_exit, targets[0:1])
            + F.cross_entropy(logits_len, targets[1:2])
            + F.cross_entropy(logits_err, targets[2:3])
        ) / 3.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        self.predictor.eval()
        return loss.item()

    def train_on_buffer(self, buffer_samples: list) -> float:
        """从缓冲区采样训练"""
        if not buffer_samples:
            return 0.0

        state_embs, intent_idxs, outputs, exit_codes = [], [], [], []
        for s in buffer_samples:
            if all(k in s for k in ("state_emb", "intent_id", "output", "exit_code")):
                state_embs.append(s["state_emb"])
                intent_idxs.append(s["intent_id"])
                outputs.append(s["output"])
                exit_codes.append(s["exit_code"])

        if len(state_embs) < 4:
            return 0.0

        s = torch.stack(state_embs).to(self.device)
        i_onehot = torch.zeros(len(intent_idxs), self.n_intents, device=self.device)
        for idx, val in enumerate(intent_idxs):
            i_onehot[idx, val if val < self.n_intents else 0] = 1.0

        # 批量提取目标
        targets = torch.stack([
            self._extract_output_features(outputs[i], exit_codes[i])
            for i in range(len(outputs))
        ]).to(self.device)

        self.predictor.train()
        self.optimizer.zero_grad()

        le, ll, lr = self.predictor(s, i_onehot)
        loss = (
            F.cross_entropy(le, targets[:, 0])
            + F.cross_entropy(ll, targets[:, 1])
            + F.cross_entropy(lr, targets[:, 2])
        ) / 3.0

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
