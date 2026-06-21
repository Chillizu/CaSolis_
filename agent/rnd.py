"""
RND (Random Network Distillation) — 好奇心模块

架构:
  Target Network:   随机初始化, 冻结 → 输出特征
  Predictor Network: 相同架构, 可训练 → 预测目标输出
  RND Error = MSE(predictor(state), target(state)) → 新颖度分数
  
高误差 = 新颖状态 → 探索奖励
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDNetwork(nn.Module):
    """RND 子网络: MLP(embed_dim → 128 → 128 → feature_dim)"""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class RND:
    """
    RND 好奇心模块
    
    Usage:
      rnd = RND(embed_dim=384)
      emb = encoder.encode(state_text, convert_to_tensor=True)
      novelty = rnd.compute_novelty(emb)
      rnd.update(emb)  # 仅训练 predictor
    """

    def __init__(self, embed_dim: int = 384, lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.target = RNDNetwork(embed_dim).to(device)
        self.predictor = RNDNetwork(embed_dim).to(device)

        # 冻结 target
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr)
        self.running_errors: list[float] = []
        self.max_error = 1.0  # adaptive normalization

    @torch.no_grad()
    def compute_novelty(self, embeddings: torch.Tensor) -> float:
        """
        计算新颖度分数 (0~1)
        Higher = more novel
        """
        self.predictor.eval()
        with torch.no_grad():
            target_feat = self.target(embeddings)
            pred_feat = self.predictor(embeddings)
            error = F.mse_loss(pred_feat, target_feat, reduction="mean").item()

        # 自适应归一化
        self.running_errors.append(error)
        if len(self.running_errors) > 100:
            self.running_errors.pop(0)
            self.max_error = max(self.running_errors) + 1e-8

        novelty = min(error / self.max_error, 1.0)
        return novelty

    def update(self, embeddings: torch.Tensor):
        """
        用当前状态训练 predictor (使其更好地预测 target)
        只在低新颖度状态上训练 (已熟悉的状态 → predictor 更准)
        """
        self.predictor.train()
        self.optimizer.zero_grad()

        target_feat = self.target(embeddings)
        pred_feat = self.predictor(embeddings)
        loss = F.mse_loss(pred_feat, target_feat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def reset(self):
        """重置新颖度: 清空历史误差, 重新初始化 predictor 权重"""
        self.running_errors.clear()
        self.max_error = 1.0
        # 重新初始化 predictor (遗忘旧的状态分布)
        for layer in self.predictor.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=1e-3)

    def soft_reset(self, factor: float = 0.5):
        """软重置: 降低 max_error, 让已熟悉的状态重新变得 mildly novel"""
        self.max_error *= factor
        if self.max_error < 0.1:
            self.max_error = 1.0

    def get_novelty_stats(self) -> dict:
        """返回新颖度统计"""
        return {
            "current_max_error": self.max_error,
            "running_errors_avg": (sum(self.running_errors) / len(self.running_errors)
                                   if self.running_errors else 0),
            "running_errors_count": len(self.running_errors),
        }
