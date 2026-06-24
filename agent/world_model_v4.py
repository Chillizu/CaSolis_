"""
世界模型 V4 — 核+叶架构 (GrowingWorldModel)

人脑类比: 新皮质 — 核心处理通用模式, 每个意图有专属"皮质柱"

设计:
  WorldModelCore: 输入 state_emb + thought, 与意图数量解耦
  IntentLeaf:     每个意图专属的预测头 (exit/length/error/value/next_thought/agreement)
  GrowingWorldModel: core + ModuleDict[intent_name → IntentLeaf]

优点:
  - 新意图只加叶, 不改 core → 零干扰
  - 不同意图输出分布独立 → 训练稀疏也没问题
  - 可解释性强 → 每个叶的 loss 独立跟踪
  - 旧 V3 完整向前兼容 → 可以渐进切换
"""

import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelCore(nn.Module):
    """
    世界模型核心 — 与意图数量解耦

    输入: state_emb(384) + thought(16) = 400
    输出: shared_hidden(128)
    """
    def __init__(self, embed_dim: int = 384, thought_dim: int = 16,
                 hidden_dim: int = 128):
        super().__init__()
        input_dim = embed_dim + thought_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, state_emb: torch.Tensor,
                thought: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, thought], dim=-1)
        return self.net(x)


class IntentLeaf(nn.Module):
    """
    意图叶 — 每个意图专属的预测头

    输出:
      - exit: 成功/失败 (2)
      - length: 输出长度分档 (3)
      - error: 错误信号 (2)
      - value: 价值预测 (1)
      - next_thought: 下步思考 (16)
      - agreement: 自洽性直觉 (2)
    """
    def __init__(self, hidden_dim: int = 128, thought_dim: int = 16):
        super().__init__()
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.exit_head = nn.Linear(hidden_dim, 2)
        self.length_head = nn.Linear(hidden_dim, 3)
        self.error_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.next_thought_head = nn.Linear(hidden_dim, thought_dim)
        self.agreement_head = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.relu(self.shared(h))
        return {
            "exit": self.exit_head(h),
            "length": self.length_head(h),
            "error": self.error_head(h),
            "value": self.value_head(h).squeeze(-1),
            "next_thought": self.next_thought_head(h),
            "agreement": self.agreement_head(h),
        }


class GrowingWorldModel(nn.Module):
    """
    增长型世界模型 — 核 + ModuleDict 叶

    用法:
      wm = GrowingWorldModel()
      wm.add_intent("READ")
      wm.add_intent("CUSTOM")
      pred = wm(state_emb, thought, intent_name)
      # → {"exit": ..., "value": ..., "next_thought": ..., ...}
    """

    def __init__(self, embed_dim: int = 384, thought_dim: int = 16,
                 hidden_dim: int = 128):
        super().__init__()
        self.core = WorldModelCore(embed_dim, thought_dim, hidden_dim)
        self.leaves = nn.ModuleDict()
        self.thought_dim = thought_dim
        self.hidden_dim = hidden_dim

        # 训练状态
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)
        self.leaf_losses: dict[str, list[float]] = {}  # intent → loss history
        self.total_steps = 0

    def add_intent(self, name: str):
        """添加新意图叶 (可运行时调用)"""
        if name not in self.leaves:
            self.leaves[name] = IntentLeaf(self.hidden_dim, self.thought_dim)
            self.leaf_losses[name] = []
            # 重新创建 optimizer 以包含新参数
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=3e-4, weight_decay=1e-4
            )

    def has_intent(self, name: str) -> bool:
        return name in self.leaves

    def forward(self, state_emb: torch.Tensor, thought: torch.Tensor,
                intent_name: str) -> dict[str, torch.Tensor]:
        """
        前向: state_emb + thought → core → intent_leaf

        Args:
          state_emb: (batch, 384)
          thought: (batch, 16)
          intent_name: str — 意图名

        Returns:
          {exit, length, error, value, next_thought, agreement}
        """
        assert intent_name in self.leaves, \
            f"Unknown intent: {intent_name}. Call add_intent('{intent_name}') first."

        h = self.core(state_emb, thought)
        leaf = self.leaves[intent_name]
        return leaf(h)

    def compute_loss(self, state_emb: torch.Tensor, thought: torch.Tensor,
                     intent_name: str, exit_code: int, output: str,
                     reward: float) -> torch.Tensor:
        """
        计算单个样本的 loss (pred 已含 batch dim)

        Returns: scalar loss
        """
        pred = self.forward(state_emb, thought, intent_name)

        exit_target = torch.tensor([min(exit_code, 1)], device=state_emb.device, dtype=torch.long)
        length_target = torch.tensor([
            min(len(output.strip().splitlines()), 2)
        ], device=state_emb.device, dtype=torch.long)
        error_target = torch.tensor([
            1 if exit_code != 0 else 0
        ], device=state_emb.device, dtype=torch.long)
        value_target = torch.tensor([reward], device=state_emb.device, dtype=torch.float)

        loss = 0.0
        # pred outputs already have batch dim: (1, C)
        loss += F.cross_entropy(pred["exit"], exit_target)
        loss += F.cross_entropy(pred["length"], length_target)
        loss += F.cross_entropy(pred["error"], error_target)
        loss += F.mse_loss(pred["value"], value_target)

        return loss

    def update(self, state_emb: torch.Tensor, thought: torch.Tensor,
               intent_name: str, exit_code: int, output: str,
               reward: float) -> float:
        """
        训练一步 (单样本)

        Returns: loss value
        """
        self.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(state_emb, thought, intent_name,
                                 exit_code, output, reward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.total_steps += 1
        self.leaf_losses.setdefault(intent_name, []).append(loss.item())

        self.eval()
        return loss.item()

    def train_on_buffer(self, samples: list[dict]) -> float:
        """
        批量训练 (用经验缓冲区)

        samples: [{state_emb, thought, intent_name, exit_cls, length_cls, error_cls, value}]
        Returns: average loss
        """
        if len(samples) < 4:
            return 0.0

        self.train()
        by_intent: dict[str, list[dict]] = {}
        for s in samples:
            by_intent.setdefault(s["intent_name"], []).append(s)

        self.optimizer.zero_grad()
        n_batches = 0
        total_loss = 0.0

        for intent_name, group in by_intent.items():
            if intent_name not in self.leaves:
                continue
            s_embs = torch.stack([s["state_emb"] for s in group])
            thoughts = torch.stack([s["thought"] for s in group])
            exits = torch.tensor([s["exit_cls"] for s in group], dtype=torch.long)
            lengths = torch.tensor([s["length_cls"] for s in group], dtype=torch.long)
            errors = torch.tensor([s["error_cls"] for s in group], dtype=torch.long)
            values = torch.tensor([s["value"] for s in group], dtype=torch.float)

            pred = self.forward(s_embs, thoughts, intent_name)
            loss = (
                F.cross_entropy(pred["exit"], exits)
                + F.cross_entropy(pred["length"], lengths)
                + F.cross_entropy(pred["error"], errors)
                + F.mse_loss(pred["value"], values)
            ) / 4.0
            loss.backward()
            total_loss += loss.item()
            n_batches += 1
            self.leaf_losses.setdefault(intent_name, []).append(loss.item())

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.total_steps += 1
        self.eval()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def simulate(self, state_emb: torch.Tensor, thought: torch.Tensor,
                 intent_name: str) -> dict:
        """
        心理模拟: 预测执行结果 (不训练)

        Returns:
          {"value": float, "agreement": float, "next_thought": Tensor, ...}
        """
        self.eval()
        pred = self.forward(state_emb, thought, intent_name)
        agreement_prob = F.softmax(pred["agreement"], dim=-1)[0, 1].item()
        return {
            "value": pred["value"].item(),
            "agreement": agreement_prob,
            "exit_prob": F.softmax(pred["exit"], dim=-1)[0, 0].item(),
            "next_thought": pred["next_thought"].squeeze(0),
        }

    def get_intent_value(self, state_emb: torch.Tensor, thought: torch.Tensor,
                         intent_name: str) -> float:
        """快速获取意图价值预测"""
        return self.simulate(state_emb, thought, intent_name)["value"]

    def get_best_intent(self, state_emb: torch.Tensor, thought: torch.Tensor,
                        candidates: list[str]) -> tuple[str, float]:
        """从候选意图中选价值最高的"""
        best_name = candidates[0]
        best_value = -999.0
        for name in candidates:
            if name not in self.leaves:
                continue
            value = self.get_intent_value(state_emb, thought, name)
            if value > best_value:
                best_value = value
                best_name = name
        return best_name, best_value

    def save(self, path: str):
        """保存 checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "core": self.core.state_dict(),
            "leaves": {k: v.state_dict() for k, v in self.leaves.items()},
            "leaf_losses": self.leaf_losses,
            "total_steps": self.total_steps,
        }
        torch.save(data, path)

    def load(self, path: str):
        """加载 checkpoint"""
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.core.load_state_dict(data["core"])
        for name, sd in data.get("leaves", {}).items():
            if name not in self.leaves:
                self.leaves[name] = IntentLeaf(self.hidden_dim, self.thought_dim)
            self.leaves[name].load_state_dict(sd)
        self.leaf_losses = data.get("leaf_losses", {})
        self.total_steps = data.get("total_steps", 0)
        # 重建 optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4, weight_decay=1e-4
        )

    def get_leaf_stats(self) -> dict:
        """每个意图叶的训练统计"""
        stats = {}
        for name, losses in self.leaf_losses.items():
            if losses:
                recent = losses[-20:]
                stats[name] = {
                    "n_samples": len(losses),
                    "avg_loss": sum(recent) / len(recent),
                    "min_loss": min(recent),
                    "max_loss": max(recent),
                }
        return stats

    def stats(self) -> dict:
        leaf_st = self.get_leaf_stats()
        return {
            "n_leaves": len(self.leaves),
            "leaf_names": list(self.leaves.keys()),
            "total_steps": self.total_steps,
            "leaf_stats": leaf_st,
            "core_params": sum(p.numel() for p in self.core.parameters()),
            "leaf_params": {
                name: sum(p.numel() for p in leaf.parameters())
                for name, leaf in self.leaves.items()
            },
        }
