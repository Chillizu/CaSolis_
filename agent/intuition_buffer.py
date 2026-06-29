"""
IntuitionBuffer — 直觉缓冲 (Phase 3-LITE)

基于余弦相似度的经验检索, 零新参数:
- 存储 (thought_vector, intent, params, reward) 
- 查询: 熟悉度, 类比, 方向建议
- O(N×d) = ~16K FLOPs, <100μs on CPU
"""
import torch
import torch.nn.functional as F
from typing import Optional


class IntuitionBuffer:
    """
    直觉缓冲: 快速模式匹配基于经验检索

    用法:
        buf = IntuitionBuffer(capacity=1024)
        buf.store(thought_vec, intent, params, reward)
        fam, analogy, direction = buf.query(thought_vec)
    """

    def __init__(self, capacity: int = 1024, top_k: int = 5):
        self.capacity = capacity
        self.top_k = top_k
        self.thoughts: list[torch.Tensor] = []  # list of (16,)
        self.intents: list[int] = []
        self.params: list[dict] = []
        self.rewards: list[float] = []
        self._index: int = 0  # 循环覆写指针 (FIFO)

    def store(self, thought: torch.Tensor, intent: int,
              params: Optional[dict] = None, reward: float = 0.0):
        """存储一条经验"""
        if len(self.thoughts) < self.capacity:
            self.thoughts.append(thought.detach().clone())
            self.intents.append(intent)
            self.params.append(params or {})
            self.rewards.append(reward)
        else:
            # FIFO 覆写
            idx = self._index % self.capacity
            self.thoughts[idx] = thought.detach().clone()
            self.intents[idx] = intent
            self.params[idx] = params or {}
            self.rewards[idx] = reward
            self._index += 1

    def query(self, thought: torch.Tensor,
              top_k: Optional[int] = None) -> dict:
        """
        查询直觉输出

        Args:
            thought: (16,) — 当前 thought 向量
            top_k: 检索数量 (默认 self.top_k)
        Returns:
            dict with:
                familiarity: float (max cosine sim, 0-1)
                analogy: dict (nearest neighbor's intent/params/reward)
                direction: dict (weighted intent vote)
                similarity: float
        """
        k = top_k or self.top_k
        if not self.thoughts:
            return {
                "familiarity": 0.0,
                "analogy": {"intent": -1, "params": {}, "reward": 0.0},
                "direction": {0: 0.0, 1: 0.0, 2: 0.0},
                "similarity": 0.0,
            }

        # Stack thoughts and compute cosine similarity
        stack = torch.stack(self.thoughts)  # (N, 16)
        sims = F.cosine_similarity(
            thought.unsqueeze(0), stack, dim=1)  # (N,)

        # Top-k
        top_sims, top_idx = sims.topk(min(k, len(sims)))
        familiarity = min(top_sims[0].item(), 1.0)

        # Analogy = nearest neighbor
        nn_idx = top_idx[0].item()
        analogy = {
            "intent": self.intents[nn_idx],
            "params": self.params[nn_idx],
            "reward": self.rewards[nn_idx],
            "similarity": top_sims[0].item(),
        }

        # Direction = weighted intent vote from top-k
        direction = {0: 0.0, 1: 0.0, 2: 0.0}
        for i in range(len(top_idx)):
            idx = top_idx[i].item()
            intent = self.intents[idx]
            weight = top_sims[i].item() * (1.0 + self.rewards[idx])
            direction[intent] = direction.get(intent, 0.0) + weight

        # Softmax-normalize direction
        d_tensor = torch.tensor(list(direction.values()))
        d_probs = F.softmax(d_tensor / max(d_tensor.max().item(), 0.01), dim=0)
        direction_normalized = {
            0: d_probs[0].item(),
            1: d_probs[1].item(),
            2: d_probs[2].item(),
        }

        return {
            "familiarity": familiarity,
            "analogy": analogy,
            "direction": direction_normalized,
            "similarity": top_sims.mean().item(),
        }

    @property
    def size(self) -> int:
        return len(self.thoughts)

    def clear(self):
        self.thoughts.clear()
        self.intents.clear()
        self.params.clear()
        self.rewards.clear()
        self._index = 0
