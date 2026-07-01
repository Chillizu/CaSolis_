"""
IntuitionBuffer — 直觉缓冲 (Phase 3 升级)

从经验中学习的直觉模块:
- 存储 (thought_vector, intent, params, reward, success, surprise)
- 查询: 返回直觉建议（权重来源于相似成功经验）
- 不依赖任何外部模型（纯余弦相似度 + 经验加权）
- O(N×d) = ~16K FLOPs, <100μs on CPU
"""
import torch
import torch.nn.functional as F
from typing import Optional


class IntuitionBuffer:
    """
    直觉缓冲: 基于经验检索的快速模式匹配

    核心变化:
    - store() 接受 success 和 surprise, 不光 reward
    - query() 按成功经验加权, 不成功的历史不会污染方向
    - direction_prob() 返回 [p_obs, p_create, p_try] 可直接采样

    Usage:
        buf = IntuitionBuffer(capacity=1024)
        buf.store(thought, intent, params, reward, success, surprise)
        result = buf.query(thought)
        direction = result["direction_probs"]  # [p0, p1, p2]
    """

    def __init__(self, capacity: int = 1024, top_k: int = 8):
        self.capacity = capacity
        self.top_k = top_k
        self.thoughts: list[torch.Tensor] = []       # (16,)
        self.state_embs: list[torch.Tensor | None] = []  # P19: 可选 384D 状态嵌入
        self.intents: list[int] = []                  # 0=OBSERVE, 1=CREATE, 2=TRY
        self.params: list[dict] = []
        self.rewards: list[float] = []
        self.successes: list[bool] = []               # 新增: 是否成功
        self.surprises: list[float] = []               # 新增: 惊奇度
        self.imagined: list[bool] = []                 # P19: 是否想象经验
        self._index: int = 0                          # 循环覆写指针

    def store(self, thought: torch.Tensor, intent: int,
              params: Optional[dict] = None,
              reward: float = 0.0,
              success: bool = False,
              surprise: float = 0.0,
              state_emb: Optional[torch.Tensor] = None,
              imagined: bool = False):
        """存储一条经验, 含成功标记和惊奇度"""
        entry = (
            thought.detach().clone(),
            intent,
            params or {},
            reward,
            success,
            surprise,
            state_emb.detach().clone() if state_emb is not None else None,
            imagined,
        )
        if len(self.thoughts) < self.capacity:
            self.thoughts.append(entry[0])
            self.intents.append(entry[1])
            self.params.append(entry[2])
            self.rewards.append(entry[3])
            self.successes.append(entry[4])
            self.surprises.append(entry[5])
            self.state_embs.append(entry[6])
            self.imagined.append(entry[7])
        else:
            idx = self._index % self.capacity
            self.thoughts[idx] = entry[0]
            self.intents[idx] = entry[1]
            self.params[idx] = entry[2]
            self.rewards[idx] = entry[3]
            self.successes[idx] = entry[4]
            self.surprises[idx] = entry[5]
            self.state_embs[idx] = entry[6]
            self.imagined[idx] = entry[7]
            self._index += 1

    def query(self, thought: torch.Tensor,
              top_k: Optional[int] = None) -> dict:
        """
        查询直觉: 从相似经验中提取方向建议

        Returns:
            familiarity: float — 当前状态与历史经验的最高相似度
            direction_probs: [p0, p1, p2] — 每个意图的加权概率
            suggested_intent: int — 概率最高的意图
            best_match: dict — 最相似的成功经验
        """
        if not self.thoughts:
            return self._empty_result()

        k = min(top_k or self.top_k, len(self.thoughts))

        # 余弦相似度
        stack = torch.stack(self.thoughts)
        sims = F.cosine_similarity(thought.unsqueeze(0), stack, dim=1)

        top_sims, top_idx = sims.topk(k)
        familiarity = max(0.0, top_sims[0].item())

        # 方向权重: 相似度 × (1.0 + reward) × (1.5 若成功)
        vote = [0.0, 0.0, 0.0]
        best_match = {"intent": 0, "reward": 0.0, "success": False,
                      "params": {}, "similarity": 0.0}

        for i in range(k):
            idx = top_idx[i].item()
            sim = max(0.0, top_sims[i].item())
            reward = self.rewards[idx]
            is_success = self.successes[idx]
            intent = self.intents[idx]
            if intent < 0 or intent > 2:
                intent = 0

            # 基础权重 = 相似度
            weight = sim
            # 高奖励加成
            weight *= (1.0 + max(0.0, reward))
            # 成功经验 ×1.5, 失败经验 ×0.5
            weight *= 1.5 if is_success else 0.5
            # 高 surprise 也有探索价值
            surprise = self.surprises[idx]
            if surprise > 0.001:
                weight *= (1.0 + min(surprise * 10, 1.0))

            vote[intent] += weight

            # 记录最佳匹配（偏向成功经验）
            if is_success and sim > best_match["similarity"]:
                best_match = {
                    "intent": intent,
                    "reward": reward,
                    "success": is_success,
                    "params": self.params[idx],
                    "similarity": sim,
                }

        # 从投票转为概率分布
        v_t = torch.tensor(vote, dtype=torch.float)
        # 确保非负, 零投票则轻微均匀
        if v_t.sum() < 1e-8:
            v_t = torch.ones(3) * 0.1
        probs = F.softmax(v_t / max(v_t.max().item(), 0.01), dim=0)

        return {
            "familiarity": familiarity,
            "direction_probs": [probs[0].item(), probs[1].item(), probs[2].item()],
            "suggested_intent": int(probs.argmax().item()),
            "best_match": best_match,
            "n_entries": len(self.thoughts),
        }

    def direction_prob(self, thought: torch.Tensor) -> list[float]:
        """快捷方法: 只返回方向概率 [p_obs, p_create, p_try]"""
        r = self.query(thought)
        return r["direction_probs"]

    def sample_batch(self, batch_size: int = 32,
                     prefer_success: bool = True) -> list[dict]:
        """
        P19: 采样一批经验，用于睡眠巩固。
        若 prefer_success=True，则高奖励/成功经验更可能被采到。
        """
        if not self.thoughts:
            return []
        n = len(self.thoughts)
        if prefer_success:
            weights = torch.tensor([
                (1.5 if self.successes[i] else 0.5) *
                max(0.1, 1.0 + self.rewards[i])
                for i in range(n)
            ])
            idx = torch.multinomial(weights, min(batch_size, n), replacement=True).tolist()
        else:
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()

        return [{
            "thought": self.thoughts[i],
            "intent": self.intents[i],
            "params": self.params[i],
            "reward": self.rewards[i],
            "success": self.successes[i],
            "surprise": self.surprises[i],
            "state_emb": self.state_embs[i],
            "imagined": self.imagined[i],
        } for i in idx]

    def real_imagined_ratio(self) -> tuple[int, int]:
        """P19: 返回真实/想象经验数量"""
        n = len(self.imagined) if self.imagined else 0
        if n == 0:
            return 0, 0
        img = sum(self.imagined)
        return n - img, img

    def _empty_result(self):
        return {
            "familiarity": 0.0,
            "direction_probs": [1/3, 1/3, 1/3],
            "suggested_intent": 0,
            "best_match": {"intent": 0, "reward": 0.0, "success": False,
                           "params": {}, "similarity": 0.0},
            "n_entries": 0,
        }

    @property
    def size(self) -> int:
        return len(self.thoughts)

    def clear(self):
        self.thoughts.clear()
        self.state_embs.clear()
        self.intents.clear()
        self.params.clear()
        self.rewards.clear()
        self.successes.clear()
        self.surprises.clear()
        self.imagined.clear()
        self._index = 0
