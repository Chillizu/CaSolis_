"""
情景记忆 (EpisodicMemory) — 存储"意外转移"

人脑类比: 海马体 — 记住意外事件, 用于未来决策

触发条件: 世界模型预测 vs 实际结果差异大 (surprise > threshold)

存储: JSONL 环形缓冲区 + 内存索引
召回: 余弦相似度 + recency decay

只在 V4 ready 后启用, 防止噪声填满记忆
"""

import os
import json
import math
from collections import deque
from typing import Optional


class Episode:
    """一条情景记忆"""

    def __init__(self, step: int, state_emb: list[float], state_text: str,
                 intent: str, params: dict, predicted_reward: float,
                 actual_reward: float, surprise: float, mode: str,
                 resolved: bool = True):
        self.step = step
        self.state_emb = state_emb[:64]  # 存前64维足够
        self.state_text = state_text[:300]
        self.intent = intent
        self.params = params
        self.predicted_reward = predicted_reward
        self.actual_reward = actual_reward
        self.surprise = surprise
        self.mode = mode
        self.resolved = resolved

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "state_emb": self.state_emb,
            "state_text": self.state_text,
            "intent": self.intent,
            "params": {k: str(v) for k, v in self.params.items()},
            "predicted_reward": self.predicted_reward,
            "actual_reward": self.actual_reward,
            "surprise": self.surprise,
            "mode": self.mode,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            d["step"], d["state_emb"], d["state_text"],
            d["intent"], d.get("params", {}),
            d["predicted_reward"], d["actual_reward"],
            d["surprise"], d.get("mode", "EXPLORE"),
            d.get("resolved", True),
        )


class EpisodicMemory:
    """
    情景记忆

    用法:
      mem = EpisodicMemory()
      mem.add(episode)           # 存储
      hits = mem.recall(emb)     # 检索相似情景
      surprise = mem.compute_surprise(predicted, actual, rnd_novelty)
    """

    def __init__(self, path: str = "data/episodic_memory.jsonl",
                 max_size: int = 2000, surprise_threshold: float = 0.8):
        self.path = path
        self.max_size = max_size
        self.surprise_threshold = surprise_threshold
        self.episodes: deque[Episode] = deque(maxlen=max_size)
        self._high_surprise_pool: list[Episode] = []  # top 5% 永久保留
        self._enabled = False  # 只在 V4 ready 后启用
        self._load()

    def enable(self):
        """启用情景记忆 (V4 ready 后调用)"""
        self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled

    def compute_surprise(self, predicted_reward: float, actual_reward: float,
                         predicted_exit: int, actual_exit: int,
                         rnd_novelty: float) -> float:
        """计算 surprise 分数"""
        reward_diff = abs(predicted_reward - actual_reward) * 1.0
        exit_diff = (0 if predicted_exit == actual_exit else 1.0) * 0.5
        novelty = rnd_novelty * 0.3
        return reward_diff + exit_diff + novelty

    def add(self, state_emb, state_text: str, intent: str, params: dict,
            predicted_reward: float, actual_reward: float,
            predicted_exit: int, actual_exit: int,
            rnd_novelty: float, mode: str, step: int):
        """计算 surprise 并决定是否存储"""
        if not self._enabled:
            return

        surprise = self.compute_surprise(
            predicted_reward, actual_reward,
            predicted_exit, actual_exit, rnd_novelty
        )
        if surprise < self.surprise_threshold:
            return

        ep = Episode(
            step=step,
            state_emb=state_emb,
            state_text=state_text,
            intent=intent,
            params=params,
            predicted_reward=predicted_reward,
            actual_reward=actual_reward,
            surprise=surprise,
            mode=mode,
            resolved=actual_exit == 0,
        )
        self.episodes.append(ep)

        # 高 surprise 永久保留
        if surprise > 1.5 and len(self._high_surprise_pool) < 100:
            self._high_surprise_pool.append(ep)

        self._append_to_file(ep)

    def recall(self, query_emb: list[float], top_k: int = 3,
               min_surprise: float = 0.3, step: int = 0) -> list[Episode]:
        """按余弦相似度召回最相关的情景记忆"""
        if not self._enabled:
            return []

        candidates = list(self.episodes) + self._high_surprise_pool
        if not candidates:
            return []

        query = query_emb[:64]
        q_norm = math.sqrt(sum(x*x for x in query))
        if q_norm < 1e-8:
            return []

        scored = []
        for ep in candidates:
            if ep.surprise < min_surprise:
                continue
            # 余弦相似度 (点积 / 范数)
            e_emb = ep.state_emb[:64]
            dot = sum(q*e for q, e in zip(query, e_emb))
            e_norm = math.sqrt(sum(x*x for x in e_emb))
            sim = dot / (q_norm * e_norm + 1e-8)

            # recency decay: 越老的 episode 权重越低
            recency = math.exp(-(step - ep.step) / 500.0) if step > ep.step else 1.0
            score = sim * (0.5 + 0.5 * recency) * ep.surprise

            scored.append((score, ep))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_k]]

    def get_high_surprise(self) -> list[Episode]:
        """获取高 surprise 永久池"""
        return self._high_surprise_pool

    # ── 持久化 ──

    def _append_to_file(self, ep: Episode):
        """追加一条记录到 JSONL"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
            # 限制文件大小
            if os.path.getsize(self.path) > 10 * 1024 * 1024:  # 10MB
                self._rewrite()
        except Exception:
            pass

    def _rewrite(self):
        """重写整个文件 (限制大小)"""
        try:
            with open(self.path, "w") as f:
                for ep in list(self.episodes)[-self.max_size:]:
                    f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _load(self):
        """从 JSONL 加载记忆"""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ep = Episode.from_dict(json.loads(line))
                        self.episodes.append(ep)
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "n_episodes": len(self.episodes),
            "n_high_surprise": len(self._high_surprise_pool),
            "n_total_recalled": sum(1 for _ in self.episodes),
            "avg_surprise": (sum(ep.surprise for ep in self.episodes) /
                            max(len(self.episodes), 1)),
        }
