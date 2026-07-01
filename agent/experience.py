"""
经验缓冲区 — 存储和采样训练经验

每条经验: (state_text, intent, params, output, reward, next_state_text, novelty)
"""

import json
import random
from collections import deque
from typing import Optional


class Experience:
    """单条经验"""
    def __init__(self, state_text: str, intent: str, params: dict,
                 output: str, reward: float, next_state_text: str,
                 novelty: float = 0.0, success: bool = False,
                 exit_code: int = 0, thought: list = None):
        self.state_text = state_text
        self.intent = intent
        self.params = params
        self.output = output
        self.reward = reward
        self.next_state_text = next_state_text
        self.novelty = novelty
        self.success = success
        self.exit_code = exit_code
        self.thought = thought or []

    def to_dict(self):
        return {
            "state_text": self.state_text,
            "intent": self.intent,
            "params": self.params,
            "output": self.output[:500],
            "reward": self.reward,
            "next_state_text": self.next_state_text,
            "novelty": self.novelty,
            "success": self.success,
            "exit_code": self.exit_code,
            "thought": self.thought[:16] if self.thought else [],
        }


class ExperienceBuffer:
    """经验回放缓冲区"""

    def __init__(self, max_size: int = 5000):
        self.buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, exp: Experience):
        self.buffer.append(exp)

    def sample(self, n: int) -> list[Experience]:
        """随机采样 n 条经验"""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def sample_by_reward(self, n: int, min_reward: float = 0.0) -> list[Experience]:
        """按奖励阈值采样"""
        candidates = [e for e in self.buffer if e.reward >= min_reward]
        if not candidates:
            return []
        n = min(n, len(candidates))
        return random.sample(candidates, n)

    def sample_novel(self, n: int) -> list[Experience]:
        """采样新颖度最高的经验"""
        sorted_exp = sorted(self.buffer, key=lambda e: e.novelty, reverse=True)
        return sorted_exp[:n]

    @property
    def size(self):
        return len(self.buffer)

    def save(self, path: str):
        data = [e.to_dict() for e in self.buffer]
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def load(self, path: str):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                self.add(Experience(
                    state_text=d["state_text"],
                    intent=d["intent"],
                    params=d["params"],
                    output=d.get("output", ""),
                    reward=d.get("reward", 0),
                    next_state_text=d.get("next_state_text", ""),
                    novelty=d.get("novelty", 0),
                    success=d.get("success", False),
                ))
