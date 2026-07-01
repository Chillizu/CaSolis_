"""
经验缓冲区 — 存储和采样训练经验

每条经验: (state_text, intent, params, output, reward, next_state_text, novelty)
支持 staleness-weighted 采样: 旧经验按年龄指数衰减权重, 
参考 Ornith-1.0 的 staleness-weighted GRPO 机制。
"""

import json
import math
import random
from collections import deque
from typing import Optional


class Experience:
    """单条经验"""
    def __init__(self, state_text: str, intent: str, params: dict,
                 output: str, reward: float, next_state_text: str,
                 novelty: float = 0.0, success: bool = False,
                 exit_code: int = 0, thought: list = None,
                 recorded_step: int = 0):
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
        self.recorded_step = recorded_step  # 记录时的 step, 用于 staleness 加权

    def staleness_weight(self, current_step: int,
                         K1: int = 50, K2: int = 200,
                         decay_lambda: float = 0.02) -> float:
        """Ornith-1.0 风格的 staleness 权重。

        w(d) = 1                  if d <= K1
               exp(-λ(d - K1))    if K1 < d <= K2
               0                  if d > K2

        其中 d = current_step - recorded_step 是经验的年龄。
        """
        d = current_step - self.recorded_step
        if d <= K1:
            return 1.0
        elif d > K2:
            return 0.0
        else:
            return math.exp(-decay_lambda * (d - K1))

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
            "recorded_step": self.recorded_step,
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

    def sample_with_staleness_weights(
        self, n: int, current_step: int,
        K1: int = 50, K2: int = 200, decay_lambda: float = 0.02,
    ) -> list[tuple[Experience, float]]:
        """采样 n 条经验, 每条附带 staleness 权重。

        返回 [(经验, 权重), ...], 权重在 [0, 1] 范围,
        可用于后续 loss 加权。
        """
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        sampled = random.sample(list(self.buffer), n)
        return [(e, e.staleness_weight(current_step, K1, K2, decay_lambda))
                for e in sampled]

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
                    recorded_step=d.get("recorded_step", 0),
                ))
