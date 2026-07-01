"""HabitSystem — 基底神经节 / 习惯系统

当某个意图-参数组合反复成功时, 将其固化为习惯, 后续可直接复用参数
而无需重新经过 Conductor/参数提取的 deliberation 路径。

设计约束 (P20):
- 无训练参数, 纯统计型。
- 置信度阈值和最小成功次数在构造时配置, 便于测试和调参。
- LEARN 模式下不应使用习惯 (由调用方控制)。
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Habit:
    """单个习惯记录"""
    intent: int
    key: str
    params: dict
    success_count: int = 0
    failure_count: int = 0
    total_reward: float = 0.0
    last_used_step: int = 0
    created_step: int = 0

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count

    @property
    def confidence(self) -> float:
        total = self.total
        if total == 0:
            return 0.0
        return self.success_count / total

    def update(self, success: bool, reward: float, step: int):
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_reward += reward
        self.last_used_step = step


class HabitSystem:
    """意图级习惯系统: 统计成功/失败, 高置信度时推荐历史参数"""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        min_success_count: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_success_count = min_success_count
        # key = intent (first cut, conservative and simple)
        self.habits: dict[int, Habit] = {}

    def _make_key(self, intent: int, params: dict) -> int:
        """习惯键: 第一版仅使用 intent, 避免过早过度特化。"""
        return intent

    def register(
        self,
        intent: int,
        params: dict,
        success: bool,
        reward: float,
        step: int,
    ) -> Habit:
        """注册一次经验, 更新或创建对应习惯。"""
        key = self._make_key(intent, params)
        if key not in self.habits:
            self.habits[key] = Habit(
                intent=intent,
                key=str(intent),
                params=copy.deepcopy(params),
                created_step=step,
            )
        habit = self.habits[key]
        habit.update(success, reward, step)
        return habit

    def suggest(
        self,
        intent: int,
        state_emb: Optional[Any] = None,
    ) -> Optional[dict]:
        """为指定意图推荐高置信度习惯的参数; 无合适习惯返回 None。"""
        candidates = [
            h for h in self.habits.values()
            if h.intent == intent
            and h.confidence >= self.confidence_threshold
            and h.success_count >= self.min_success_count
        ]
        if not candidates:
            return None

        # 按置信度 -> 总奖励 -> 最近使用步数排序
        best = max(
            candidates,
            key=lambda h: (h.confidence, h.total_reward, h.last_used_step),
        )
        return copy.deepcopy(best.params)

    def get_stats(self) -> dict:
        """返回习惯统计, 用于日志和调试。"""
        if not self.habits:
            return {"n_habits": 0, "top": []}

        sorted_habits = sorted(
            self.habits.values(),
            key=lambda h: (h.confidence, h.success_count, h.total_reward),
            reverse=True,
        )
        return {
            "n_habits": len(self.habits),
            "top": [
                {
                    "intent": h.intent,
                    "key": h.key,
                    "confidence": h.confidence,
                    "success_count": h.success_count,
                    "failure_count": h.failure_count,
                    "total_reward": h.total_reward,
                    "last_used_step": h.last_used_step,
                }
                for h in sorted_habits[:5]
            ],
        }
