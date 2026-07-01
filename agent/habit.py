"""HabitSystem — 基底神经节 / 习惯系统

当某个意图-参数组合反复成功时, 将其固化为习惯, 后续可直接复用参数
而无需重新经过 Conductor/参数提取的 deliberation 路径。

设计约束 (P20):
- 无训练参数, 纯统计型。
- 置信度阈值和最小成功次数在构造时配置, 便于测试和调参。
- LEARN 模式下不应使用习惯 (由调用方控制)。
- 习惯键为 (intent, 命令模板), 不同命令在同一意图下各自累积统计,
  避免第一个成功的命令锁死整个意图。
"""

import copy
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Habit:
    """单个习惯记录"""
    intent: int
    key: str          # 命令模板名 (如 "cat", "free", "tool_gather_packages")
    composite_key: tuple  # (intent, key) 完整键
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
    """(意图, 命令)级习惯系统: 按具体命令统计成功/失败, 高置信度时推荐历史参数。

    通过将习惯键从纯意图提升到 (意图, 命令模板), 解决「一个 intent 只有一个解法」的固化问题。
    不同命令在同一 intent 下各自积累统计, 互不干扰。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        min_success_count: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_success_count = min_success_count
        self.habits: dict[tuple[int, str], Habit] = {}

    def _normalize_command(self, intent: int, params: dict) -> str:
        """从 params 提取命令模板名。

        TRY: custom_args 的第一个词 (如 cat, free, tool_gather_packages)
        OBSERVE/CREATE: path 的 basename 或动作类型
        回退: intent 名称
        """
        custom = params.get("custom_args")
        if custom and isinstance(custom, (list, tuple)) and len(custom) > 0:
            cmd = str(custom[0])
            return re.sub(r'\.(py|sh|pl)$', '', cmd.split('/')[-1])

        path = params.get("path")
        if path and isinstance(path, str):
            return path.split('/')[-1].split('.')[0] or "file"

        return f"intent_{intent}"

    def _make_key(self, intent: int, params: dict) -> tuple[int, str]:
        """习惯键: (intent, 命令模板)"""
        return (intent, self._normalize_command(intent, params))

    def register(
        self,
        intent: int,
        params: dict,
        success: bool,
        reward: float,
        step: int,
    ) -> Habit:
        """注册一次经验, 更新或创建对应命令的习惯。"""
        key = self._make_key(intent, params)
        if key not in self.habits:
            self.habits[key] = Habit(
                intent=intent,
                key=key[1],
                composite_key=key,
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
        current_params: Optional[dict] = None,
    ) -> Optional[dict]:
        """为指定意图和当前参数推荐高置信度习惯的参数。

        当提供 current_params 时, 只匹配与当前命令相同模板的习惯;
        若该命令无习惯 (或置信度不足), 返回 None。
        这避免了一个意图下不同命令互相覆盖参数。

        无合适习惯返回 None。
        """
        candidates = [
            h for h in self.habits.values()
            if h.intent == intent
            and h.confidence >= self.confidence_threshold
            and h.success_count >= self.min_success_count
        ]

        if current_params is not None:
            current_key = self._make_key(intent, current_params)
            candidates = [h for h in candidates if h.composite_key == current_key]
            if not candidates:
                return None

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
                    "confidence": round(h.confidence, 3),
                    "success_count": h.success_count,
                    "failure_count": h.failure_count,
                    "total_reward": round(h.total_reward, 4),
                    "last_used_step": h.last_used_step,
                }
                for h in sorted_habits[:5]
            ],
        }
