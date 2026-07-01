"""SalienceSignal — 杏仁核 / 全局显著性信号

将多个来源的新奇、惊喜、成功和奖励整合为一个标量显著性信号,
用于调制注意力、记忆编码、MODE 切换和默认模式网络活动。

设计约束 (P20):
- 无训练参数, 纯加权组合 + Sigmoid, 保证 CPU 上的可解释性和低开销。
- 运行窗口保留最近 50 个显著性值, 供 boredom/默认模式决策使用。
"""

import math
from collections import deque
from typing import Optional


class SalienceSignal:
    """全局显著性信号: 输入 → 显著性标量 → 窗口统计"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._window: deque[float] = deque(maxlen=window_size)
        self._last_salience: float = 0.0

    def update(
        self,
        rnd_novelty: Optional[float] = None,
        wm_surprise: Optional[float] = None,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> float:
        """根据最新观测更新显著性信号并返回该值。"""
        rnd_novelty = rnd_novelty if rnd_novelty is not None else 0.0
        wm_surprise = wm_surprise if wm_surprise is not None else 0.0
        reward = reward if reward is not None else 0.0

        # 成功信号: 成功时增加 0.2, 失败时 0
        success_signal = 0.2 if success is True else 0.0

        # 奖励信号: 仅使用正向奖励, 负奖励视为不显著
        reward_signal = max(0.0, reward) * 0.3

        raw = rnd_novelty + wm_surprise + success_signal + reward_signal
        salience = self._sigmoid(2.0 * raw)

        self._last_salience = salience
        self._window.append(salience)
        return salience

    @property
    def last_salience(self) -> float:
        return self._last_salience

    def recent_mean(self) -> float:
        """窗口内显著性的平均值; 窗口为空返回 0.0。"""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def recent_max(self) -> float:
        """窗口内显著性的最大值; 窗口为空返回 0.0。"""
        if not self._window:
            return 0.0
        return max(self._window)

    def recent_min(self) -> float:
        """窗口内显著性的最小值; 窗口为空返回 0.0。"""
        if not self._window:
            return 0.0
        return min(self._window)

    def get_stats(self) -> dict:
        """返回当前显著性统计。"""
        return {
            "last": self._last_salience,
            "mean": self.recent_mean(),
            "max": self.recent_max(),
            "min": self.recent_min(),
            "window_len": len(self._window),
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        # 避免数值溢出
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
