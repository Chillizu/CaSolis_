"""
Stream 环境 — 模型持续交互的微型世界

v0.1: 数字序列环境（预测 + 好奇心的最小测试平台）
"""

import random
import torch
import numpy as np


class NumberSequenceEnv:
    """
    数字序列环境

    生成有规律的序列（包含模式和噪声），模型通过输出"动作"影响后续序列。
    这是最简环境，用来验证好奇心机制是否工作。

    规则：
    - 环境维护一个隐藏的"模式"
    - 观察值 token = 当前数字（0-127）
    - 模型输出的动作影响下一个数字
    - 如果模型输出正确预测，序列保持稳定
    - 如果模型探索新模式，序列会变化

    好奇心测试：
    - 当序列稳定时，预测误差低 → 模型感到"无聊"
    - 当模型选择探索时，可能遇到意外 → 预测误差高 → 好奇心触发
    """

    def __init__(
        self,
        vocab_size: int = 128,
        seed: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

        # 隐藏模式参数
        self._mode = "linear"  # linear, periodic, chaotic
        self._params = {"a": 1, "b": 0}
        self._phase = 0

        self.current_value = self.rng.randint(0, vocab_size - 1)
        self.step_count = 0
        self.modes = ["linear", "periodic", "chaotic"]
        self._frozen = False

    def reset(self) -> tuple[int, dict]:
        """重置环境，返回初始观察值"""
        self.step_count = 0
        self._mode = self.rng.choice(self.modes)
        self._phase = 0
        self._params = {
            "a": self.rng.randint(1, 5),
            "b": self.rng.randint(0, 10),
        }
        self.current_value = self.rng.randint(0, self.vocab_size - 1)
        self._frozen = False
        return self.current_value, {
            "mode": self._mode,
            "step": self.step_count,
        }

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """
        执行一步

        参数:
            action: 模型输出的动作 (0-15)

        返回:
            next_obs: 下一个观察值 token
            reward: 外在奖励（本环境为 0，好奇心是内部驱动）
            done: 是否结束（永不为 True，持续运行）
            info: 额外信息
        """
        self.step_count += 1
        self._phase += 1

        # 动作 0-7: 预测模式（尝试匹配预期）
        # 动作 8-15: 探索模式（主动改变序列）

        is_explore = action >= 8

        if self._mode == "linear":
            # 线性模式：每次加 a，模 vocab_size
            expected = (self.current_value + self._params["a"]) % self.vocab_size
        elif self._mode == "periodic":
            # 周期模式：sin 波
            period = self._params["a"] + 2
            expected = int(
                (np.sin(self._phase * 2 * np.pi / period) + 1) / 2
                * (self.vocab_size - 1)
            )
        elif self._mode == "chaotic":
            # 混沌模式：简单 logistic 映射的变体
            x = self.current_value / (self.vocab_size - 1)
            r = 3.5 + self._params["b"] * 0.1
            x = r * x * (1 - x)
            expected = int(x * (self.vocab_size - 1)) % self.vocab_size
        else:
            expected = (self.current_value + 1) % self.vocab_size

        if is_explore:
            # 探索：产生一个偏离预测的值（大小取决于探索强度）
            deviation = (action - 7) * self.rng.randint(3, 15)
            next_val = (expected + deviation) % self.vocab_size
        else:
            # 预测：尽量接近预期
            noise = self.rng.randint(-2, 2) if self._mode == "chaotic" else 0
            next_val = (expected + noise) % self.vocab_size

        # 偶尔切换模式（环境自身变化，测试模型的适应能力）
        if self.step_count > 0 and self.step_count % 50 == 0:
            self._mode = self.rng.choice(self.modes)
            info = {
                "mode_change": True,
                "new_mode": self._mode,
                "step": self.step_count,
            }
        else:
            info = {
                "mode": self._mode,
                "step": self.step_count,
                "expected": expected,
                "is_explore": is_explore,
            }

        self.current_value = next_val
        return next_val, 0.0, False, info

    def get_state(self) -> int:
        """获取当前观察值"""
        return self.current_value

    @property
    def mode(self) -> str:
        return self._mode
