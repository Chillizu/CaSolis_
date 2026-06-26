"""
状态编码器 V3 — P15+: FactGraph 驱动的动态状态文本

核心变化:
  - 不再手写 ~200 行的 if/else 决定什么信息重要
  - 直接从 FactGraph 选 top 事实, 按 MODE 决定优先级
  - 所有硬编码路径列表和正则提取逻辑移除
"""

import re
from typing import Optional


class StateEncoder:
    """将环境状态编码为文本 — 动态版"""

    def __init__(self, workbench=None):
        self.current_dir = "/"
        self.current_goal = "探索系统"
        self._last_output_summary: str = ""
        self.workbench = workbench
        self._step: int = 0
        self._mode: str = "EXPLORE"
        self._last_reward: float = 0.0
        self._recent_rewards: list[float] = []

    def update(self, intent: str, command: str, output: str):
        """更新状态: 记录刚刚执行的命令和输出"""
        self._last_output_summary = self._summarize_output(intent, output)

    def set_step(self, step: int):
        self._step = step

    def set_mode(self, mode: str):
        self._mode = mode

    def set_reward(self, reward: float):
        self._last_reward = reward
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 20:
            self._recent_rewards = self._recent_rewards[-20:]

    def set_dir(self, path: str):
        self.current_dir = path

    def set_goal(self, goal: str):
        self.current_goal = goal

    def _summarize_output(self, intent: str, output: str) -> str:
        """将命令输出压缩成 1-2 行摘要"""
        if not output or len(output.strip()) == 0:
            return "(空)"
        lines = output.strip().splitlines()
        for line in lines[:10]:
            line = line.strip()
            if line and not line.startswith(("---", "total", "drwx", "-rw", "-", "lrwx")):
                return line[:80]
        first = lines[0].strip() if lines else "(空)"
        return first[:80]

    def get_state_text(self, thought_label: str = "") -> str:
        """
        生成状态文本 — 动态从 FactGraph 选取事实

        不再有硬编码的"CPU: xxx 内存: xxx"段落。
        改为: FactGraph 根据 MODE 选出最重要的节点。
        """
        parts = []

        # 1. 环境上下文 (始终保留)
        parts.append(f"步 {self._step}")
        parts.append(f"模式 {self._mode}")
        parts.append(f"dir {self.current_dir}")
        if thought_label:
            parts.append(thought_label)
        avg_r = sum(self._recent_rewards[-5:]) / max(len(self._recent_rewards[-5:]), 1)
        parts.append(f"rew {avg_r:.2f}")

        # 2. 从 FactGraph 选事实 (去掉硬编码的 if/else)
        facts_text = self._get_dynamic_facts()
        if facts_text:
            parts.append(facts_text)

        # 3. 最后输出摘要
        if self._last_output_summary:
            parts.append(f"out {self._last_output_summary[:50]}")

        return " ".join(parts)

    def _get_dynamic_facts(self, max_facts: int = 6) -> str:
        """
        从 FactGraph 动态选取最重要的节点

        选择策略 (由 MODE 决定):
          - EXPLORE: 系统事实 + 缺口 + 命令发现
          - CREATE: 工具结果 + 能力 + 创作相关
          - LEARN: 预测误差 + 新事实 + 意外

        没有任何硬编码的类别/路径/正则。
        """
        if not self.workbench:
            return ""
        graph = getattr(self.workbench, 'graph', None)
        if not graph or not graph.nodes:
            return ""

        # MODE → 偏好类别
        mode_cats = {
            "EXPLORE": {"system", "file", "package", "command", "network", "capability"},
            "CREATE": {"tool_result", "capability", "script", "package"},
            "LEARN": {"system", "tool_result", "command"},
        }
        prefer = mode_cats.get(self._mode, mode_cats["EXPLORE"])

        # 评分: 偏好类别 * 置信度 * 近期性
        scored = []
        for key, node in graph.nodes.items():
            score = 0.0
            # 类别匹配
            if node.category in prefer:
                score += 2.0
            # 高频类别削弱
            if node.category in ("general", "script", "explore"):
                score -= 1.0
            # 置信度
            score += node.confidence * 0.5
            # 近期性 (step 越大越新)
            score += min(node.step / 100, 1.0) * 0.3
            scored.append((score, key, node))

        scored.sort(key=lambda x: -x[0])

        # 取 top facts
        selected = scored[:max_facts]
        if not selected:
            return ""

        fact_parts = []
        for _, key, node in selected:
            val = str(node.value)[:30]
            fact_parts.append(f"{key}={val}")

        return " ".join(fact_parts)

    @property
    def explored_paths(self):
        """兼容旧接口: 不再追踪, 返回空集"""
        return set()

    def get_embedding_text(self) -> str:
        """更短的嵌入文本 (用于 RND)"""
        parts = [
            f"dir {self.current_dir}",
            f"mode {self._mode}",
        ]
        # 从 FactGraph 取 3 个最高置信度节点
        graph = getattr(self.workbench, 'graph', None) if self.workbench else None
        if graph and graph.nodes:
            top = sorted(graph.nodes.items(),
                         key=lambda x: (x[1].confidence, x[1].step),
                         reverse=True)[:3]
            for key, node in top:
                val = str(node.value)[:20]
                parts.append(f"{key}={val}")
        return " ".join(parts)


class RandomStateGenerator:
    """生成随机环境状态 (用于训练数据扩充)"""

    @classmethod
    def random_state_text(cls) -> str:
        import random
        dirs = ["/", "/etc", "/proc", "/tmp", "/usr/bin"]
        modes = ["EXPLORE", "CREATE", "LEARN"]
        return (
            f"步 {random.randint(1,300)} "
            f"模式 {random.choice(modes)} "
            f"dir {random.choice(dirs)} "
            f"out sample output"
        )
