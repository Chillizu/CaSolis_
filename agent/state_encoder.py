"""
状态编码器 — 将环境观察编码为分类器的 state_text

格式: "当前目录: {dir} 已知文件: {file} 上步: {goal} 历史: {history}"
"""

import os
import random
from typing import Optional


class StateEncoder:
    """将环境状态编码为文本 (匹配分类器训练格式)"""

    def __init__(self):
        self.conversation_history: list[dict] = []  # [{intent, command, output}]
        self.current_dir = "/"
        self.known_files: set[str] = set()
        self.current_goal = "探索系统"

    def update(self, intent: str, command: str, output: str):
        """更新状态: 记录刚刚执行的命令和输出"""
        self.conversation_history.append({
            "intent": intent,
            "command": command,
            "output": output[:500],
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

        # 更新已知文件
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith(("/", "total", "drwx", "-rw")):
                # 可能是文件或目录名
                pass

    def set_dir(self, path: str):
        self.current_dir = path

    def set_goal(self, goal: str):
        self.current_goal = goal

    def get_state_text(self) -> str:
        """生成当前状态文本 (与分类器训练格式一致)"""
        hist = self._format_history()
        known = "/".join(sorted(self.known_files)[:3]) if self.known_files else "未知"
        return (
            f"当前目录: {self.current_dir} "
            f"已知文件: {known} "
            f"上步: {self.current_goal} "
            f"历史: {hist}"
        )

    def _format_history(self) -> str:
        """从 conversation_history 中提取简洁的历史摘要"""
        if not self.conversation_history:
            return "无"
        # 只用最近 3 条
        recent = self.conversation_history[-3:]
        parts = []
        for entry in recent:
            intent = entry["intent"]
            output = entry["output"][:60].replace("\n", " ")
            parts.append(f"{intent}: {output}")
        return " → ".join(parts)

    def get_embedding_text(self) -> str:
        """获取只含状态的文本 (不含历史) 用于 RND"""
        known = "/".join(sorted(self.known_files)[:3]) if self.known_files else "未知"
        return f"当前目录: {self.current_dir} 已知文件: {known} 上步: {self.current_goal} 历史: 无"


class RandomStateGenerator:
    """生成随机环境状态 (用于训练数据扩充)"""

    DIRS = ["/", "/etc", "/var/log", "/proc", "/tmp", "/home", "/opt", "/dev", "/usr/bin"]
    FILES = ["passwd", "hostname", "hosts", "syslog", "cpuinfo", "meminfo", 
             "shadow", "group", "fstab", "services"]

    @classmethod
    def random_state_text(cls) -> str:
        d = random.choice(cls.DIRS)
        f = random.choice(cls.FILES)
        hist_opts = ["无", "cd → ls", "cat passwd", "grep root", "df -h"]
        return (
            f"当前目录: {d} "
            f"已知文件: {f} "
            f"上步: 探索{os.path.basename(d) or '系统'} "
            f"历史: {random.choice(hist_opts)}"
        )
