"""
状态编码器 V2 — P3: 知识状态跟踪

格式: "当前目录: {dir} 已知文件: {file} 上步: {goal}
       输出: {output_summary} 工作区: {workspace} 历史: {history}"

新增:
  - 输出摘要: 上一条命令的输出要点 (取代单纯intent label)
  - 工作区: /workspace 中的文件列表
  - 已知文件自动发现: 从 ls/cat/du 输出中提取文件路径
"""

import os
import random
import re
from typing import Optional


class StateEncoder:
    """将环境状态编码为文本 (匹配分类器训练格式)"""

    def __init__(self, workbench=None):
        self.conversation_history: list[dict] = []  # [{intent, command, output}]
        self.current_dir = "/"
        self.known_files: set[str] = set()
        self.current_goal = "探索系统"
        # P3: 缓存上次输出摘要
        self._last_output_summary: str = ""
        # P4: 工作栏引用 (不拥有, 外部共享)
        self.workbench = workbench

    def update(self, intent: str, command: str, output: str):
        """更新状态: 记录刚刚执行的命令和输出"""
        # P3: 生成输出摘要 (首行 + 关键内容)
        self._last_output_summary = self._summarize_output(intent, output)

        self.conversation_history.append({
            "intent": intent,
            "command": command,
            "output": output[:500],
        })
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

        # P3: 从输出中自动发现已知文件
        self._discover_files_from_output(intent, output)

    def _summarize_output(self, intent: str, output: str) -> str:
        """将命令输出压缩成 1-2 行摘要"""
        if not output or len(output.strip()) == 0:
            return "(空)"

        lines = output.strip().splitlines()
        
        # 提取第一个有意义的结果行
        for line in lines[:10]:
            line = line.strip()
            if line and not line.startswith(("---", "total", "drwx", "-rw", "-", "lrwx")):
                return line[:80]
        
        # 没有有意义行, 用第一行
        first = lines[0].strip() if lines else "(空)"
        return first[:80]

    def _discover_files_from_output(self, intent: str, output: str):
        """从命令输出中提取文件路径"""
        # ls 输出: 提取文件名
        for line in output.splitlines():
            # 模式1: -rwxr-xr-x 1 root root 1234 Jun 21 10:00 filename
            m = re.match(r"^[dl-][rwxst-]{9}\s+\d+\s+\S+\s+\S+\s+\d+\s+\S+\s+\S+\s+(\S+)$", line)
            if m:
                name = m.group(1)
                if name not in (".", ".."):
                    self.known_files.add(name)
                    continue
            
            # 模式2: 所有绝对路径 (只存basename)
            for path in re.findall(r"/(?:[a-zA-Z0-9_./-]+)", line):
                if any(path.startswith(p) for p in ("/etc/", "/tmp/", "/proc/", "/var/", "/workspace/", "/usr/", "/bin/", "/sbin/")):
                    base = os.path.basename(path)
                    if base not in (".", "..", ""):
                        self.known_files.add(base)

    def set_dir(self, path: str):
        self.current_dir = path

    def set_goal(self, goal: str):
        self.current_goal = goal

    def get_state_text(self) -> str:
        """P3+P4: 生成包含输出摘要、工作栏事实和思考向量的状态文本"""
        hist = self._format_history()
        # 已知文件: 最多3个
        file_priority = sorted(self.known_files, key=lambda f: (not f.startswith(("etc", "proc", "tmp", "bin", "usr", "var"))))
        known = "/".join(file_priority[:3]) if self.known_files else "未知"
        ws = self._get_workspace_files() if self.current_dir != "/workspace" else "."
        
        # P4: 工作栏事实摘要
        fact_summary = self.workbench.get_state_summary() if self.workbench else ""
        fact_part = f"事实: {fact_summary} " if fact_summary and fact_summary != "无" else ""
        
        # P4: 后续方向 (如果有推荐)
        follow_up = self.workbench.get_follow_up() if self.workbench else None
        goal_part = f"方向: {follow_up[0]} " if follow_up else ""
        
        return (
            f"当前目录: {self.current_dir} "
            f"已知文件: {known} "
            f"上步: {self.current_goal} "
            f"{fact_part}"
            f"{goal_part}"
            f"输出: {self._last_output_summary[:60]} "
            f"工作区: {ws} "
            f"历史: {hist}"
        )

    def _get_workspace_files(self) -> str:
        """查询 /workspace 文件列表"""
        try:
            # 尝试从文件系统读取 (在某些运行时可用)
            if os.path.isdir("/workspace"):
                files = os.listdir("/workspace")
                if files:
                    return ",".join(files[:5])
        except Exception:
            pass
        return "无"

    def _format_history(self) -> str:
        """P3: 历史格式改为 意图+输出摘要"""
        if not self.conversation_history:
            return "无"
        recent = self.conversation_history[-3:]
        parts = []
        for entry in recent:
            intent = entry["intent"]
            summary = self._summarize_output(intent, entry["output"])
            parts.append(f"{intent}({summary})")
        return " → ".join(parts)

    def get_embedding_text(self) -> str:
        """RND 用: 更短的嵌入文本, 不含长历史"""
        known = "/".join(sorted(self.known_files)[:3]) if self.known_files else "未知"
        ws = self._get_workspace_files() if self.current_dir != "/workspace" else "."
        # P4: 工作栏事实
        fact_summary = self.workbench.get_state_summary(max_keys=2) if self.workbench else ""
        fact_part = f"事实: {fact_summary} " if fact_summary and fact_summary != "无" else ""
        return (
            f"当前目录: {self.current_dir} "
            f"已知文件: {known} "
            f"上步: {self.current_goal} "
            f"{fact_part}"
            f"输出: {self._last_output_summary[:40]} "
            f"工作区: {ws} "
        )


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
