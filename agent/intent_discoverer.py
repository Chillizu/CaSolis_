"""
自动意图发现 — 从 CUSTOM 成功轨迹中挖掘新意图

流程:
  1. 收集 CUSTOM 成功执行: (state_text, command, output, exit_code)
  2. 用 MiniLM 编码 state_text → embeddings
  3. KMeans 聚类 → 每个聚类 = 候选新意图
  4. 从聚类中提取: 典型命令模板 + 意图名称
  5. 生成训练数据 → 下次重训分类器时加入

用法:
  discoverer = IntentDiscoverer()
  discoverer.add_custom_trajectory(state_text, intent, cmd_args, output, success)
  if discoverer.ready():
      new_intents = discoverer.discover()
      discoverer.save_training_data()
"""

import json
import re
import os
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np


class IntentDiscoverer:
    """
    从 CUSTOM 轨迹中自动发现新意图
    
    当成功 CUSTOM 轨迹积累到一定量后,
    用聚类分析找出重复的行为模式 → 新意图
    """

    def __init__(self, min_trajectories: int = 20, n_clusters: int = 2):
        self.min_trajectories = min_trajectories
        self.n_clusters = n_clusters
        self.trajectories: list[dict] = []
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    def add_custom_trajectory(
        self,
        state_text: str,
        cmd_args: list[str],
        output: str,
        success: bool,
    ):
        """记录一条 CUSTOM 执行轨迹"""
        if not success:
            return  # 只记录成功轨迹

        self.trajectories.append({
            "state_text": state_text,
            "command": " ".join(cmd_args) if cmd_args else "",
            "cmd_base": cmd_args[0] if cmd_args else "",
            "output_summary": output[:200],
            "timestamp": len(self.trajectories),
        })

    def ready(self) -> bool:
        """是否可以进行一次意图发现"""
        return len(self.trajectories) >= self.min_trajectories

    def discover(self) -> list[dict]:
        """
        运行聚类分析, 发现新意图
        
        Returns:
            [{
                "name": "新意图名",
                "cmd_template": ["cmd", "{path}"],
                "keywords": ["关键词列表"],
                "n_samples": 聚类样本数,
                "state_example": 典型状态文本,
            }, ...]
        """
        if not self.ready():
            return []

        texts = [t["state_text"] for t in self.trajectories]
        cmds = [t["cmd_base"] for t in self.trajectories]

        # MiniLM 嵌入
        embs = self._encoder.encode(texts, show_progress_bar=False)
        embs = np.array(embs)

        # 按命令基名分组 (比纯聚类更稳定)
        cmd_groups = defaultdict(list)
        for i, cmd in enumerate(cmds):
            cmd_groups[cmd].append(i)

        discovered = []
        # 找出高频命令组 (出现 >= 3 次) → 候选新意图
        for cmd_base, indices in cmd_groups.items():
            if len(indices) < 3:
                continue

            # 取这个命令的典型状态文本
            sample_idx = indices[0]
            sample_state = texts[sample_idx]

            # 生成意图名
            intent_name = self._generate_intent_name(cmd_base, sample_state)

            # 生成命令模板
            cmd_template = self._extract_template(cmd_base, self.trajectories[sample_idx]["command"])

            # 提取关键词
            keywords = self._extract_keywords(sample_state)

            discovered.append({
                "name": intent_name,
                "cmd_base": cmd_base,
                "cmd_template": cmd_template,
                "keywords": keywords,
                "n_samples": len(indices),
                "state_example": sample_state,
            })

        # 按样本数降序
        discovered.sort(key=lambda x: -x["n_samples"])
        return discovered

    def _generate_intent_name(self, cmd_base: str, state_text: str) -> str:
        """从命令和状态生成意图名"""
        # 从 state_text 提取关键词
        dirs = re.findall(r"/\w+", state_text)
        dir_hint = dirs[0].lstrip("/") if dirs else "sys"

        # 命令-意图映射
        cmd_intent_map = {
            "cat": "READ_ETC",
            "file": "CHECK_TYPE",
            "stat": "FILE_STAT",
            "du": "DISK_USAGE",
            "ps": "PROCESS_LIST",
            "mount": "MOUNT_INFO",
            "lspci": "PCI_DEVICES",
            "lsusb": "USB_DEVICES",
            "lsmod": "KERNEL_MODULES",
            "dns": "DNS_CONFIG",
            "env": "ENV_VARS",
            "timedatectl": "TIME_INFO",
        }
        return cmd_intent_map.get(cmd_base, f"{cmd_base.upper()}_{dir_hint.upper()}")

    def _extract_template(self, cmd_base: str, full_command: str) -> list[str]:
        """从完整命令中提取模板 (参数用 {path} 代替)"""
        if cmd_base == "cat":
            return ["cat", "{path}"]
        if cmd_base == "file":
            return ["file", "{path}"]
        if cmd_base == "stat":
            return ["stat", "{path}"]
        if cmd_base == "du":
            return ["du", "-sh", "{path}"]
        if cmd_base == "which":
            return ["which", "{cmd}"]
        if cmd_base == "ps":
            return ["ps", "-eo", "pid,ppid,cmd,%mem,%cpu", "--sort=-%mem"]
        # 通用: 返回原始 args (no template vars)
        return full_command.split()

    def _extract_keywords(self, state_text: str) -> list[str]:
        """从状态文本中提取关键词"""
        words = set()
        for w in state_text.lower().split():
            if len(w) > 2 and w not in ("当前", "目录", "已知", "文件", "上步", "历史"):
                words.add(w)
        return list(words)[:5]

    def generate_training_data(self) -> list[dict]:
        """
        从发现的新意图生成训练数据
        
        Returns:
            [{state_text, intent, intent_id}]
        """
        intents = self.discover()
        if not intents:
            return []

        samples = []
        for intent in intents:
            # 每个新意图生成 10 条训练样本
            for _ in range(10):
                state = self._synthesize_state(intent)
                samples.append({
                    "source": "auto_discovered",
                    "state_text": state,
                    "intent": intent["name"],
                    "intent_id": -1,  # 调用者分配 ID
                })

        return samples

    def _synthesize_state(self, intent: dict) -> str:
        """从意图生成一个合成状态文本"""
        kw = intent["keywords"]
        kw_str = kw[0] if kw else "系统"
        dirs = ["/etc", "/proc", "/", "/tmp", "/var/log"]
        d = random.choice(dirs)
        return (
            f"当前目录: {d} 已知文件: {kw_str} "
            f"上步: {intent['name']} 历史: 无"
        )

    def save_trajectories(self, path: str):
        """保存轨迹"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for t in self.trajectories:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def load_trajectories(self, path: str):
        """加载轨迹"""
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    self.trajectories.append(json.loads(line))
