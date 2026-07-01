"""Planner — 多步骤创作编排

GoalGenerator 不再输出单句意图, 而是输出结构化的多步计划。
每步有: 描述、前置依赖、输出格式。
DeepSeek 逐步实现, 每步一个独立文件。
"""

import json, os, random
from pathlib import Path
from typing import Optional


class Plan:
    """一个多步骤创作计划"""

    def __init__(self, plan_id: str, steps: list[dict],
                 created_step: int = 0, topic: str = ""):
        self.plan_id = plan_id
        self.steps = steps          # [{"id","style","desc","deps","done"}, ...]
        self.current_idx = 0
        self.created_step = created_step
        self.topic = topic
        self._find_first_ready()

    def _find_first_ready(self):
        """找到第一个所有依赖已完成的步骤"""
        done_ids = {s["id"] for s in self.steps if s.get("done")}
        for i, s in enumerate(self.steps):
            if s.get("done"):
                continue
            if all(d in done_ids for d in s.get("deps", [])):
                self.current_idx = i
                return
        self.current_idx = len(self.steps)  # all done

    @property
    def current_step(self) -> Optional[dict]:
        if self.current_idx >= len(self.steps):
            return None
        return self.steps[self.current_idx]

    @property
    def done(self) -> bool:
        return self.current_idx >= len(self.steps)

    def mark_step_done(self, step_id: int, file_path: str = ""):
        for s in self.steps:
            if s["id"] == step_id:
                s["done"] = True
                s["file"] = file_path
                break
        self._find_first_ready()

    def remaining(self) -> int:
        return sum(1 for s in self.steps if not s.get("done"))

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "topic": self.topic,
            "steps": self.steps,
            "current_idx": self.current_idx,
            "done": self.done,
            "created_step": self.created_step,
        }


# ── 计划模板 ──
# GoalGenerator 根据观察到的标签 + 工作栏数据选择模板
# 每个模板含 2-4 步, 有依赖链

PLAN_TEMPLATES: dict[str, list[dict]] = {

    "kernel-deep-probe": [
        {"id": 0, "style": "code", "desc": "Build a recursive scanner for {domain} that walks all files and returns key-value pairs as a dictionary",
         "deps": []},
        {"id": 1, "style": "code", "desc": "Write a change detector that compares two snapshots of {domain} and reports differences",
         "deps": [0]},
        {"id": 2, "style": "code", "desc": "Add persistent logging of detected changes to a JSONL file with timestamps",
         "deps": [1]},
        {"id": 3, "style": "report",
         "desc": "Generate a summary report of the top 10 most volatile parameters in {domain} based on the change log",
         "deps": [0, 1, 2]},
    ],

    "network-security-scan": [
        {"id": 0, "style": "code",
         "desc": "Build a parser for /proc/net/tcp and /proc/net/udp that extracts connection states and remote addresses",
         "deps": []},
        {"id": 1, "style": "code",
         "desc": "Write a function that tracks new external connections over time and flags suspicious IPs (high port count, known bad patterns)",
         "deps": [0]},
        {"id": 2, "style": "report",
         "desc": "Generate a security report summarizing all active connections, suspicious IPs, and listening services",
         "deps": [0, 1]},
    ],

    "memory-profiler": [
        {"id": 0, "style": "code",
         "desc": "Write a detailed /proc/meminfo parser that extracts all memory tiers (anonymous, page cache, slab, hugepages) into structured dict",
         "deps": []},
        {"id": 1, "style": "code",
         "desc": "Build a trend tracker that takes multiple snapshots of memory usage over time and computes deltas for each tier",
         "deps": [0]},
        {"id": 2, "style": "analysis",
         "desc": "Analyze the memory consumption pattern: which tier is growing, which is stable, any anomalies compared to total RAM",
         "deps": [0, 1]},
    ],

    "process-watcher": [
        {"id": 0, "style": "code",
         "desc": "Build a process snapshot tool that reads /proc/*/status and extracts PID, name, state, memory, and uptime for all processes",
         "deps": []},
        {"id": 1, "style": "code",
         "desc": "Add top-N sorting by Memory and CPU time, and detect zombie/defunct processes",
         "deps": [0]},
        {"id": 2, "style": "report",
         "desc": "Generate a process health report with top consumers, zombie count, and anomaly flags",
         "deps": [0, 1]},
    ],

    "filesystem-audit": [
        {"id": 0, "style": "code",
         "desc": "Write a /proc/fs parser that reads filesystem statistics: file handles, inodes, dentries, and mount info",
         "deps": []},
        {"id": 1, "style": "code",
         "desc": "Build a threshold monitor that compares current fs metrics against kernel limits and reports utilization ratios",
         "deps": [0]},
        {"id": 2, "style": "report",
         "desc": "Generate a filesystem capacity report showing current usage vs limits for each subsystem",
         "deps": [0, 1]},
    ],

    "code-archive-analysis": [
        {"id": 0, "style": "code",
         "desc": "Write a scanner for the code archive directory that collects metadata: file size, function count, import statements for each .py file",
         "deps": []},
        {"id": 1, "style": "analysis",
         "desc": "Analyze the code archive growth trend: how total size, function count, and import diversity change over time",
         "deps": [0]},
    ],
}


def generate_plan(tag: str, domain: str, plan_id: str) -> Plan:
    """从标签生成多步计划"""
    template_key = None
    for key in PLAN_TEMPLATES:
        if key in tag or tag in key:
            template_key = key
            break
    if not template_key:
        # fallback: simple 2-step
        template_key = "code-archive-analysis"
    template = PLAN_TEMPLATES[template_key]
    # 填充 domain
    steps = []
    for s in template:
        step = dict(s)
        step["desc"] = step["desc"].replace("{domain}", domain)
        step["done"] = False
        step["file"] = ""
        steps.append(step)
    return Plan(plan_id, steps, topic=domain)
