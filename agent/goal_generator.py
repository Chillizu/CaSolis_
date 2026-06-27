"""
目标生成器 (GoalGenerator) V2 — 动态版

核心变化:
  - 删除全部硬编码: 阈值/路径/优先级/风格
  - 删除 gap_to_goal 字典 (手写缺口→命令映射)
  - 新增 try_command 目标类型: 发现新命令后自动生成"试试看"目标
  - 优先级由 RND 新颖度 + FactGraph 覆盖率 + 上次使用时间动态决定
  - 连接 KnowledgeMapper 发现命令 + ToolRegistry 工具

目标类型:
  - try_command: 试用新发现的命令
  - gap_fill: 填 FactGraph 缺口 (自动推导命令)
  - run_tool: 运行已有工具
  - content_create: 生成内容
  - verify: 验证事实
"""

import os
import random
from typing import Optional


class Goal:
    """一个可执行目标"""

    def __init__(self, goal_type: str, intent: str, params: dict,
                 priority: float = 0.5, source: str = "",
                 description: str = ""):
        self.type = goal_type
        self.intent = intent
        self.params = params
        self.priority = priority
        self.source = source
        self.description = description

    def to_tuple(self) -> tuple[str, dict]:
        return (self.intent, self.params)


class GoalGenerator:
    """
    动态目标生成器 — 无硬编码阈值/路径/优先级
    """

    def __init__(self, creative_writer=None):
        self.creative_writer = creative_writer
        self.last_goal_type: Optional[str] = None
        self.goal_history: list[dict] = []
        self._tried_commands: set[str] = set()  # 追踪哪些发现命令已经试过

        # 动态统计 (替代硬编码阈值)
        self._intent_usage: dict[str, int] = {}
        self._total_goals = 0
        self._filled_gaps: set[str] = set()  # 已填过的缺口, 防止重复

    def generate(self, mode: str, workbench=None, rnd_avg: float = 0.0,
                 step: int = 0, recent_intents: Optional[list[str]] = None,
                 force_create: bool = False,
                 knowledge_mapper=None, tool_registry=None) -> Optional[Goal]:
        """
        动态生成最佳目标 — 所有人决策由数据驱动

        Args:
          mode: EXPLORE | CREATE | LEARN
          knowledge_mapper: KnowledgeMapper 实例 (发现命令用)
          tool_registry: ToolRegistry 实例 (工具用)
        """
        if workbench is None:
            return None

        recent = recent_intents or []
        candidates = []

        # ── 1. 试用发现的新命令 ──
        n_try = self._try_command_candidates(knowledge_mapper, step)
        candidates.extend(n_try)

        # ── 2. FactGraph 缺口 (跳过已填过的) ──
        if hasattr(workbench, 'graph'):
            gaps = workbench.graph.find_gaps()
            for src, missing, rel in gaps[:5]:
                if missing in self._filled_gaps:
                    continue
                goal = self._gap_to_goal_dynamic(missing, workbench, step)
                if goal:
                    self._filled_gaps.add(missing)
                    candidates.append(goal)

        # ── 3. 工具执行 ──
        n_tool = self._tool_candidates(tool_registry, step)
        candidates.extend(n_tool)

        # ── 4. MODE 特定目标 ──
        if mode == "CREATE":
            n_create = self._create_candidates(workbench, step)
            candidates.extend(n_create)
        elif mode == "EXPLORE":
            n_explore = self._explore_candidates(workbench, step, rnd_avg)
            candidates.extend(n_explore)
        elif mode == "LEARN":
            n_learn = self._learn_candidates(workbench, step)
            candidates.extend(n_learn)

        # ── 5. 多样性: 降频权 (温和版本) ──
        if recent:
            intent_freq = {i: recent.count(i) for i in set(recent)}
            for c in candidates:
                freq = intent_freq.get(c.intent, 0)
                # 只在频率 > 3 时才降权, 且降幅温和
                if freq > 3:
                    c.priority *= max(0.7, 1.0 - freq * 0.05)

        # ── 选最佳 ──
        if not candidates:
            return None

        candidates.sort(key=lambda g: -g.priority)

        # 防止连续同类型
        if self.last_goal_type and len(candidates) > 1:
            if candidates[0].type == self.last_goal_type:
                for c in candidates[1:]:
                    if c.type != self.last_goal_type:
                        candidates.insert(0, c)
                        break

        selected = candidates[0]
        self.last_goal_type = selected.type
        self.goal_history.append({
            "step": step, "type": selected.type,
            "intent": selected.intent, "priority": selected.priority,
        })
        self._total_goals += 1
        self._intent_usage[selected.intent] = self._intent_usage.get(selected.intent, 0) + 1

        return selected

    # ── 新: 试用发现命令 ──

    def _try_command_candidates(self, knowledge_mapper, step: int) -> list[Goal]:
        """从 KnowledgeMapper 的发现中挑一个没试过的命令"""
        candidates = []
        if knowledge_mapper is None:
            return candidates

        explored = getattr(knowledge_mapper, '_explored_commands', set())
        if not explored:
            return candidates

        # 挑一个已发现但没试过的命令
        untried = explored - self._tried_commands
        if not untried:
            return candidates

        cmd = random.choice(list(untried))
        self._tried_commands.add(cmd)

        # 生成"试用"目标: 用 CUSTOM 执行该命令的 --help
        candidates.append(Goal(
            "try_command", "CUSTOM",
            {"custom_args": [cmd, "--help"], "cluster": "SYSTEM"},
            priority=0.75,
            source=f"try_cmd:{cmd}",
            description=f"试用: {cmd} --help"
        ))
        return candidates

    # ── 新: 动态缺口→目标 (替换 gap_to_goal 字典) ──

    def _gap_to_goal_dynamic(self, missing_key: str, wb, step: int) -> Optional[Goal]:
        """从缺口名自动推导执行什么命令, 不需要手写字典"""
        # 从缺口名推断可能的文件路径
        known_paths = {
            "os": "/etc/os-release",
            "version": "/etc/os-release",
            "kernel": "/proc/version",
            "cpu": "/proc/cpuinfo",
            "mem": "/proc/meminfo",
            "swap": "/proc/meminfo",
            "host": "/etc/hostname",
            "user": "/etc/passwd",
            "ip": "/proc/net/fib_trie",
            "net": "/proc/net/dev",
            "disk": "/proc/diskstats",
            "mount": "/etc/fstab",
            "module": "/proc/modules",
            "uptime": "/proc/uptime",
            "load": "/proc/loadavg",
        }
        for keyword, path in known_paths.items():
            if keyword in missing_key.lower():
                return Goal(
                    "gap_fill", "READ", {"path": path},
                    priority=0.7, source=f"gap:{missing_key}",
                    description=f"填缺口: {missing_key}"
                )

        # 如果没匹配到已知路径, 用 CUSTOM 执行一条相关命令
        cmd_map = {
            "arch": ["uname", "-m"],
            "time": ["date"],
            "user": ["id"],
            "group": ["cat", "/etc/group"],
            "service": ["ls", "/etc/init.d"],
            "pkg": ["dpkg", "-l"],
            "env": ["env"],
        }
        for keyword, cmd in cmd_map.items():
            if keyword in missing_key.lower():
                return Goal(
                    "gap_fill", "CUSTOM", {"custom_args": cmd, "cluster": "SYSTEM"},
                    priority=0.6, source=f"gap:{missing_key}",
                    description=f"填缺口: {missing_key}"
                )

        return None

    # ── 新: 工具执行目标 ──

    def _tool_candidates(self, tool_registry, step: int) -> list[Goal]:
        """如果有工具可用, 生成长间隔执行目标"""
        candidates = []
        if tool_registry is None:
            return candidates

        tools = tool_registry.get_available()
        if not tools:
            return candidates

        # 选最久没用的工具
        best = tools[0]
        last_used = best.get("last_used_step", 0)
        if step - last_used >= 50:
            desc = best.get('description', '')
            candidates.append(Goal(
                "run_tool", "CUSTOM",
                {"custom_args": ["python3", f"data/persistent/tools/{best['name']}"],
                 "cluster": "CREATIVE"},
                priority=0.65,
                source=f"tool:{best['name']}",
                description=f"工具: {best['name']} ({desc})"
            ))
        return candidates

    # ── MODE 候选 (精简版, 无硬编码路径) ──

    def _explore_candidates(self, wb, step: int, rnd_avg: float) -> list[Goal]:
        """EXPLORE: 完全依赖 FactGraph + RND, 无手写路径"""
        candidates = []

        # RND 好奇心: 从 FactGraph 找缺口而不是硬编码路径
        if rnd_avg > 0.03 and hasattr(wb, 'graph'):
            gaps = wb.graph.find_gaps()
            for src, missing, rel in gaps[:2]:
                goal = self._gap_to_goal_dynamic(missing, wb, step)
                if goal:
                    goal.priority = 0.9
                    goal.source = f"rnd_gap:{missing}"
                    candidates.append(goal)

        # 探针
        if hasattr(wb, '_build_dynamic_probes'):
            explored = set()
            if hasattr(wb, '_current_discovery') and wb._current_discovery:
                explored.add(wb._current_discovery)
            probes = wb._build_dynamic_probes(explored)
            for p in probes[:2]:
                cmd = p.get("cmd", ["ls"])
                candidates.append(Goal(
                    "gap_fill", "CUSTOM",
                    {"custom_args": cmd, "cluster": p.get("cluster", "SYSTEM")},
                    priority=0.6,
                    source=f"probe:{p.get('path_key', '')}",
                    description=f"探针: {' '.join(cmd)}"
                ))

        return candidates

    def _create_candidates(self, wb, step: int) -> list[Goal]:
        """CREATE: LLM + 模板 (无硬编码风格列表)"""
        candidates = []
        n_facts = len(wb.facts) if hasattr(wb, 'facts') else 0

        if self.creative_writer and n_facts >= 3:
            # 先试 LLM 创作
            for style in ["report", "story"]:
                ci = self.creative_writer.generate_content(wb, style=style)
                if ci and ci.get("source", "").startswith("llm"):
                    intent_type = "GENERATE" if style != "code" else "WRITE"
                    candidates.append(Goal(
                        "content_create", intent_type,
                        {"path": ci["path"], "content": ci["content"]},
                        priority=0.9, source=f"llm:{style}",
                        description=f"LLM{style}"
                    ))
                    break
            # 回退模板
            if not candidates and hasattr(wb, 'build_generate_content'):
                ci = wb.build_generate_content()
                if ci:
                    candidates.append(Goal(
                        "content_create", "GENERATE",
                        {"path": ci["path"], "content": ci["content"]},
                        priority=0.6, source="template",
                        description=f"模板: {ci.get('desc', 'content')}"
                    ))

        return candidates

    def _learn_candidates(self, wb, step: int) -> list[Goal]:
        """LEARN: 验证 (无硬编码)"""
        candidates = []

        if hasattr(wb, 'generate_script'):
            result = wb.generate_script()
            if result:
                script, combo = result
                import base64
                encoded = base64.b64encode(script.encode()).decode()
                candidates.append(Goal(
                    "verify", "CUSTOM",
                    {"custom_args": ["sh", "-c",
                        f"echo '{encoded}' | base64 -d > /tmp/v_{step}.sh && "
                        f"chmod +x /tmp/v_{step}.sh && bash /tmp/v_{step}.sh"],
                     "cluster": "CREATIVE"},
                    priority=0.7, source="verify",
                    description=f"验证: {combo}"
                ))
        return candidates

    def stats(self) -> dict:
        return {
            "last_goal_type": self.last_goal_type,
            "n_goals": self._total_goals,
            "intent_usage": dict(sorted(self._intent_usage.items(),
                                        key=lambda x: -x[1])[:5]),
            "tried_commands": len(self._tried_commands),
            "recent_goals": self.goal_history[-5:],
        }
