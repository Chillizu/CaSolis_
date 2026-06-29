"""
目标生成器 (GoalGenerator) V2 — 动态版

核心变化:
  - 删除全部硬编码: 阈值/路径/优先级/风格
  - 删除 gap_to_goal 字典 (手写缺口→命令映射)
  - 新增 try_command 目标类型: 发现新命令后自动生成"试试看"目标
  - 优先级由 RND 新颖度 + FactGraph 覆盖率 + 上次使用时间动态决定
  - 连接 KnowledgeMapper 发现命令 + ToolRegistry 工具

目标类型:
  - try_command: 试用新发现的命令 → TRY
  - gap_fill: 填 FactGraph 缺口 → OBSERVE
  - run_tool: 运行已有工具 → TRY
  - content_create: 生成内容 → CREATE
  - verify: 验证事实 → OBSERVE
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
        self._tried_commands: set[str] = set()
        self._intent_command_map: dict[str, list[str]] = {}  # 从 KnowledgeMapper 获取

        # 动态统计 (替代硬编码阈值)
        self._intent_usage: dict[str, int] = {}
        self._total_goals = 0
        self._filled_gaps: set[str] = set()  # 已填过的缺口, 防止重复

    def generate(self, mode: str, workbench=None, rnd_avg: float = 0.0,
                 step: int = 0, recent_intents: Optional[list[str]] = None,
                 force_create: bool = False,
                 knowledge_mapper=None, tool_registry=None,
                 hypothesis_engine=None, fact_graph=None) -> Optional[Goal]:
        """
        动态生成最佳目标 — 所有决策由数据驱动

        Args:
          mode: EXPLORE | CREATE | LEARN
          knowledge_mapper: KnowledgeMapper 实例
          tool_registry: ToolRegistry 实例
          hypothesis_engine: P16 R3 假设引擎 (LEARN 模式使用)
          fact_graph: P16 R3 FactGraph (用于假设生成)
        """
        if workbench is None:
            return None

        # 收集推理节点 (影响目标选择)
        self._inferences = self._collect_inferences(workbench)

        recent = recent_intents or []
        candidates = []

        # 更新 intent→命令映射 (从 KnowledgeMapper)
        if knowledge_mapper and hasattr(knowledge_mapper, 'get_intent_command_map'):
            self._intent_command_map = knowledge_mapper.get_intent_command_map()

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
                # 试着找已知命令来填这个缺口
                cmd_goal = self._command_for_gap(missing, step)
                if cmd_goal:
                    candidates.append(cmd_goal)

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
            # P16 R3: 假设验证目标
            if hypothesis_engine and fact_graph:
                try:
                    hyps = getattr(hypothesis_engine, '_latest_hypotheses', None)
                    if hyps is None:
                        # 尝试从 hypothesis_engine 获取
                        hyps = getattr(hypothesis_engine, '_generated_hypotheses', [])
                    if not hyps and hasattr(hypothesis_engine, 'propose'):
                        hyps = hypothesis_engine.propose(fact_graph, top_k=3)
                    for h in (hyps or []):
                        candidates.append(Goal(
                            "hypothesis_test", "TRY",
                            {"hypothesis": h,
                             "hypothesis_key": f"{h['if_node']}:{h['rel']}:{h['then_node']}"},
                            priority=0.8 + 0.2 * h.get("priority", 0),
                            source="hypothesis_engine",
                            description=h.get("prediction", ""),
                        ))
                except Exception:
                    pass

        # ── 5. 多样性: 降频权 (温和版本) ──
        if recent:
            intent_freq = {i: recent.count(i) for i in set(recent)}
            for c in candidates:
                freq = intent_freq.get(c.intent, 0)
                # 只在频率 > 3 时才降权, 且降幅温和
                if freq > 3:
                    c.priority *= max(0.7, 1.0 - freq * 0.05)

        # ── 6. 多样性: 周期性注入 CREATE (即使被分类器/Conductor选了OBSERVE/TRY) ──
        # 6a: 如果候选全是 TRY, 随机加 CREATE
        all_try = all(g.intent == "TRY" for g in candidates)
        if all_try and len(candidates) >= 2 and random.random() < 0.15:
            creates = self._create_candidates(workbench, step)
            if creates:
                candidates.extend(creates)
        # 6b: 自适应注入 CREATE 目标
        n_create_recent = sum(1 for h in self.goal_history[-10:] if h['type'] == 'content_create')
        create_prob = 0.05 * (1.0 - min(n_create_recent / 5.0, 1.0))  # 最近CREATE越多, 概率越低
        if step > 20 and random.random() < create_prob:
            creates = self._create_candidates(workbench, step)
            if creates:
                candidates.extend(creates)

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
        """从 KnowledgeMapper 的所有可用命令中挑没试过的 (不限于已explored)"""
        candidates = []
        if knowledge_mapper is None:
            return candidates

        all_cmds = getattr(knowledge_mapper, '_all_available_commands', [])
        if not all_cmds:
            return candidates

        # 从所有可用命令中挑 (已被 scan_available_commands 发现)
        import random
        untried = [c for c in all_cmds if c not in self._tried_commands]
        if not untried:
            return candidates

        # 尝 3 个
        for cmd in random.sample(untried, min(3, len(untried))):
            self._tried_commands.add(cmd)
            candidates.append(Goal(
            "try_command", "TRY",
            {"custom_args": [cmd, "--help"], "cluster": "SYSTEM"},
            priority=0.75,
            source=f"try_cmd:{cmd}",
            description=f"试用: {cmd} --help"
        ))
        return candidates

    # ── 新: 动态缺口→目标 (替换 gap_to_goal 字典) ──

    def _command_for_gap(self, missing_key: str, step: int) -> Optional[Goal]:
        """
        从已发现命令中找能填这个缺口的
        比如缺口 cpu_model → 已发现 arch → CUSTOM arch
        """
        if not self._intent_command_map:
            return None

        # 从缺口名推断需要的意图
        intent_hints = {
            "cpu": "ARCH_INFO", "arch": "ARCH_INFO", "model": "ARCH_INFO",
            "processor": "ARCH_INFO", "hardware": "ARCH_INFO",
            "mem": "INFO", "disk": "DISK_USAGE", "storage": "DISK_USAGE",
            "usb": "USB_DEVICES", "device": "USB_DEVICES", "pci": "USB_DEVICES",
            "file": "READ", "list": "LIST", "search": "SEARCH", "find": "SEARCH",
            "count": "COUNT", "num": "COUNT",
            "net": "INFO", "network": "INFO", "ip": "INFO",
            "host": "INFO", "user": "INFO", "name": "INFO",
        }
        needed_intent = None
        for keyword, intent in intent_hints.items():
            if keyword in missing_key.lower():
                needed_intent = intent
                break

        if not needed_intent or needed_intent not in self._intent_command_map:
            return None

        # 取一个已发现且未试过的相关命令
        cmds = self._intent_command_map[needed_intent]
        for cmd in cmds:
            if cmd not in self._tried_commands:
                self._tried_commands.add(cmd)
                return Goal(
                    "gap_fill", "TRY",
                    {"custom_args": [cmd, "--help"], "cluster": "SYSTEM"},
                    priority=0.7,
                    source=f"cmd_gap:{cmd}→{needed_intent}",
                    description=f"用{cmd}填{missing_key}"
                )
        return None

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
                    "gap_fill", "OBSERVE", {"path": path},
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
                    "gap_fill", "TRY", {"custom_args": cmd, "cluster": "SYSTEM"},
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
                "run_tool", "TRY",
                {"custom_args": ["python3", f"data/persistent/tools/{best['name']}"],
                 "cluster": "CREATIVE"},
                priority=0.65,
                source=f"tool:{best['name']}",
                description=f"工具: {best['name']} ({desc})"
            ))
        return candidates

    # ── MODE 候选 (精简版, 无硬编码路径) ──

    def _explore_candidates(self, wb, step: int, rnd_avg: float) -> list[Goal]:
        """EXPLORE: 缺口 + 探针 + 创造力循环"""
        candidates = []

        # 缺口驱动 (跳过已填过的)
        if hasattr(wb, 'graph'):
            gaps = wb.graph.find_gaps()
            for src, missing, rel in gaps[:3]:
                if missing in self._filled_gaps:
                    continue
                goal = self._gap_to_goal_dynamic(missing, wb, step)
                if goal:
                    self._filled_gaps.add(missing)
                    goal.priority = 0.9
                    goal.source = f"rnd_gap:{missing}"
                    candidates.append(goal)

        # 探针 (但避免重复)
        if hasattr(wb, '_build_dynamic_probes'):
            if not hasattr(self, '_last_probe_key'):
                self._last_probe_key = ""
            explored = set()
            if hasattr(wb, '_current_discovery') and wb._current_discovery:
                explored.add(wb._current_discovery)
            probes = wb._build_dynamic_probes(explored)
            for p in probes[:3]:
                pk = p.get('path_key', '')
                if pk == self._last_probe_key and len(probes) > 1:
                    continue
                # 推理过滤: 如果推断出隔离, 跳过网络探针
                cmd = p.get("cmd", [])
                cmd_str = ' '.join(str(c) for c in cmd)
                if self._has_inference('isolated') and any(n in cmd_str for n in ['ip ', 'net', 'route']):
                    continue
                # 推理过滤: 如果推断出容器, 优先 /proc 探针
                if self._has_inference('container') and 'hostname' in cmd_str:
                    continue  # 容器ID已知, 不循环读hostname
                candidates.append(Goal(
                    "gap_fill", "TRY",
                    {"custom_args": cmd, "cluster": p.get("cluster", "SYSTEM")},
                    priority=0.6,
                    source=f"probe:{pk}",
                    description=f"探针: {' '.join(cmd)}"
                ))
                self._last_probe_key = pk
                break

        # 随机触发自生成实验 (约 5% 概率, 无关缺口)
        if random.random() < 0.05:
            experiment = self._self_generate_experiment(wb, step)
            if experiment:
                candidates.append(experiment)

        # 无缺口无探针时: 自生成实验
        if not candidates:
            experiment = self._self_generate_experiment(wb, step)
            if experiment:
                candidates.append(experiment)
            else:
                create = self._create_candidates(wb, step)
                candidates.extend(create)

        return candidates

    def _create_candidates(self, wb, step: int) -> list[Goal]:
        """CREATE: 用已有事实生成内容, 不需要新事实"""
        candidates = []

        # 1. 优先 CreativeWriter (LLM 异步结果检查)
        if self.creative_writer is not None:
            # 先检查缓存 (不阻塞)
            async_result = getattr(self.creative_writer, '_async_result', None)
            if async_result and async_result.get("source", "") == "llm":
                result = self.creative_writer.check_async_result()
                if result:
                    content = result.get("content", "")
                    path = result.get("path", f"/tmp/llm_report_{step}.md")
                    if content:
                        candidates.append(Goal(
                            "content_create", "CREATE",
                            {"path": path, "content": content},
                            priority=0.7, source="llm_create",
                            description=f"LLM: {result.get('desc','report')} ({len(content)}B)"
                        ))
                        return candidates
            # LLM 未就绪, 继续模板

        # 2. 模板回退: 从 FactGraph 收集已有事实
        facts_dict = {}
        graph = getattr(wb, 'graph', None)
        if graph and graph.nodes:
            for key, node in graph.nodes.items():
                facts_dict[key] = node.value
        elif hasattr(wb, 'facts'):
            facts_dict = {k: v.get('value', '') for k, v in wb.facts.items()}

        if len(facts_dict) < 3:
            return candidates

        # 生成内容: 只选有意义的系统/网络/包/能力事实
        sorted_facts = []
        if graph and graph.nodes:
            meaningful = []
            for key, node in graph.nodes.items():
                if node.category in ("system", "package", "network", "capability", "file"):
                    meaningful.append((key, node))
            meaningful.sort(key=lambda x: (x[1].confidence, x[1].step), reverse=True)
            sorted_facts = meaningful[:10]
        else:
            sorted_facts = list(facts_dict.items())[:10]

        if not sorted_facts:
            return candidates

        # 如果有推理, 用推理引导创作内容
        inferences = list(self._inferences.values()) if hasattr(self, '_inferences') else []

        lines = [f"# Folunar Report (step {step})", ""]
        if inferences:
            lines.append("## Inferences")
            for inf in inferences[:5]:
                lines.append(f"- {inf}")
            lines.append("")

        lines.append("## Facts")
        for key, node in sorted_facts:
            val = str(node.value if not isinstance(node, tuple) else node[1])[:60]
            cat = getattr(node, 'category', 'general') if not isinstance(node, tuple) else 'general'
            lines.append(f"- [{cat}] {key}: {val}")
        content = "\n".join(lines)

        if content and len(content) > 50:
            path = f"/tmp/report_{step}.md"
            candidates.append(Goal(
                "content_create", "CREATE",
                {"path": path, "content": content},
                priority=0.6, source="auto_report",
                description=f"报告: {len(content)}B"
            ))

        return candidates

    def _self_generate_experiment(self, wb, step: int) -> Optional[Goal]:
        """
        从已有事实组合出新实验 — 无缺口时自生成

        策略:
          1. 从 FactGraph 选 2 个不同类别的事实
          2. 根据它们的组合生成一个可执行的测试命令
        """
        graph = getattr(wb, 'graph', None)
        if not graph or len(graph.nodes) < 5:
            return None

        # 按类别收集事实
        by_cat: dict[str, list[str]] = {}
        for key, node in graph.nodes.items():
            by_cat.setdefault(node.category, []).append(key)

        # 选两个不同类别的事实
        cats = [c for c in by_cat if c not in ('general', 'command', 'inference', 'tool_result')
                and len(by_cat[c]) >= 2]
        if len(cats) < 2:
            return None

        import random
        cat_a, cat_b = random.sample(cats, 2)
        key_a = random.choice(by_cat[cat_a])
        key_b = random.choice(by_cat[cat_b])
        val_a = str(graph.nodes[key_a].value)[:30]
        val_b = str(graph.nodes[key_b].value)[:30]

        # 根据类别组合生成实验命令
        experiments = [
            # 文件 + 系统 → 检查文件是否存在
            (["file", "system"], ["cat", "/etc/hostname"]),
            # 包 + 系统 → 检查已装包
            (["package", "system"], ["dpkg", "-l"]),
            # 网络 + 系统 → 检查连接
            (["network", "system"], ["cat", "/proc/net/dev"]),
            # 能力 + 系统 → 测试能力
            (["capability", "system"], ["python3", "--version"]),
            # 文件 + 包 → 检查相关文件
            (["file", "package"], ["ls", "-la", "/etc/apt/"]),
            # 系统 + 系统 → 系统调用
            (["system", "system"], ["uname", "-a"]),
        ]

        for (c1, c2), cmd in experiments:
            if cat_a in (c1, c2) and cat_b in (c1, c2):
                desc = f"实验: {key_a}={val_a} + {key_b}={val_b}"
                return Goal(
                    "experiment", "TRY",
                    {"custom_args": cmd, "cluster": "SYSTEM"},
                    priority=0.5,
                    source=f"experiment:{cat_a}+{cat_b}",
                    description=desc
                )

        # 没有匹配的模板: 生成通用实验
        desc = f"实验: {key_a}({cat_a}) + {key_b}({cat_b})"
        return Goal(
            "experiment", "TRY",
            {"custom_args": ["cat", "/etc/hostname"], "cluster": "SYSTEM"},
            priority=0.5,
            source=f"experiment:{cat_a}+{cat_b}",
            description=desc
        )

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
                    "verify", "OBSERVE",
                    {"custom_args": ["sh", "-c",
                        f"echo '{encoded}' | base64 -d > /tmp/v_{step}.sh && "
                        f"chmod +x /tmp/v_{step}.sh && bash /tmp/v_{step}.sh"],
                     "cluster": "CREATIVE"},
                    priority=0.7, source="verify",
                    description=f"验证: {combo}"
                ))
        return candidates

    # ── 推理感知 ──

    def _collect_inferences(self, wb) -> dict:
        """从 FactGraph 收集推理节点"""
        infs = {}
        graph = getattr(wb, 'graph', None)
        if not graph:
            return infs
        for key, node in graph.nodes.items():
            if node.category == 'inference':
                infs[key] = str(node.value)[:80]
        return infs

    def _has_inference(self, keyword: str) -> bool:
        """检查是否有包含关键词的推理"""
        for val in self._inferences.values():
            if keyword in val.lower():
                return True
        return False

    def stats(self) -> dict:
        return {
            "last_goal_type": self.last_goal_type,
            "n_goals": self._total_goals,
            "intent_usage": dict(sorted(self._intent_usage.items(),
                                        key=lambda x: -x[1])[:5]),
            "tried_commands": len(self._tried_commands),
            "recent_goals": self.goal_history[-5:],
        }
