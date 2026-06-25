"""
目标生成器 (GoalGenerator) — 将 MODE + 事实图缺口 → 具体可执行目标

人脑类比: 基底核 + 前扣带回 — 把抽象意图分解为子目标

流程:
  MODE (探索/创作/学习)
    → GoalGenerator.collect_candidates() 收集候选目标
    → GoalGenerator.prioritize() 按分数排序
    → GoalGenerator.select() 选择最高分的目标

目标类型:
  - gap_fill: 填事实缺口 (来自 FactGraph.find_gaps())
  - content_create: 生成内容 (WRITE/GENERATE)
  - verify: 真实验证 (检查事实是否仍然成立)
  - curiosity: 好奇心驱动 (RND 高新颖度方向)
  - chain: 链式验证 (从 Workbench 的 follow-up 链)
"""

from typing import Optional


class Goal:
    """一个可执行目标"""

    def __init__(self, goal_type: str, intent: str, params: dict,
                 priority: float = 0.5, source: str = "",
                 description: str = ""):
        self.type = goal_type          # gap_fill | content_create | verify | curiosity | chain
        self.intent = intent            # READ | CUSTOM | WRITE | GENERATE | ...
        self.params = params           # 执行参数
        self.priority = priority       # 0~1
        self.source = source           # 来源说明
        self.description = description

    def to_tuple(self) -> tuple[str, dict]:
        return (self.intent, self.params)


class GoalGenerator:
    """
    目标生成器

    用法:
      gg = GoalGenerator()
      goal = gg.generate(mode="EXPLORE", ...)
      # → Goal(type="gap_fill", intent="READ", params={"path": "/proc/cpuinfo"})
    """

    def __init__(self, min_facts_for_create: int = 5):
        self.min_facts_for_create = min_facts_for_create
        self.last_goal_type: Optional[str] = None
        self.goal_history: list[dict] = []

    def generate(self, mode: str, workbench=None, rnd_avg: float = 0.0,
                 step: int = 0, recent_intents: Optional[list[str]] = None) -> Optional[Goal]:
        """
        根据 MODE + 当前状态生成最佳目标

        Args:
          mode: EXPLORE | CREATE | LEARN
          workbench: Workbench 实例 (含 graph + facts + follow-up)
          rnd_avg: RND 新颖度均值
          step: 当前步数
          recent_intents: 最近执行的意图列表

        Returns:
          Goal 或 None (无合适目标时)
        """
        if workbench is None:
            return None

        recent = recent_intents or []
        candidates = []

        # ── MODE 驱动的候选目标收集 ──

        if mode == "EXPLORE":
            candidates = self._explore_candidates(workbench, step, rnd_avg)
        elif mode == "CREATE":
            candidates = self._create_candidates(workbench, step)
        elif mode == "LEARN":
            candidates = self._learn_candidates(workbench, step)

        # ── MODE-specific repetition guard ──
        if recent:
            custom_count = sum(1 for i in recent[-10:] if i == "CUSTOM")
            custom_ratio = custom_count / min(len(recent), 10)
        else:
            custom_ratio = 0.0

        # EXPLORE 下 CUSTOM 占比 ≥ 60% 硬性过滤所有 CUSTOM
        if mode == "EXPLORE" and custom_ratio >= 0.6:
            candidates = [c for c in candidates if c.intent != "CUSTOM"]

        # ── 条件化 follow_up: 只在链进行中或图有缺口时加入 ──
        in_chain = workbench.has_active_goal() if hasattr(workbench, 'has_active_goal') else False
        has_gaps = len(workbench.graph.find_gaps()) > 0 if hasattr(workbench, 'graph') and hasattr(workbench.graph, 'find_gaps') else False
        if in_chain or has_gaps:
            if custom_ratio < 0.6:
                follow_up = workbench.get_follow_up() if hasattr(workbench, 'get_follow_up') else None
                if follow_up:
                    intent, params = follow_up
                    fu_priority = 0.7
                    if intent == "CUSTOM" and custom_ratio > 0.3:
                        fu_priority = 0.2
                    candidates.append(Goal(
                        "chain", intent, params,
                        priority=fu_priority, source="follow_up",
                        description=f"链式: {intent}"
                    ))

        # ── CUSTOM 硬性过滤: 占比 > 30% 排除 CUSTOM 候选 ──
        if custom_ratio > 0.3 and len(candidates) > 1:
            non_custom = [c for c in candidates if c.intent != "CUSTOM"]
            if non_custom:
                candidates = non_custom

        # ── 优先排序 ──
        if not candidates:
            return None
        candidates.sort(key=lambda g: -g.priority)

        # ── 防止连续同类型 ──
        if self.last_goal_type and len(candidates) > 1:
            if candidates[0].type == self.last_goal_type:
                for c in candidates[1:]:
                    if c.type != self.last_goal_type:
                        candidates.insert(0, c)
                        break

        # P10: utility gate — 没好目标就让 A/B 路径接管
        if candidates[0].priority < 0.6:
            return None

        selected = candidates[0]

    # ── MODE 专用候选生成 ──

    def _explore_candidates(self, wb, step: int, rnd_avg: float) -> list[Goal]:
        """EXPLORE 模式: 事实缺口 + 好奇心 + 探针"""
        candidates = []

        # 1. FactGraph 缺口 (最高优先)
        if hasattr(wb, 'graph'):
            gaps = wb.graph.find_gaps()
            for src, missing, rel in gaps[:3]:
                goal = self._gap_to_goal(missing, wb, step)
                if goal:
                    goal.priority = 0.9
                    goal.source = f"gap:{rel}:{missing}"
                    candidates.append(goal)

        # 2. RND 好奇心 (新颖度高时探索)
        if rnd_avg > 0.03:
            # 从已知事实的反向方向探索
            system_keys = wb.get_facts_by_category("system") if hasattr(wb, 'get_facts_by_category') else []
            if system_keys:
                # 选一个未探索的系统文件
                unread_paths = [
                    "/proc/version", "/proc/loadavg", "/proc/uptime",
                    "/proc/stat", "/proc/partitions", "/proc/modules",
                    "/etc/resolv.conf", "/etc/fstab", "/etc/timezone",
                    "/proc/1/status", "/proc/self/status",
                ]
                for p in unread_paths:
                    key = p.replace("/", "_").strip("_")
                    if key not in wb.facts and (
                        not hasattr(wb, 'graph') or key not in wb.graph.nodes
                    ):
                        candidates.append(Goal(
                            "curiosity", "READ", {"path": p},
                            priority=0.8, source=f"rnd_unread:{p}",
                            description=f"好奇心: 读 {p}"
                        ))
                        break

        # 3. 探针: _build_dynamic_probes
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
                    priority=p.get("base_score", 0.5),
                    source=f"probe:{p.get('path_key', '')}",
                    description=f"探针: {' '.join(cmd)}"
                ))

        return candidates

    def _create_candidates(self, wb, step: int) -> list[Goal]:
        """CREATE 模式: 内容生成 + 报告"""
        candidates = []

        if not hasattr(wb, 'build_write_content'):
            return candidates

        # 1. GENERATE (优先: 最丰富的内容)
        n_facts = len(wb.facts) if hasattr(wb, 'facts') else 0
        if n_facts >= self.min_facts_for_create:
            ci = wb.build_generate_content()
            candidates.append(Goal(
                "content_create", "GENERATE",
                {"path": ci["path"], "content": ci["content"]},
                priority=0.9, source="build_generate",
                description=f"生成: {ci['desc']} ({ci['size']}B)"
            ))

        # 2. WRITE (结构化报告)
        if n_facts >= 3:
            ci = wb.build_write_content()
            candidates.append(Goal(
                "content_create", "WRITE",
                {"path": ci["path"], "content": ci["content"]},
                priority=0.7, source="build_write",
                description=f"创作: {ci['desc']} ({ci.get('size', 0)}B)"
            ))

        return candidates

    def _learn_candidates(self, wb, step: int) -> list[Goal]:
        """LEARN 模式: 真实验证 + 训练"""
        candidates = []

        # 1. 真实验证: 用脚本验证已知事实
        if hasattr(wb, 'generate_script'):
            result = wb.generate_script()
            if result:
                script, combo = result
                # 写脚本 + 执行
                import base64
                encoded = base64.b64encode(script.encode()).decode()
                candidates.append(Goal(
                    "verify", "CUSTOM",
                    {"custom_args": ["sh", "-c",
                        f"echo '{encoded}' | base64 -d > /tmp/verify_{step}.sh && "
                        f"chmod +x /tmp/verify_{step}.sh && bash /tmp/verify_{step}.sh"],
                     "cluster": "CREATIVE"},
                    priority=0.8, source="verify_script",
                    description=f"验证: combo={combo}"
                ))

        # 2. 交叉验证: 读一个已知文件确认事实
        known_keys = list(wb.facts.keys()) if hasattr(wb, 'facts') else []
        if known_keys:
            key = known_keys[-1]
            fact = wb.facts.get(key, {})
            if fact and 'source_cmd' in fact:
                src = fact.get('source_cmd', '')
                if src and src.startswith('['):
                    import ast
                    try:
                        cmd_list = ast.literal_eval(src)
                        if isinstance(cmd_list, list):
                            candidates.append(Goal(
                                "verify", "CUSTOM",
                                {"custom_args": cmd_list, "cluster": "SYSTEM"},
                                priority=0.5, source=f"reverify:{key}",
                                description=f"重验: {key}"
                            ))
                    except:
                        pass

        return candidates

    # ── 辅助 ──

    def _gap_to_goal(self, missing_key: str, wb, step: int) -> Optional[Goal]:
        """将事实缺口映射为可执行目标"""
        gap_map = {
            "os_version_id": ("READ", {"path": "/etc/os-release"}),
            "os_version_codename": ("READ", {"path": "/etc/os-release"}),
            "kernel_release": ("CUSTOM", {"custom_args": ["uname", "-a"], "cluster": "SYSTEM"}),
            "cpu_model": ("READ", {"path": "/proc/cpuinfo"}),
            "mem_total": ("CUSTOM", {"custom_args": ["free", "-h"], "cluster": "SYSTEM"}),
            "swap_total": ("CUSTOM", {"custom_args": ["free", "-h"], "cluster": "SYSTEM"}),
            "etchosts_hosts": ("READ", {"path": "/etc/hosts"}),
            "hostname_cmd": ("CUSTOM", {"custom_args": ["hostname"], "cluster": "SYSTEM"}),
            "uid_info": ("CUSTOM", {"custom_args": ["id"], "cluster": "USER"}),
            "current_user": ("CUSTOM", {"custom_args": ["whoami"], "cluster": "USER"}),
            "ip_addr": ("CUSTOM", {"custom_args": ["ip", "addr"], "cluster": "NETWORK"}),
            "mac_addr": ("CUSTOM", {"custom_args": ["ip", "addr"], "cluster": "NETWORK"}),
            "disk_persistent": ("CUSTOM", {"custom_args": ["df", "-h"], "cluster": "SYSTEM"}),
        }
        if missing_key in gap_map:
            intent, params = gap_map[missing_key]
            return Goal(
                "gap_fill", intent, params,
                priority=0.85, source=f"gap:{missing_key}",
                description=f"填缺口: {missing_key}"
            )
        return None

    def stats(self) -> dict:
        return {
            "last_goal_type": self.last_goal_type,
            "n_goals_generated": len(self.goal_history),
            "recent_goals": self.goal_history[-5:],
        }
