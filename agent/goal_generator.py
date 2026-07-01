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
        self._format_stats: dict[str, dict[str, int]] = {
            "code": {"ok": 0, "fail": 0},
            "analysis": {"ok": 0, "fail": 0},
            "report": {"ok": 0, "fail": 0},
        }
        self._inferences: dict = {}
        # P20: 多步计划编排
        self._active_plan: Optional[Plan] = None
        self._plan_history: list = []
        self._plan_probability = 0.40  # 40% 概率输出计划而非单句


    def decide_creative_intention(self, wb, step: int) -> dict:
        """小模型思维: 看数据, 产生真实的好奇心, 发具体任务给 DeepSeek 实现"""
        if not hasattr(self, '_create_type_history'):
            self._create_type_history = []

        facts = wb.facts if hasattr(wb, 'facts') and wb.facts else {}
        cats = {}
        for k, v in facts.items():
            cat = v.get('category', 'general') if isinstance(v, dict) else 'general'
            cats[cat] = cats.get(cat, 0) + 1

        observations = []
        
        # 数值类观察: 扫描全部事实, 发现有趣的值
        key_patterns = {
            ("kernel-limit", "shmmax", "shmall", "shm"): lambda v: v > 1e15,
            ("cpu", "cpu_core", "nproc", "core", "cpu"): lambda v: 1 < v < 10000,
            ("memory", "mem", "memory", "swap", "anon", "cache", "buffer"): lambda v: v > 1e6,
            ("fs-limit", "nr_open", "file-max", "inode", "max_files", "max_user"): lambda v: v > 1e6,
            ("process", "pid_max", "process", "pid"): lambda v: 100 < v < 1e8,
            ("erratic", "tcp_ehash", "negative", "invalid"): lambda v: v < 0,
        }
        for k, v in facts.items():
            if isinstance(v, dict): v_raw = v.get('value', '')
            else: v_raw = v
            try: fv = float(v_raw)
            except: continue
            kl = k.lower()
            for (tag, *keywords), check in key_patterns.items():
                if any(kw in kl for kw in keywords):
                    if check(fv):
                        short = k.split('.')[-1]
                        if tag == "kernel-limit":
                            observations.append((tag, f"{short}={fv:.0e}"))
                        elif tag == "memory":
                            observations.append((tag, f"{short}={fv/1e6:.0f}MB"))
                        elif tag == "cpu":
                            observations.append((tag, f"{short}={fv}"))
                        else:
                            observations.append((tag, f"{short}={fv:.0e}" if abs(fv) > 1e6 else f"{short}={fv:.0f}"))
                        break

        # P19: CodeArchive — 自我成长观测
        if hasattr(self, 'code_archive') and self.code_archive:
            try:
                arch_obs = self.code_archive.get_observations()
                observations.extend(arch_obs)
            except Exception:
                pass
        # P18: WM5 Surprise signal — prediction error drives creative direction
        recent_surprise = getattr(self, '_recent_surprise', [])
        if recent_surprise and len(recent_surprise) >= 5:
            avg_surprise = sum(recent_surprise[-10:]) / max(len(recent_surprise[-10:]), 1)
            # Adaptive thresholds via running statistics
            if not hasattr(self, '_surprise_running'):
                self._surprise_running = []
            self._surprise_running.append(avg_surprise)
            if len(self._surprise_running) > 100:
                self._surprise_running = self._surprise_running[-100:]
            if len(self._surprise_running) >= 10:
                # Running statistics for adaptive thresholds
                run_mean = sum(self._surprise_running) / len(self._surprise_running)
                run_var = sum((s - run_mean)**2 for s in self._surprise_running) / len(self._surprise_running)
                run_std = max(run_var ** 0.5, 1e-8)
                z = (avg_surprise - run_mean) / run_std
                if z < -1.0:
                    # Significantly below average → predictable → random explore
                    observations.append(("boredom",
                        f"z_surprise={z:.2f} (too predictable)"))
                elif z > 1.5:
                    # Significantly above average → unexpected → keep focus
                    if observations and hasattr(self, '_tag_history') and self._tag_history:
                        last_tag = self._tag_history[-1]
                        for otag, otxt in observations[:]:
                            if otag == last_tag:
                                observations.append((otag, otxt + " (surprise spike)"))
                                break
            else:
                # Not enough stats yet: use heuristic based on raw range
                if avg_surprise < 0.0003:
                    observations.append(("boredom",
                        f"avg_surprise={avg_surprise:.6f} (too predictable)"))
        if not observations:
            observations.append(("random-explore", "nothing stands out"))
        
        # 标签级多样性: 避免同一观察类型被连续选
        if not hasattr(self, '_tag_history'):
            self._tag_history = []
        recent_tags = self._tag_history[-4:]
        weighted_obs = []
        for otag, otxt in observations:
            penalty = 0.5 if otag in recent_tags else 0.0
            score = 1.0 - penalty
            weighted_obs.append((score, otag, otxt))
        scores = [s for s, _, _ in weighted_obs]
        total = sum(scores) + 0.001
        probs = [s / total for s in scores]
        idx = random.choices(range(len(weighted_obs)), weights=probs, k=1)[0]
        _, tag, obs_text = weighted_obs[idx]
        self._tag_history.append(tag)
        if len(self._tag_history) > 20:
            self._tag_history = self._tag_history[-20:]
        
        # 基于观察类型选择最匹配的模板组
        tag_mode_map = {
            "cpu":       "cpu",
            "kernel-limit": "kernel-limit",
            "memory":    "memory",
            "erratic":   "erratic",
            "fs-limit":  "fs-limit",
            "process":   "process",
            "network":   "network",
            "command-catalog": "command-catalog",
            "content-overload": "content-overload",
            "no-inference": "no-inference",
            "system":    "system",
            "boredom":   "boredom",
            "code-archive": "code-archive",
            "code-growth": "code-growth",
            "code-stuck": "code-stuck",
            "code-compose": "code-compose",
            "code-milestone": "code-milestone",
            "random-explore": "random-explore",
        }

        template_key = tag_mode_map.get(tag, "random-explore")
        ideas = self._generate_ideas(template_key, obs_text)
        
        if not ideas:
            ideas = [(0.3, "code", f"I wonder what happens if I explore /proc/sys more deeply. Let me write a Python script that reads every writable file in /proc/sys and reports current values.")]

        # ── 阶段3: 多样性筛选 ──
        recent = self._create_type_history[-6:]
        scored = []
        for score, style, idea in ideas:
            # 与近期想法去重 (基于关键词重叠)
            idea_words = set(idea.lower().split())
            overlap = 0.0
            for h in recent:
                h_words = set(h.lower().split())
                jac = len(idea_words & h_words) / max(len(idea_words | h_words), 1)
                overlap = max(overlap, jac)
            penalty = 0.6 if overlap > 0.4 else 0.0
            scored.append((score - penalty, style, idea))
        
        scored.sort(key=lambda x: -x[0])
        score, style, idea = scored[0]
        self._create_type_history.append(idea)
        if len(self._create_type_history) > 50:
            self._create_type_history = self._create_type_history[-50:]
        return {"style": style, "intention": idea, "category": "idea"}

    def decide_plan(self, wb, step: int):
        """动态生成多步计划 — 零模板, 全由结构化组合产生"""
        if not hasattr(self, '_plan_probability'):
            return None
        if random.random() > self._plan_probability:
            return None
        from agent.planner import Plan
        # 1. 找当前观察标签
        facts = wb.facts if hasattr(wb, 'facts') and wb.facts else {}
        tags = []
        for k, v in facts.items():
            if isinstance(v, dict):
                cat = v.get('category', '')
                if cat and cat not in ('general', 'content', 'meta'):
                    tags.append(cat)
        tag_counts = {}
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        if not tag_counts:
            return None
        top_tag = max(tag_counts, key=tag_counts.get)
        # 2. 生成步骤链: 保证 code → analysis → report 各一步
        steps = []
        formats_needed = ["code", "analysis", "report"]  # 构建→分析→报告
        # 打乱前两个顺序但保持报告在最后
        random.shuffle(formats_needed[:-1])
        for i, fmt in enumerate(formats_needed):
            # 临时设 format_bias 只产出当前格式
            old_bias = self._format_bias_cache if hasattr(self, '_format_bias_cache') else None
            candidates = self._generate_ideas(top_tag, f"{top_tag}=observed")
            # 过滤当前格式
            matching = [c for c in candidates if c[1] == fmt]
            if matching:
                _, style, idea = random.choice(matching)
            else:
                # 没匹配到: 用通用描述
                domains = {"cpu":"CPU layer","kernel-limit":"kernel parameters",
                    "memory":"memory hierarchy","fs-limit":"filesystem limits",
                    "process":"process lifecycle","network":"network topology",
                    "erratic":"anomalous behavior","system":"system state"}
                domain = domains.get(top_tag, "system")
                fmt_verbs = {"code":["build","probe","monitor","catalog"],
                    "analysis":["analyze","study","examine"],
                    "report":["document","report","summarize"]}
                verb = random.choice(fmt_verbs.get(fmt, ["explore"]))
                idea = f"I want to {verb} the {domain}"
            if fmt == "code":
                deps = []
                desc = idea + ". Build the tool/module that collects data."
            elif fmt == "analysis":
                deps = [s["id"] for s in steps]  # 依赖前面所有已完成步骤
                desc = idea + ". Use data from prior steps for analysis."
            else:
                deps = [s["id"] for s in steps]
                desc = idea + ". Synthesize all prior steps into a structured report."
            steps.append({"id": i, "style": fmt, "desc": desc,
                "deps": deps, "done": False, "file": ""})
        if not steps:
            return None
        domains = {"cpu":"CPU layer","kernel-limit":"kernel parameter space",
            "memory":"memory hierarchy","erratic":"anomalous behavior",
            "fs-limit":"filesystem boundary","process":"process lifecycle",
            "network":"network topology","system":"system baseline"}
        plan = Plan(f"plan_{step}", steps, created_step=step,
                    topic=domains.get(top_tag, "system state"))
        self._active_plan = plan
        self._plan_history.append(plan.to_dict())
        print(f"  [PLAN] {plan.plan_id}: {len(plan.steps)}步 (零模板), topic=\"{plan.topic}\"")
        for s in plan.steps:
            print(f"    步{s['id']}: [{s['style']:10s}] {s['desc'][:70]}")
        return plan



    
    def record_format_result(self, fmt: str, success: bool):
        """记录格式执行结果, 用于自适应调权"""
        if fmt not in self._format_stats:
            self._format_stats[fmt] = {"ok": 0, "fail": 0}
        if success:
            self._format_stats[fmt]["ok"] += 1
        else:
            self._format_stats[fmt]["fail"] += 1
    
    def _generate_ideas(self, tag: str, obs: str) -> list:
        """结构化组合 + 输出格式选择"""
        KEY = obs.split("=")[0].strip() if "=" in obs else tag
        VALUE = obs.split("=")[-1].strip() if "=" in obs else obs
        
        domains = {
            "cpu": "CPU layer", "kernel-limit": "kernel parameter space",
            "memory": "memory hierarchy", "erratic": "anomalous behavior",
            "fs-limit": "filesystem boundary", "process": "process lifecycle",
            "network": "network topology", "command-catalog": "command ecosystem",
            "content-overload": "content collection", "no-inference": "correlation landscape",
            "system": "system baseline", "boredom": "unexplored territory",
            "random-explore": "unknown subsystem",
            "code-archive": "my code history", "code-growth": "my skill trajectory",
            "code-stuck": "my current plateau", "code-compose": "my script collection",
            "code-milestone": "my recent output",
        }
        domain = domains.get(tag, "system state")
        source = self._derive_source(tag, KEY, VALUE)
        
        # 输出格式: 不同标签偏好不同格式
        format_bias = {
            "cpu":  {"code": 0.5, "analysis": 0.4, "report": 0.1},
            "kernel-limit":  {"code": 0.6, "analysis": 0.3, "report": 0.1},
            "memory":  {"code": 0.4, "analysis": 0.5, "report": 0.1},
            "erratic":  {"code": 0.3, "analysis": 0.6, "report": 0.1},
            "fs-limit":  {"code": 0.5, "analysis": 0.4, "report": 0.1},
            "process":  {"code": 0.4, "analysis": 0.5, "report": 0.1},
            "network":  {"code": 0.4, "analysis": 0.5, "report": 0.1},
            "command-catalog":  {"code": 0.2, "analysis": 0.6, "report": 0.2},
            "content-overload":  {"code": 0.2, "analysis": 0.6, "report": 0.2},
            "no-inference":  {"code": 0.3, "analysis": 0.6, "report": 0.1},
            "system":  {"code": 0.2, "analysis": 0.3, "report": 0.5},
            "boredom":  {"code": 0.5, "analysis": 0.3, "report": 0.2},
            "random-explore":  {"code": 0.4, "analysis": 0.4, "report": 0.2},
            "code-archive":  {"code": 0.6, "analysis": 0.2, "report": 0.2},
            "code-growth":  {"code": 0.6, "analysis": 0.3, "report": 0.1},
            "code-stuck":  {"code": 0.6, "analysis": 0.3, "report": 0.1},
            "code-compose":  {"code": 0.6, "analysis": 0.3, "report": 0.1},
            "code-milestone":  {"code": 0.6, "analysis": 0.3, "report": 0.1},
        }
        probs = format_bias.get(tag, {"code": 0.5, "analysis": 0.3, "report": 0.2})
        # 经验调整: 某个格式总失败就降权
        if hasattr(self, '_format_stats'):
            for fmt in list(probs.keys()):
                s = self._format_stats.get(fmt, {"ok": 1, "fail": 1})
                ratio = s["ok"] / max(s["ok"] + s["fail"], 1)
                if ratio < 0.2:
                    probs[fmt] *= 0.3
        fmt_choices = list(probs.keys())
        fmt_weights = [max(probs[f], 0.05) for f in fmt_choices]
        output_format = random.choices(fmt_choices, weights=fmt_weights, k=1)[0]
        
        # 格式决定动词表和框架
        if output_format == "analysis":
            verbs = ["analyze", "examine", "investigate", "study", "review"]
            frames = [
                "I want to analyze the {domain} using {source}",
                "I want to study {domain} — examine {source} data",
                "I want to investigate {domain} through {source}",
                "I want to look into {domain} by checking {source}",
            ]
        elif output_format == "report":
            verbs = ["document", "summarize", "report", "record", "describe"]
            frames = [
                "I want to document the {domain} from {source}",
                "I want a report on {domain} — data from {source}",
                "I want to record findings about {domain} using {source}",
                "I want to summarize what {source} tells us about {domain}",
            ]
        else:
            verbs = ["analyze", "benchmark", "catalog", "cross-reference",
                     "monitor", "probe", "stress-test", "validate", "visualize"]
            frames = [
                "I want to write a Python script that {verb}s the {domain} from {source}",
                "I want write Python code to {verb} {domain} — data: {source}",
                "I want to build a tool that {verb}s {domain} via {source}",
                "I want a Python script to {verb} the {domain} using {source}",
            ]
        
        selected = random.choices(verbs, k=min(4, len(verbs)))
        result = []
        for verb in selected:
            frame = random.choice(frames)
            text = frame.format(verb=verb, domain=domain, source=source)
            result.append((0.8, output_format, text))
        return result
    
    def _derive_source(self, tag: str, key: str, value: str) -> str:
        """从事实的键名推导数据源 — 无硬编码路径"""
        kl = key.lower()
        if 'cmd_' in kl or 'command' in kl:
            return "package metadata"
        if 'generic_kernel' in kl:
            return "/proc/sys/kernel"
        if 'generic_net' in kl:
            return "/proc/sys/net"
        if 'generic_fs' in kl:
            return "/proc/sys/fs"
        if 'llm_' in kl or 'content' in kl:
            return "generated content files"
        if 'inf_' in kl:
            return "inference results"
        if 'hostname' in kl or 'node' in kl:
            return "system identification"
        if 'mem' in kl or 'swap' in kl:
            return "/proc/meminfo"
        if 'cpu' in kl or 'nproc' in kl:
            return "/proc/cpuinfo"
        if 'load' in kl or 'uptime' in kl:
            return "/proc/loadavg"
        if 'net' in kl or 'tcp' in kl or 'ip' in kl:
            return "/proc/net"
        if 'pid' in kl or 'process' in kl or 'thread' in kl:
            return "/proc filesystem"
        if 'sch' in kl or 'sched' in kl:
            return "/proc/sched_debug"
        if 'interrupt' in kl or 'irq' in kl:
            return "/proc/interrupts"
        if 'disk' in kl or 'io' in kl or 'block' in kl:
            return "/proc/diskstats"
        if 'module' in kl or 'driver' in kl or 'device' in kl:
            return "/sys filesystem"
        if 'archive' in kl or 'code' in kl:
            return "my code archive"
        fbs = {
            "kernel-limit": "/proc/sys", "fs-limit": "/proc/sys/fs",
            "erratic": "/proc/sys", "random-explore": "/proc and /sys",
            "boredom": "alternate /proc interfaces",
        }
        return fbs.get(tag, "/proc filesystem")
    
    def generate(self, mode: str, workbench=None, rnd_avg: float = 0.0,
                 step: int = 0, recent_intents: Optional[list[str]] = None,
                 force_create: bool = False,
                 knowledge_mapper=None, tool_registry=None,
                 hypothesis_engine=None, fact_graph=None) -> Optional[Goal]:
        """
        动态生成最佳目标 — 所有决策由数据驱动
        """
        candidates: list[Goal] = []
    
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
        if recent_intents:
            intent_freq = {i: recent_intents.count(i) for i in set(recent_intents)}
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

            # 收集推理用于过滤探针
            self._inferences = self._collect_inferences(wb)
            # 探针 (但避免重复)
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
        """CREATE: 只产出 LLM 真正生成的内容。模板不在了。"""
        # 有 LLM: 只消费异步结果, 不生成任何模板内容
        if self.creative_writer is not None:
            # 读异步结果但不消费 (check_async_result 会清除, 留给 step() 去消费)
            async_result = getattr(self.creative_writer, '_async_result', None)
            if async_result and async_result.get("source", "") == "llm":
                content = async_result.get("content", "")
                if content and len(content) > 40:
                    style = async_result.get("style", "output")
                    path = async_result.get("path", f"/tmp/llm_{style}_{step}.md")
                    return [Goal(
                        "content_create", "CREATE",
                        {"path": path, "content": content},
                        priority=0.8, source="llm_create",
                        description=f"LLM: {async_result.get('desc',style)} ({len(content)}B)"
                    )]
            return []

        # 无 LLM: 极简模板 fallback
        graph = getattr(wb, 'graph', None)
        facts = {}
        if graph and graph.nodes:
            for k, n in graph.nodes.items():
                facts[k] = n.value
        elif hasattr(wb, 'facts'):
            facts = {k: v.get('value','') for k, v in wb.facts.items()}
        if len(facts) < 3:
            return []
        lines = [f"# State Snapshot (step {step})"]
        for k, v in list(facts.items())[:8]:
            lines.append(f"- {k}: {str(v)[:60]}")
        c = "\n".join(lines)
        if len(c) < 40:
            return []
        return [Goal(
            "content_create", "CREATE",
            {"path": f"/tmp/state_{step}.md", "content": c},
            priority=0.5, source="fallback",
            description=f"快照: {len(c)}B"
        )]
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
        if not hasattr(self, '_inferences'):
            return False
        if not self._inferences:
            return False
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
