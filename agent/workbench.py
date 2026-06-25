"""
P4.0 Workbench — 动作记忆 + 事实提取

不只是"存文件名"。
这是一个工作台: 系统把发现摆在上面, 后续步骤可以消费这些发现。

生命周期:
  执行 → extract_facts() → fact 上工作台
  选择意图 → get_state_summary() → state_text 包含已知事实
  目标驱动 → get_current_discovery() + get_follow_up() → 推荐下一步

提取规则:
  cat /etc/hostname    → hostname = content
  uname -a             → kernel, node, arch
  cat /etc/os-release  → os_name, os_version
  free -h              → mem_total, swap_total
  df -h                → disk_{mount} = avail
  ls /tmp, /etc, ...   → dir_{name} = items
  hostname             → hostname_cmd = output
  cat /proc/cpuinfo    → cpu_cores = count
"""

import re
from typing import Optional

from agent.fact_graph import FactGraph, EDGE_REQUIRES, EDGE_VERIFIES, EDGE_EXTENDS


class Workbench:
    """工作栏: 事实存储 + 自动提取 + 状态摘要"""

    def __init__(self, max_facts: int = 40, meta_learner=None):
        self.facts: dict[str, dict] = {}  # key → {value, source, step, confidence, count} (legacy, dual-write)
        self.graph = FactGraph(max_nodes=max_facts * 5)  # P10: 图结构知识库
        self.max_facts = max_facts
        self._current_discovery: Optional[str] = None  # 最新关键事实 key
        self._step_counter = 0
        # P4.1: 链式追踪
        self.chain_step: int = 0
        self.chain_completed_at: int = 0
        self.last_follow_up: Optional[tuple[str, dict]] = None
        # P5.3: 从配置文件加载规则
        self.rules = self._load_rules()
        # P5.4: 元学习器 (外部注入)
        self.meta = meta_learner

    # ── 核心: 事实提取 ──

    def extract_facts(self, intent: str, cmd_name: str, output: str,
                      params: dict | None = None, step: int = 0):
        """从命令输出自动提取事实 (内容优先, 命令名辅助)"""
        self._step_counter = step
        if not output or len(output.strip()) < 2:
            return

        lower = cmd_name.lower()
        text = output.strip()

        # ── 内容优先的提取 (不依赖命令名) ──

        # uname -a 输出: Linux hostname 7.0.11-arch ...
        text_lines = text.splitlines()
        text_stripped = "\n".join(
            l for l in text_lines
            if not l.startswith("---")
        ).strip()
        first_line = text_lines[0] if text_lines else ""
        
        # 多命令输出: 跳过 --- [N] --- 包装, 取每段实际内容
        segments = []
        current = []
        for line in text_lines:
            if line.startswith("--- [") and line.endswith("] ---"):
                if current:
                    segments.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            segments.append("\n".join(current))

        for seg_text in segments:
            seg = seg_text.strip()
            if not seg:
                continue
            
            # uname -a (最少 4 个词: Linux hostname 5.x.y arch)
            if seg.startswith("Linux ") and len(seg.split()) >= 4:
                self._extract_uname(seg, intent, cmd_name, step)
                continue

            # os-release
            if seg.startswith("PRETTY_NAME") or (
                "release" in seg[:60] and any(
                    k in seg for k in ("PRETTY_NAME", "VERSION_ID", "VERSION_CODENAME")
                )
            ):
                self._extract_os_release(seg, intent, cmd_name, step)
                continue

            # free
            if seg.startswith("Mem:") or "Mem:" in seg[:20]:
                self._extract_free(seg, intent, cmd_name, step)
                continue

            # df
            seg_first = seg.splitlines()[0] if seg.splitlines() else ""
            if seg.startswith("/dev/") or seg_first.startswith("Filesystem"):
                self._extract_df(seg, intent, cmd_name, step)
                continue

            # ls
            if seg_first.startswith(("total", "drwx", "-rw", "-r-", "lrwx", "crw", "brw", "srw")):
                path = ""
                if params and "path" in params:
                    path = params["path"]
                self._extract_ls(seg, intent, cmd_name, step, path)
                continue

            # cpuinfo
            if seg_first.startswith("processor") or "processor" in seg[:20]:
                self._extract_cpuinfo(seg, intent, cmd_name, step)
                continue

            # passwd
            if "root:x:0:0" in seg[:60]:
                self._extract_passwd(seg, intent, cmd_name, step)
                continue


        # ── 命令名辅助的回退 (内容未命中时的备用规则) ──
        # 这些规则不使用 segments (因为内容模式不匹配),
        # 而是直接检查命令名和输出结构

        # cat /etc/hostname: 输出是短单行, 命令名含 hostname
        if "hostname" in lower and ("cat" in lower or "etc" in lower):
            lines = text.splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("#", ";")):
                    self._add_fact("hostname", line, intent, cmd_name, step, category="system")
                    self._current_discovery = "hostname"
                    break

        # wc -l /etc/passwd: 输出 "123 /etc/passwd"
        if "passwd" in lower and "wc" in lower:
            import re as re2
            m = re2.match(r"^\s*(\d+)", text)
            if m:
                self._add_fact("passwd_line_count", m.group(1), intent, cmd_name, step, category="system",
                              confidence=0.7)

        # ── 命令名辅助的提取 (单命令 / CUSTOM) ──

        # hostname (裸命令)
        if lower.strip() == "hostname":
            val = text.splitlines()[0].strip() if text.strip() else ""
            if val and len(val) < 80:
                self._add_fact("hostname_cmd", val, intent, cmd_name, step,
                               category="system")

        # whoami
        if lower.strip() == "whoami":
            self._extract_whoami(text, intent, cmd_name, step)

        # id
        if lower.strip() == "id":
            self._extract_id(text, intent, cmd_name, step)

        # hostname (裸命令)
            val = text.splitlines()[0].strip() if text.strip() else ""
            if val and len(val) < 80:
                self._add_fact("hostname_cmd", val, intent, cmd_name, step,
                               category="system")

        # ip addr or ifconfig
        if any(k in lower for k in ("ip addr", "ifconfig", "ip a")):
            self._extract_network(text, intent, cmd_name, step)

        # /etc/hosts
        if "127.0.0.1" in text or "localhost" in text:
            self._extract_etchosts(text, intent, cmd_name, step)

        # P5.3: 用户自定义规则 (系统自改进)
        self._match_user_rules(output, intent, cmd_name, step)

        # P9.4: 通用回退提取 — 任何输出都可能包含有用信息 (始终尝试)
        self._extract_generic(text, intent, cmd_name, step, params)

    # ── 各提取规则 ──

    def _extract_hostname_file(self, text: str, intent: str, cmd: str, step: int):
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("#", ";", "//")):
                self._add_fact("hostname", line, intent, cmd, step, category="system")
                self._current_discovery = "hostname"
                break

    def _extract_uname(self, text: str, intent: str, cmd: str, step: int):
        parts = text.split()
        if len(parts) >= 2:
            self._add_fact("node_name", parts[1], intent, cmd, step, category="system")
        if len(parts) >= 3:
            self._add_fact("kernel", parts[2], intent, cmd, step, category="system")
        if len(parts) >= 4:
            self._add_fact("kernel_release", parts[3], intent, cmd, step, category="system")
        if len(parts) >= 2:
            self._add_fact("architecture", parts[-1], intent, cmd, step, category="system")
        self._current_discovery = "kernel"

    def _extract_os_release(self, text: str, intent: str, cmd: str, step: int):
        for line in text.splitlines():
            m = re.match(r'^(PRETTY_NAME|NAME|VERSION_ID|ID|VERSION_CODENAME)\s*=\s*"?([^"]*?)"?\s*$', line)
            if m:
                key = f"os_{m.group(1).lower()}"
                val = m.group(2).strip()
                if val:
                    self._add_fact(key, val, intent, cmd, step, category="system")
                    self._current_discovery = key

    def _extract_free(self, text: str, intent: str, cmd: str, step: int):
        lines = text.splitlines()
        for line in lines:
            parts = line.split()
            if line.startswith("Mem:") and len(parts) >= 3:
                self._add_fact("mem_total", parts[1], intent, cmd, step, category="system")
                if len(parts) >= 3:
                    self._add_fact("mem_avail", parts[-1], intent, cmd, step, category="system")
                self._current_discovery = "mem_total"
            elif line.startswith("Swap:") and len(parts) >= 2:
                self._add_fact("swap_total", parts[1], intent, cmd, step, category="system")

    def _extract_df(self, text: str, intent: str, cmd: str, step: int):
        for line in text.splitlines():
            if line.startswith("/"):
                parts = line.split()
                if len(parts) >= 6:
                    mount = parts[5] if len(parts) > 5 else parts[0]
                    mount_name = mount.replace("/", "_").strip("_") or "root"
                    self._add_fact(f"disk_{mount_name}", parts[3], intent, cmd, step, category="system")
                    self._current_discovery = "disk"

    def _extract_ls(self, text: str, intent: str, cmd: str, step: int, path: str = ""):
        dir_name = path.replace("/", "_").strip("_") or "root"
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("total", "drwx", "-rw", "-r-", "lrwx", "crw", "brw", "srw")):
                continue
            items.append(line.split()[-1] if " " in line else line)
        if items:
            key = f"dir_{dir_name}"
            val = ",".join(items[:5])
            self._add_fact(key, val, intent, cmd, step, confidence=0.6, category="explore")
            self._current_discovery = key

    def _extract_cpuinfo(self, text: str, intent: str, cmd: str, step: int):
        count = 0
        for line in text.splitlines():
            if line.strip().startswith("processor"):
                count += 1
        if count > 0:
            self._add_fact("cpu_cores", str(count), intent, cmd, step, category="system")
            self._current_discovery = "cpu_cores"
        # 提取 model name
        for line in text.splitlines():
            if "model name" in line:
                model = line.split(":")[-1].strip()
                if model:
                    self._add_fact("cpu_model", model, intent, cmd, step, category="system")
                    break

    def _extract_passwd(self, text: str, intent: str, cmd: str, step: int):
        users = set()
        for line in text.splitlines():
            parts = line.split(":")
            if len(parts) >= 1 and parts[0] and not parts[0].startswith("#"):
                users.add(parts[0].strip())
        if users:
            user_list = ",".join(sorted(users)[:8])
            self._add_fact("users", user_list, intent, cmd, step, category="system", confidence=0.7)

    def _extract_whoami(self, text: str, intent: str, cmd: str, step: int):
        """whoami 输出: 当前用户名"""
        val = text.splitlines()[0].strip() if text.strip() else ""
        if val and len(val) < 40:
            self._add_fact("current_user", val, intent, cmd, step, category="system")

    def _extract_id(self, text: str, intent: str, cmd: str, step: int):
        """id 输出: uid=0(root) gid=0(root)"""
        uid_match = re.search(r'uid=(\d+)\(([^)]+)\)', text)
        gid_match = re.search(r'gid=(\d+)\(([^)]+)\)', text)
        if uid_match:
            self._add_fact("uid_info", f"uid={uid_match.group(1)}({uid_match.group(2)})", intent, cmd, step, category="system")
        if gid_match:
            self._add_fact("gid_info", f"gid={gid_match.group(1)}({gid_match.group(2)})", intent, cmd, step, category="system")
            if uid_match and uid_match.group(2) == "root":
                self._add_fact("is_root", "yes", intent, cmd, step, category="system")

    def _extract_network(self, text: str, intent: str, cmd: str, step: int):
        for line in text.splitlines():
            line = line.strip()
            # IPv4
            m = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if m:
                ip = m.group(1)
                if not ip.startswith("127."):
                    self._add_fact("ip_addr", ip, intent, cmd, step, category="network")
                    break
        # MAC
        for line in text.splitlines():
            m = re.search(r'ether ([0-9a-f:]{17})', line.strip())
            if m:
                self._add_fact("mac_addr", m.group(1), intent, cmd, step, category="network")

    def _extract_etchosts(self, text: str, intent: str, cmd: str, step: int):
        hostnames = set()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                for p in parts[1:]:
                    if p and not p.startswith("#"):
                        hostnames.add(p)
        if hostnames:
            self._add_fact("etchosts_hosts", ",".join(sorted(hostnames)[:5]), intent, cmd, step, category="network")

    # ── P9.4: 通用回退提取 ──

    def _extract_generic(self, text: str, intent: str, cmd: str, step: int,
                         params: dict | None = None):
        """
        P9.4: 通用提取 — 从任意命令输出中提取 key: value 对
        """
        lines = text.strip().splitlines()
        if not lines:
            return

        added = 0
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            if line.startswith(("---", "==>", "<==", "(", "[", "*")):
                continue
            if added >= 3:
                break

            # 模式1: key: value  (冒号分隔)
            if ":" in line and not line.startswith("#"):
                parts = line.split(":", 1)
                key = parts[0].strip().lower().replace(" ", "_").replace("\t", "_").rstrip("_,.:;!?")
                val = parts[1].strip()
                if val and len(key) > 1 and len(key) < 30 and key.isascii() and not key.startswith("/"):
                    if self._is_valid_value(val):
                        fact_key = f"generic_{key}"
                        if fact_key not in self.facts:
                            self._add_fact(fact_key, val[:80], intent, cmd, step,
                                           confidence=0.5, category="general")
                            added += 1
                            continue

            # 模式2: =分隔的 key=value  (env, /etc/os-release 风格但未匹配的)
            if "=" in line and not line.startswith("#"):
                parts = line.split("=", 1)
                key = parts[0].strip().lower().replace(" ", "_").rstrip("_,.:;!?")
                val = parts[1].strip().strip('"\'')
                if val and len(key) > 1 and len(key) < 30 and key.isascii():
                    if self._is_valid_value(val):
                        fact_key = f"generic_{key}"
                        if fact_key not in self.facts:
                            self._add_fact(fact_key, val[:80], intent, cmd, step,
                                           confidence=0.5, category="general")
                            added += 1
                            continue

            # 模式3: 单行短输出 (可能是命令的直接返回值)
            if len(lines) == 1 and len(line) < 60 and len(line) >= 2:
                words = line.split()
                if len(words) <= 2:
                    cmd_based_key = cmd.replace(" ", "_").split("/")[-1].replace(" ", "_")
                    if cmd_based_key and "|" not in cmd_based_key and "\n" not in cmd_based_key:
                        fact_key = f"raw_{cmd_based_key[:20]}"
                        if fact_key not in self.facts and self._is_valid_value(line):
                            self._add_fact(fact_key, line[:60], intent, cmd, step,
                                           confidence=0.4, category="general")
                            added += 1
                            continue

        if added > 0:
            self._current_discovery = "generic"

    # ── 内部: 事实管理 ──

    @staticmethod
    def _is_valid_value(val: str) -> bool:
        """P6.1: 校验提取的值是否有效 (拒绝垃圾)"""
        if not val or len(val) < 1:
            return False
        # 只含标点符号的垃圾
        stripped = val.strip("=._-:;~!@#$%^&*()[]{}\"'")
        if not stripped:
            return False
        # 以 = 或 _ 开头 (解析伪影)
        if val.startswith(("=", "_", ".", ")", "(")):
            return False
        # 看起来像残留的键=值解析
        if "=" in val and val.startswith("="):
            return False
        # 少于2个字母数字字符
        letter_count = sum(c.isalnum() for c in val)
        if letter_count < 2:
            return False
        return True

    def _add_fact(self, key: str, value: str, source_intent: str,
                  source_cmd: str, step: int, confidence: float = 1.0,
                  category: str = "general"):
        """添加或更新事实 (置信度递增, P5.4: 追踪提取效用)

        双写: self.facts (legacy) + self.graph (P10 FactGraph)
        """
        if not value or len(value) > 100:
            value = value[:100]
        # P6.1: 拒绝垃圾值
        if not self._is_valid_value(value):
            return

        # P10: 写入图
        was_new = self.graph.add_node(key, value, category, confidence, step, source_cmd)

        # 兼容: 写入 facts dict
        is_new = key not in self.facts
        if key in self.facts:
            old = self.facts[key]
            old["value"] = value
            old["confidence"] = min(old["confidence"] + 0.15, 1.0)
            old["step"] = step
            old["count"] = old.get("count", 1) + 1
            old["category"] = category
        else:
            if len(self.facts) >= self.max_facts:
                oldest = min(self.facts, key=lambda k: self.facts[k]["step"])
                del self.facts[oldest]
            self.facts[key] = {
                "value": value,
                "source_intent": source_intent,
                "source_cmd": source_cmd[:50],
                "step": step,
                "confidence": confidence,
                "count": 1,
                "category": category,
            }
        # P5.4: 记录提取效用
        if self.meta and (is_new or was_new):
            if is_new:
                self.meta.register(f"extract_{key}", "extraction",
                                  {"key": key, "source": str(source_cmd)[:40]}, step)
                self.meta.record(f"extract_{key}", 1.0, step)
        # P9.6: 自动追踪历史值变化
        self._track_fact_history()

    # ── 查询接口 ──

    def get_current_discovery(self) -> Optional[str]:
        """返回最新发现的关键事实 key"""
        return self._current_discovery

    def get_fact(self, key: str) -> Optional[str]:
        """按 key 查询事实值"""
        return self.facts[key]["value"] if key in self.facts else None

    def get_facts_by_category(self, category: str) -> list[str]:
        """按类别查询所有事实 key (图优先 + facts 兼容)"""
        graph_keys = self.graph.get_nodes_by_category(category)
        if graph_keys:
            return graph_keys
        return [k for k, v in self.facts.items() if v.get("category") == category]

    # ── P9.6: 跨事实推理引擎 ──

    def _track_fact_history(self):
        """
        P9.6: 追踪事实历史值, 用于变化检测
        P10: 委托给 FactGraph
        """
        self.graph.track_fact_history()

    def _build_change_report(self) -> str:
        """
        P9.6: 检测事实变化, 生成变化报告
        P10: 委托给 FactGraph
        """
        return self.graph.build_change_report()

    def _build_cross_analysis(self) -> str:
        """
        P9.6: 跨事实推理 — 从多事实联合推断深层信息
        P10: 委托给 FactGraph
        """
        return self.graph.build_cross_analysis()

    def _build_fact_analysis(self, keys: list[str]) -> str:
        """
        P9.5: 从一组事实 key 生成自然语言分析段落
        P10: 兼容 graph (可能比 facts dict 多)
        """
        def v(k):
            if k in self.facts:
                return self.facts[k]["value"]
            n = self.graph.nodes.get(k)
            return n.value if n else None

        parts = []

        # 1. 操作系统摘要
        os_name = v("os_name") or v("generic_os_name")
        os_ver = v("os_version_id")
        os_codename = v("os_version_codename") or v("generic_os_version_codename")
        arch = v("architecture") or v("generic_architecture")
        kernel = v("kernel") or v("generic_kernel")

        os_parts = []
        if os_name:
            os_parts.append(os_name)
        if os_ver:
            os_parts.append(os_ver)
        if os_codename:
            os_parts.append(f"({os_codename})")
        if os_parts:
            s = "System: " + " ".join(os_parts)
            if kernel:
                s += f", kernel {kernel}"
            if arch:
                s += f", arch {arch}"
            parts.append(s)

        # 2. CPU 摘要
        cpu_cores = v("cpu_cores")
        cpu_model = v("cpu_model") or v("generic_cpu_model") or v("generic_model_name")
        if cpu_cores or cpu_model:
            s = "CPU: "
            if cpu_cores:
                s += f"{cpu_cores} cores"
                if cpu_model:
                    s += f" x {cpu_model}"
            elif cpu_model:
                s += cpu_model
            parts.append(s)

        # 3. 内存摘要
        mem = v("mem_total") or v("generic_mem")
        swap = v("swap_total") or v("generic_swap")
        if mem or swap:
            s = "Memory: "
            mem_items = []
            if mem:
                mem_items.append(f"RAM {mem}")
            if swap:
                mem_items.append(f"Swap {swap}")
            s += ", ".join(mem_items)
            parts.append(s)

        # 4. 磁盘摘要
        disk_keys = [k for k in self.facts if k.startswith("disk_") or k == "generic_size"]
        if disk_keys:
            s = "Disk: " + ", ".join(
                f"{k.replace('disk_','')}={self.facts[k]['value']}" for k in disk_keys[:3]
            )
            parts.append(s)

        # 5. 网络/身份
        hostname = v("hostname") or v("node_name") or v("generic_node_name")
        if hostname:
            parts.append(f"Host: {hostname}")

        users = v("users") or v("generic_uid")
        if users:
            parts.append(f"Users: {users}")

        return "\n".join(parts) if parts else "(no analysis available)"

    # ── P9.6: GENERATE 意图 — 从事实生成新内容 ──

    def build_generate_content(self) -> dict:
        """
        P9.6: GENERATE 意图 — 产生比 WRITE 更丰富的内容
        返回 {"content": str, "path": str, "desc": str, "size": int}
        """
        import json, random
        self._track_fact_history()
        # P10: 优先从 graph 获取
        system_keys = self.graph.get_nodes_by_category("system") or self.get_facts_by_category("system")
        analysis = self._build_fact_analysis(system_keys)
        cross = self._build_cross_analysis()
        changes = self._build_change_report()

        styles = ["profile", "experiment", "discovery_log"]
        style = styles[getattr(self, "_gen_style_idx", 0) % len(styles)]
        if not hasattr(self, "_gen_style_idx"):
            self._gen_style_idx = 0
        self._gen_style_idx += 1

        if style == "profile":
            # 系统画像: 全面文档
            lines = [
                f"# Folunar System Profile",
                f"# Generated at step {self._step_counter}",
                f"# Facts collected: {len(self.facts)}",
                "",
                "## Overview",
                analysis,
                "",
            ]
            if cross:
                lines.append(cross)
                lines.append("")
            if changes:
                lines.append(changes)
                lines.append("")
            lines.append("## Fact Inventory")
            for k, v in sorted(self.facts.items(), key=lambda x: x[1].get("category", "z")):
                val = v["value"][:50]
                cat = v.get("category", "?")
                lines.append(f"- [{cat}] {k}: {val}")
            lines.append("")
            lines.append("---")
            lines.append(f"_Auto-generated by Folunar at step {self._step_counter}_")
            content = "\n".join(lines)
            path = f"/tmp/profile_{self._step_counter}.md"
            desc = "系统画像"

        elif style == "experiment":
            # 实验脚本: 探测未知领域
            # 找还没有提取的事实类别
            cat_set = set(v.get("category", "") for v in self.facts.values())
            script_lines = [
                "#!/bin/bash",
                "# Folunar Experiment Script",
                f"# Generated at step {self._step_counter}",
                f"# Testing unverified hypotheses",
                "",
                "echo '=== Folunar Experiment ==='",
                f"echo 'Step: {self._step_counter}'",
                f"echo 'Known facts: {len(self.facts)}'",
                "",
            ]
            # 假设1: 检查系统时间
            script_lines.append("# Hypothesis 1: System clock is synchronized")
            script_lines.append('echo "[TEST] System clock:"')
            script_lines.append('date -u')
            script_lines.append('cat /proc/uptime 2>/dev/null | awk "{print \"Uptime: \" $$1 \" seconds\"}"')
            script_lines.append("")
            # 假设2: 检查进程
            script_lines.append("# Hypothesis 2: Init process is PID 1")
            script_lines.append('ps -p 1 -o comm= 2>/dev/null && echo "[PASS] PID 1 exists" || echo "[FAIL] No PID 1"')
            script_lines.append("")
            # 假设3: 如果缺 network 类别, 尝试探测网络
            if "network" not in cat_set:
                script_lines.append("# Hypothesis 3: Network is available (unverified)")
                script_lines.append('ping -c 1 127.0.0.1 2>&1 | head -2')
                script_lines.append('cat /etc/hosts 2>/dev/null | head -5')
                script_lines.append("")
            # 假设4: 如果缺某些系统参数
            if "mem_total" not in self.facts:
                script_lines.append("# Hypothesis 4: Memory info is accessible")
                script_lines.append('cat /proc/meminfo 2>/dev/null | head -5 || echo "[INFO] /proc/meminfo not accessible"')
                script_lines.append("")
            script_lines.append("echo '=== Experiment Complete ==='")
            script_lines.append("exit 0")
            content = "\n".join(script_lines)
            path = f"/tmp/experiment_{self._step_counter}.sh"
            desc = "实验脚本"

        else:
            # discovery_log: 生成 JSON 格式的发现日志
            cats = {}
            for k, v in self.facts.items():
                cat = v.get("category", "unknown")
                cats.setdefault(cat, []).append({"key": k, "value": v["value"][:60]})
            log = {
                "_meta": {"step": self._step_counter, "elapsed": self._step_counter * 0.3},
                "analysis": analysis,
                "inference": cross,
                "changes": changes,
                "categories": {c: items for c, items in cats.items()},
            }
            content = json.dumps(log, ensure_ascii=False, indent=2)
            path = f"/tmp/discovery_{self._step_counter}.json"
            desc = "发现日志"

        return {"content": content, "path": path, "desc": desc, "size": len(content)}

    def build_write_content(self) -> dict:
        """
        P3/P9.5: 从工作栏事实生成有价值的写入内容
        返回 {"content": str, "path": str, "desc": str}
        """
        import json, random
        system_facts = self.get_facts_by_category("system")
        if len(system_facts) < 3:
            lines = [f"# Folunar Snapshot (step {self._step_counter})", ""]
            for k, v in list(self.facts.items())[:5]:
                lines.append(f"{k}={v['value'][:40]}")
            return {"content": "\n".join(lines), "path": "/tmp/folunar_summary.txt", "desc": "简单摘要", "size": len("\n".join(lines))}

        styles = ["json", "report", "script"]
        style = styles[getattr(self, "_write_style_idx", 0) % len(styles)]
        if not hasattr(self, "_write_style_idx"):
            self._write_style_idx = 0
        self._write_style_idx += 1

        selected = random.sample(system_facts, min(6, len(system_facts)))
        # P10: 兼容 graph 和 facts dict (graph 可能有 facts 没有的 key)
        records = {}
        for k in selected:
            if k in self.facts:
                records[k] = self.facts[k]["value"][:60]
            elif k in self.graph.nodes:
                records[k] = self.graph.nodes[k].value[:60]
        analysis = self._build_fact_analysis(selected)
        # P9.6: 跨事实推理 + 变化检测
        self._track_fact_history()
        cross_analysis = self._build_cross_analysis()
        change_report = self._build_change_report()

        if style == "json":
            structured = {
                "_meta": {"step": self._step_counter, "fact_count": len(self.facts)},
                "_summary": analysis,
                "_inference": cross_analysis,
                "_changes": change_report,
                "facts": records,
            }
            content = json.dumps(structured, ensure_ascii=False, indent=2)
            path = f"/tmp/facts_{self._step_counter}.json"
            desc = "JSON事实+分析"
        elif style == "report":
            lines = [
                "# Folunar System Report",
                f"# Generated at step {self._step_counter}",
                f"# Total facts: {len(self.facts)}",
                "",
                "## System Analysis",
                analysis,
                "",
            ]
            if cross_analysis:
                lines.append(cross_analysis)
                lines.append("")
            if change_report:
                lines.append(change_report)
                lines.append("")
            lines.append("## Raw Facts")
            for k, v in records.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
            lines.append("---")
            lines.append(f"_Report auto-generated by Folunar agent at step {self._step_counter}_")
            content = "\n".join(lines)
            path = "/tmp/report.md"
            desc = "Markdown分析报告"
        else:
            # script: 真实验证脚本 (P9.5: 含对比+差异报告+退出码)
            script_lines = [
                "#!/bin/bash",
                "# Folunar Verification Script",
                f"# Generated at step {self._step_counter}",
                f"# Facts: {len(self.facts)}",
                "",
                "FAILED=0",
                "REPORT=",
            ]
            for k, v in records.items():
                expected = v[:30].replace("'", "'\\''")
                if k.startswith("disk_"):
                    script_lines.append(f'  if ! df -h 2>/dev/null | grep -q "{v[:10]}"; then')
                    script_lines.append(f'    echo "[WARN] {k}: expected size {expected} not found"')
                    script_lines.append('    FAILED=$((FAILED+1))')
                    script_lines.append('  else')
                    script_lines.append(f'    echo "[OK] {k}: disk {expected} mounted"')
                    script_lines.append('  fi')
                elif k.startswith("os_"):
                    kn = k.replace("os_", "")
                    script_lines.append(f'  actual=$(grep -i {kn}= /etc/os-release 2>/dev/null | cut -d= -f2 | xargs)')
                    script_lines.append(f'  if [[ "$actual" = "{expected}" ]]; then')
                    script_lines.append(f'    echo "[OK] {k}: {expected}"')
                    script_lines.append('  else')
                    script_lines.append(f'    echo "[WARN] {k}: expected={expected} actual=$actual"')
                    script_lines.append('    FAILED=$((FAILED+1))')
                    script_lines.append('  fi')
                elif k == "kernel":
                    script_lines.append(f'  actual=$(uname -r 2>/dev/null)')
                    script_lines.append(f'  if echo "$actual" | grep -q "{v[:10]}"; then')
                    script_lines.append(f'    echo "[OK] {k}: $actual"')
                    script_lines.append('  else')
                    script_lines.append(f'    echo "[WARN] {k}: expected={expected} actual=$actual"')
                    script_lines.append('    FAILED=$((FAILED+1))')
                    script_lines.append('  fi')
                else:
                    script_lines.append(f'  echo "[INFO] {k}={expected}"')
                script_lines.append("")
            script_lines.append("# Summary")
            script_lines.append('if [[ $FAILED -eq 0 ]]; then')
            script_lines.append('  echo "[PASS] All checks passed"')
            script_lines.append('else')
            script_lines.append('  echo "[FAIL] $FAILED check(s) failed"')
            script_lines.append('fi')
            script_lines.append("exit $FAILED")
            content = "\n".join(script_lines)
            path = f"/tmp/check_{self._step_counter}.sh"
            desc = "检测脚本(含对比)"

        return {"content": content, "path": path, "desc": desc}

    def generate_self_goal(self) -> Optional[tuple[str, dict]]:
        """
        P9.1: 基于事实缺口自生成目标
        P10: 优先用图缺口驱动, 再创作, 再补全
        """
        # P10: 如果有图缺口, 用它
        gaps = self.graph.find_gaps()
        if gaps:
            src, missing, rel = gaps[0]
            gap_map = {
                "os_version_id": ("READ", {"path": "/etc/os-release"}),
                "cpu_model": ("READ", {"path": "/proc/cpuinfo"}),
                "mem_total": ("CUSTOM", {"custom_args": ["free", "-h"], "cluster": "SYSTEM"}),
                "swap_total": ("CUSTOM", {"custom_args": ["free", "-h"], "cluster": "SYSTEM"}),
                "etchosts_hosts": ("READ", {"path": "/etc/hosts"}),
                "hostname_cmd": ("CUSTOM", {"custom_args": ["hostname"], "cluster": "SYSTEM"}),
                "current_user": ("CUSTOM", {"custom_args": ["whoami"], "cluster": "USER"}),
                "ip_addr": ("CUSTOM", {"custom_args": ["ip", "addr"], "cluster": "NETWORK"}),
            }
            if missing in gap_map:
                return gap_map[missing]

        # 1. P9.4: 优先用内容生成器产出丰富内容
            ci = self.build_write_content()
            content = ci["content"]
            path = ci["path"]
            # 用 python3 base64 安全写入
            import base64 as _b64
            encoded = _b64.b64encode(content.encode()).decode()
            shell_cmd = (
                f"python3 -c \"import base64; "
                f"data=base64.b64decode('{encoded}'); "
                f"f=open('{path}','wb'); "
                f"f.write(data); f.close(); "
                f"print(f'written {{len(data)}} bytes to {path}')\""
            )
            return ("CUSTOM", {
                "custom_args": ["sh", "-c", shell_cmd],
                "cluster": "CREATIVE"
            })

        # 2. 类别补全: 有 system 事实但缺 network/explore?
        categories = set(v.get("category", "") for v in self.facts.values())
        cat_goals = {
            "network": ("INFO", {"target": "ip_addr"}),
            "explore": ("EXPLORE", {"path": "/tmp"}),
            "general": ("EXPLORE", {"path": "/etc"}),
        }
        for cat, goal in cat_goals.items():
            if cat not in categories:
                return goal

        # 3. 事实深度: 有 os_name 但没 os_version?
        paired = [("os_name", "os_version_id"), ("kernel", "kernel_modules"),
                  ("hostname", "etchosts_hosts"), ("cpu_cores", "cpu_model"),
                  ("mem_total", "swap_total"), ("current_user", "uid_info")]
        for have, want in paired:
            if have in self.facts and want not in self.facts:
                return ("CUSTOM", {"custom_args": ["cat", "/proc/meminfo"],
                                   "cluster": "SYSTEM"})

        # 4. 如果有 follow-up, 用它
        fu = self._compute_follow_up()
        if fu:
            return fu

        return None

    def get_follow_up(self) -> Optional[tuple[str, dict]]:
        """推荐下一步, 并存储到 last_follow_up"""
        result = self._compute_follow_up()
        self.last_follow_up = result
        return result

    def _compute_follow_up(self) -> Optional[tuple[str, dict]]:
        """实际的推荐逻辑 (P10: 先查图缺口, 回退配置链, 再硬编码)"""

        # P10: FactGraph 缺口驱动
        gaps = self.graph.find_gaps()
        if gaps:
            src, missing, rel = gaps[0]
            # 将缺口映射到意图+参数
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
            if missing in gap_map:
                return gap_map[missing]

        # P5.3: 配置驱动的链
        for chain in self.rules.get("follow_up_chains", []):
            trigger = chain.get("trigger_key", "")
            needs = chain.get("needs_key", "")
            if trigger in self.facts and needs not in self.facts:
                intent = chain.get("intent", "CUSTOM")
                if intent == "CUSTOM":
                    args = chain.get("args", ["ls"])
                    return ("CUSTOM", {"custom_args": args, "cluster": chain.get("cluster", "SYSTEM")})
                elif intent == "READ":
                    path = chain.get("path", "/etc/hostname")
                    return ("READ", {"path": path})

        # ── 验证链: 发现X → 验证X → 扩展X ──
        if "hostname" in self.facts and "hostname_cmd" not in self.facts:
            return ("CUSTOM", {"custom_args": ["hostname"], "cluster": "SYSTEM"})
        if "hostname" in self.facts and "hostname_cmd" in self.facts and "etchosts_hosts" not in self.facts:
            return ("READ", {"path": "/etc/hosts"})

        # ── 系统探查链 ──
        if "cpu_cores" in self.facts and "mem_total" not in self.facts:
            return ("CUSTOM", {"custom_args": ["free", "-h"], "cluster": "SYSTEM"})
        if "mem_total" in self.facts and "disk_root" not in self.facts:
            return ("CUSTOM", {"custom_args": ["df", "-h"], "cluster": "SYSTEM"})

        # ── 身份链 ──
        if "users" in self.facts and "current_user" not in self.facts:
            return ("CUSTOM", {"custom_args": ["whoami"], "cluster": "USER"})
        if "current_user" in self.facts and "uid_info" not in self.facts:
            return ("CUSTOM", {"custom_args": ["id"], "cluster": "USER"})

        # ── 内核链 ──
        if "kernel" in self.facts and "os_pretty_name" not in self.facts:
            return ("READ", {"path": "/etc/os-release"})
        if "os_pretty_name" in self.facts and "kernel_modules" not in self.facts:
            return ("CUSTOM", {"custom_args": ["lsmod"], "cluster": "SYSTEM"})

        # ── 默认: 从目录读文件 ──
        dir_facts = [k for k in self.facts if k.startswith("dir_")]
        for dk in sorted(dir_facts, key=lambda k: -self.facts[k]["confidence"]):
            if dk not in ("dir_tmp", "dir_root"):
                val = self.facts[dk]["value"]
                first_file = val.split(",")[0].strip()
                if first_file and first_file not in (".", ".."):
                    path = f"/etc/{first_file}" if dk == "dir_etc" else first_file
                    return ("CUSTOM", {"custom_args": ["head", "-5", path], "cluster": "FILE_READ"})

        return None

    def _build_dynamic_probes(self, explored_paths: set[str]) -> list[dict]:
        """
        P8.4: 动态构建探针候选 — 从工作栏事实推导, 替代固定 priority 列表

        三层:
          1. 目标驱动: 从 follow-up 链推导应读的文件
          2. 类别补全: 已有哪些类别, 缺什么关键文件
          3. 元学习历史高分探针
        """
        now = self._step_counter
        candidates = []

        # ── 已知文件集合 ──
        known_paths = set()
        for k, v in self.facts.items():
            src = v.get("source_cmd", "")
            if "/" in src:
                parts = src.split()
                for p in parts:
                    if p.startswith("/") and len(p) > 4:
                        known_paths.add(p)
            # 从事实值里的路径猜测
            val = v.get("value", "")
            if "/" in val and len(val) < 60:
                for word in val.split():
                    if word.startswith("/") and len(word) > 4:
                        known_paths.add(word)

        # ── 1. 目标驱动探针: follow-up 翻译成文件读命令 ──
        for intent, params in [self._compute_follow_up()] if self._compute_follow_up() else []:
            if intent == "READ":
                path = params.get("path", "")
                if path and path not in known_paths:
                    candidates.append({
                        "cmd": ["cat", path],
                        "cluster": "FILE_READ",
                        "path_key": f"cat {path}",
                        "base_score": 0.9,
                    })
            elif intent == "CUSTOM":
                args = params.get("custom_args", [])
                if args:
                    path_key = " ".join(str(a) for a in args)
                    candidates.append({
                        "cmd": args,
                        "cluster": params.get("cluster", "SYSTEM"),
                        "path_key": path_key,
                        "base_score": 0.8,
                    })

        # ── 2. 类别补全探针: 缺少什么关键系统文件 ──
        key_files = {
            "/etc/hostname": "hostname",
            "/etc/hosts": "etchosts_hosts",
            "/etc/os-release": "os_pretty_name",
            "/proc/version": "kernel",
            "/proc/cpuinfo": "cpu_cores",
            "/proc/meminfo": "mem_total",
            "/proc/loadavg": "loadavg",
            "/proc/uptime": "uptime",
            "/etc/resolv.conf": "dns_config",
            "/etc/passwd": "users",
            "/etc/group": "groups",
            "/proc/modules": "kernel_modules",
            "/proc/1/status": "init_process",
            "/etc/fstab": "fstab",
            "/etc/timezone": "timezone",
            "/proc/partitions": "partitions",
        }
        for path, fact_key in key_files.items():
            if fact_key not in self.facts and path not in known_paths:
                # 先检查是否已读
                if path not in explored_paths:
                    decay = sum(1 for k in key_files if k != path and k not in explored_paths)
                    base = max(0.3, 1.0 - decay * 0.1)
                    candidates.append({
                        "cmd": ["cat", path],
                        "cluster": "PROCFS" if "/proc/" in path else "SYSTEM",
                        "path_key": f"cat {path}",
                        "base_score": base,
                    })

        # ── 3. 元学习历史高分探针 ──
        if self.meta:
            for b in self.meta.get_best("probe_path", n=15, min_trials=2):
                path = b.get("params", {}).get("path", "")
                if not path:
                    continue
                # 检查 path 是否已过度探索
                if path in known_paths and path in explored_paths:
                    continue
                utility = b.get("utility", 0.0)
                if utility > -0.1:  # 非负效用才复用
                    candidates.append({
                        "cmd": ["cat", path] if "/" in path else path.split(),
                        "cluster": "PROCFS" if "/proc/" in path else "FILE_READ",
                        "path_key": path,
                        "base_score": 0.3 + utility * 2.0,
                    })

        # 去重 (按 path_key)
        seen = set()
        unique = []
        for c in candidates:
            if c["path_key"] not in seen:
                seen.add(c["path_key"])
                unique.append(c)

        return unique

    def get_curiosity_probe(self, explored_paths: set[str] | None = None) -> Optional[tuple[str, dict]]:
        """
        P8.4: 动态探针 — 从工作栏事实推导候选, 替代固定 priority 列表

        流程:
          1. 构建动态候选探针 (目标驱动 + 类别补全 + 元学习高分)
          2. 从 meta 读取历史效用分
          3. 按 基础分 + 效用×2 - 已探索惩罚 - 近期使用惩罚 排序
          4. 跳过总分 < -0.5 的探针
          5. 回退到 fallback_commands (确保探索不枯竭)
        """
        if explored_paths is None:
            return None

        now = self._step_counter
        candidates = []

        # 1. P8.4: 动态构建候选探针
        candidates = self._build_dynamic_probes(explored_paths)

        # 2. 回退安全网 (动态候选不足时补充)
        probe_cfg = self.rules.get("curiosity_probes", {})
        fallbacks = probe_cfg.get("fallback_commands", [])
        if len(candidates) < 3 and fallbacks:
            for fb in fallbacks:
                cmd = list(fb)
                path_key = " ".join(cmd)
                if path_key not in set(c["path_key"] for c in candidates):
                    candidates.append({
                        "cmd": cmd,
                        "cluster": "FILE_FIND" if "find" in path_key else "FILE_LIST",
                        "path_key": path_key,
                        "base_score": 0.25,
                    })

        # 3. 从元学习器取历史效用 (已使用过的探针)
        meta_scores = {}
        if self.meta:
            for b in self.meta.get_best("probe_path", n=30, min_trials=2):
                path = b.get("params", {}).get("path", "")
                meta_scores[path] = b.get("utility", 0.0)

        # 4. 打分并选择
        scored = []
        for c in candidates:
            path_key = c["path_key"]
            utility = meta_scores.get(path_key, 0.0)
            score = c["base_score"] + utility * 2.0

            # 已探索过的路径略降
            if path_key in explored_paths:
                score -= 0.25

            # 近期使用惩罚
            if self.meta:
                probe_id = f"probe_{path_key[:40].replace(' ', '_')}"
                last_step = self.meta.data.get(probe_id, {}).get("last_step", 0)
                steps_ago = now - last_step
                if steps_ago < 5:
                    score -= 0.6
                elif steps_ago < 15:
                    score -= 0.2

            scored.append((score, c))

        scored.sort(key=lambda x: -x[0])

        for score, c in scored:
            if score < -0.5:
                continue

            # 首次出现则注册到 meta
            if self.meta:
                probe_id = f"probe_{c['path_key'][:40].replace(' ', '_')}"
                self.meta.register(probe_id, "probe_path",
                                   {"path": c["path_key"]}, now)

            return ("CUSTOM", {
                "custom_args": c["cmd"],
                "cluster": c["cluster"],
            })

        return None

    def check_chain_completed(self, executed_intent: str, executed_params: dict) -> bool:
        """检查链步骤完成 (chain_step满3后自动重置)"""
        if self.last_follow_up is None or self.chain_step >= 3:
            return False
        fu_intent, fu_params = self.last_follow_up
        if executed_intent != fu_intent:
            return False
        if fu_intent == "CUSTOM":
            if executed_params.get("custom_args", []) != fu_params.get("custom_args", []):
                return False
        elif fu_intent == "READ":
            if executed_params.get("path", "") != fu_params.get("path", ""):
                return False
        self.chain_step += 1
        self.chain_completed_at = self._step_counter
        if self.chain_step >= 3:
            self.last_follow_up = None

    def has_active_goal(self) -> bool:
        """工作栏有可执行的目标? (链进行中 或 未开始但有缓存推荐)"""
        if self.chain_step > 0 and self.chain_step < 3 and self.last_follow_up:
            return True
        if self.chain_step == 0 and self.last_follow_up is not None:
            return True
        return False

    def get_current_goal(self) -> Optional[tuple[str, dict]]:
        """返回当前目标"""
        if self.chain_step > 0 and self.chain_step < 3 and self.last_follow_up:
            return self.last_follow_up
        return self.get_follow_up()  # 链满, 重置

    def get_chain_bonus(self) -> float:
        """链奖励递减: 第1步+1.5, 第2步+1.0, 第3步+0.5"""
        if self.chain_completed_at == self._step_counter:
            if self.chain_step == 1:
                return 1.5
            elif self.chain_step == 2:
                return 1.0
            elif self.chain_step >= 3:
                return 0.5
        return 0.0

    def reset_chain(self):
        """重置链状态"""
        self.chain_step = 0
        self.chain_completed_at = 0
        self.last_follow_up = None

    # ── P5.3: 可配置规则管理 ──

    def _load_rules(self) -> dict:
        """从配置文件加载规则, 不存在时返回默认"""
        import os, json
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "workbench_rules.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_rules(self):
        """保存规则到配置文件"""
        import os, json
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config", "workbench_rules.json")
        try:
            with open(path, "w") as f:
                json.dump(self.rules, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def add_user_rule(self, trigger_type: str, trigger_pattern: str,
                       key: str, category: str = "system"):
        """追加用户自定义的提取规则 (系统自改进用)"""
        if "user_rules" not in self.rules:
            self.rules["user_rules"] = []
        rule = {
            "trigger": trigger_type,
            "pattern": trigger_pattern,
            "key": key,
            "category": category,
        }
        # 去重
        for existing in self.rules["user_rules"]:
            if existing.get("key") == key:
                return
        self.rules["user_rules"].append(rule)
        self._save_rules()

    def _match_user_rules(self, output: str, intent: str, cmd_name: str, step: int):
        """尝试用户自定义规则匹配输出 (P5.4: 追踪效用)"""
        for rule in self.rules.get("user_rules", []):
            trigger = rule.get("trigger", "")
            pattern = rule.get("pattern", "")
            key = rule.get("key", "")
            cat = rule.get("category", "general")
            if not pattern or not key:
                continue
            rule_id = f"rule_{key}"
            if self.meta:
                self.meta.register(rule_id, "extraction_rule",
                                  {"key": key, "pattern": pattern, "trigger": trigger}, step)
            if trigger == "output_contains" and pattern in output:
                idx = output.find(pattern) + len(pattern)
                after = output[idx:idx+60].strip()
                val = after.split()[0] if after else ""
                # P6.1: 清理提取值中的垃圾 (去掉前导 = 和引号)
                val = val.lstrip("=:_-").strip('\\"\"')
                if val and len(val) < 40:
                    self._add_fact(key, val, intent, cmd_name, step, confidence=0.5, category=cat)
                    if self.meta:
                        self.meta.record(rule_id, 0.5, step)
                else:
                    if self.meta:
                        self.meta.record(rule_id, -0.2, step)
            elif trigger == "cmd_starts" and cmd_name.startswith(pattern):
                val = output.splitlines()[0].strip()[:40] if output.strip() else ""
                if val:
                    self._add_fact(key, val, intent, cmd_name, step, confidence=0.5, category=cat)

    # ── P5.3c: 脚本创作 ──

    def generate_script(self, script_name: str = "") -> Optional[str]:
        """
        P6.2: 基于工作栏事实生成 shell 脚本 (meta 偏置组合选择)
        返回 (脚本内容, combo_key) 或 None
        """
        if len(self.facts) < 3:
            return None
        import random as _rnd
        
        # 收集可用事实
        system_facts = self.get_facts_by_category("system")
        if not system_facts:
            return None
        
        # P6.2: 从 meta 读取历史最佳组合
        best_combo = None
        if self.meta:
            best_scripts = self.meta.get_best("script_output", n=5, min_trials=1)
            for bs in best_scripts:
                saved_combo = bs.get("params", {}).get("combo", "")
                if saved_combo:
                    combo_keys = [k.strip() for k in saved_combo.split(",")]
                    if all(k in system_facts for k in combo_keys):
                        best_combo = combo_keys
                        break
        
        if best_combo and _rnd.random() < 0.4:
            # 40% 概率复用历史最佳组合
            selected = best_combo
        else:
            # 随机选 2-4 个事实
            n_facts = min(len(system_facts), 2 + _rnd.randint(0, 2))
            selected = _rnd.sample(system_facts, n_facts)
        
        self._last_script_combo = ",".join(sorted(selected))
        
        lines = ["#!/bin/bash", "", "# Auto-generated by Folunar Workbench", f"# Facts: {', '.join(selected)}", ""]
        
        for key in selected:
            fact = self.facts.get(key)
            if not fact:
                continue
            val = fact["value"][:40]
            cmd = fact.get("source_cmd", "") or ""
            
            # P8.3: 通用模板 — 所有事实统一处理, 消除硬编码分支
            lines.append(f"echo '=== {key} ==='")
            lines.append(f"echo '{key}: {val}'")
            
            # 从 _find_command_for_key 获取推荐命令, 或用 source_cmd
            script_cmd = self._find_command_for_key(key)
            if not script_cmd:
                if cmd and cmd.startswith("["):
                    import ast
                    try:
                        cmd_list = ast.literal_eval(cmd)
                        if isinstance(cmd_list, list):
                            script_cmd = " ".join(cmd_list)
                    except:
                        script_cmd = cmd
                else:
                    script_cmd = cmd or ":"  # noop fallback
            lines.append(script_cmd)
            lines.append("")
        
        # 总结: 所有选定事实
        lines.append("echo '=== Summary ==='")
        for k in selected:
            fact = self.facts.get(k)
            if fact:
                lines.append(f"echo '{k}={fact["value"]}'")
        lines.append("")
        
        return ("\n".join(lines), self._last_script_combo)

    def get_last_script_combo(self) -> str:
        """P6.2: 返回最近一次脚本的事实组合 key"""
        return self._last_script_combo

    def _find_command_for_key(self, key: str) -> str:
        """
        P8.4c: 从事实推断可执行的 shell 命令
        优先用 source_cmd, 后备用 key 模式匹配
        """
        # 1. 如果有这个事实, 从 source_cmd 重建
        fact = self.facts.get(key)
        if fact:
            src = fact.get("source_cmd", "") or ""
            if src and src.startswith("["):
                import ast
                try:
                    cmd_list = ast.literal_eval(src)
                    if isinstance(cmd_list, list) and cmd_list:
                        return " ".join(str(c) for c in cmd_list)
                except:
                    pass
            elif src and src != key:
                return src

        # 2. 按 key 的模式匹配
        patterns: list[tuple[str, str]] = [
            ("kernel", "uname -a"),
            ("node_name", "uname -a"),
            ("architecture", "uname -a"),
            ("hostname_cmd", "hostname"),
            ("hostname", "cat /etc/hostname"),
            ("current_user", "whoami"),
            ("uid_info", "id"),
            ("cpu_", "cat /proc/cpuinfo"),
            ("mem_", "free -h"),
            ("swap_", "free -h"),
            ("disk_", "df -h"),
            ("ip_", "ip addr"),
            ("users", "cat /etc/passwd | cut -d: -f1"),
            ("etchosts", "cat /etc/hosts"),
            ("os_", "cat /etc/os-release"),
            ("loadavg", "cat /proc/loadavg"),
            ("uptime", "uptime"),
            ("partitions", "cat /proc/partitions"),
            ("timezone", "cat /etc/timezone"),
            ("groups", "cat /etc/group | head -20"),
            ("fstab", "cat /etc/fstab"),
            ("dns_", "cat /etc/resolv.conf"),
            ("init_", "cat /proc/1/status"),
            ("kernel_modules", "lsmod"),
        ]
        for prefix, cmd in patterns:
            if key.startswith(prefix) or key.endswith(prefix):
                return cmd

        # 3. 用 key 本身的来源作为猜测
        return ""

    # ── 状态摘要 ──

    def get_state_summary(self, max_keys: int = 4) -> str:
        """生成简短文本摘要 → 注入 StateEncoder.fact_summary (P10: 图优先)"""
        graph_summary = self.graph.get_state_summary(max_keys)
        if graph_summary != "无":
            return graph_summary
        if not self.facts:
            return "无"
        ranked = sorted(
            self.facts.items(),
            key=lambda x: (
                x[1]["confidence"] * (1.0 if (x[1]["step"] or 0) == self._step_counter else 0.5),
                -(x[1]["step"] or 0)
            ),
            reverse=True,
        )
        parts = []
        for key, fact in ranked[:max_keys]:
            val = fact["value"][:35]
            parts.append(f"{key}={val}")
        return " | ".join(parts)

    def save(self, path: str = ""):
        """持久化工作栏状态到文件 (含事实 + 图 + 链状态)"""
        if not path:
            import os
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "data", "persistent", "workbench_snapshot.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = {
                "facts": {k: {sk: sv for sk, sv in v.items()}
                         for k, v in self.facts.items()},
                "graph": self.graph.to_dict(),
                "chain_step": self.chain_step,
                "chain_completed_at": self.chain_completed_at,
                "_current_discovery": self._current_discovery,
            }
            import json
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"  \u26a0\ufe0f 工作栏保存失败: {e}")

    def load(self, path: str = ""):
        """从文件恢复工作栏状态 (含图 + facts)"""
        if not path:
            import os
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "data", "persistent", "workbench_snapshot.json")
        try:
            import json
            with open(path) as f:
                data = json.load(f)
            self.facts.clear()
            for k, v in data.get("facts", {}).items():
                self.facts[k] = v
            # P10: 恢复图
            graph_data = data.get("graph")
            if graph_data:
                self.graph = FactGraph.from_dict(graph_data)
            self.chain_step = data.get("chain_step", 0)
            self.chain_completed_at = data.get("chain_completed_at", 0)
            self._current_discovery = data.get("_current_discovery")
            return len(self.facts)
        except Exception:
            return 0

    def reset(self):
        """清空工作栏 (新会话用)"""
        self.facts.clear()
        self._current_discovery = None

    def stats(self) -> dict:
        graph_st = self.graph.stats()
        return {
            "n_facts": len(self.facts),
            "n_facts_graph": graph_st["n_nodes"],
            "n_edges": graph_st["n_edges"],
            "n_gaps": graph_st["n_gaps"],
            "schema_coverage": graph_st["schema_coverage"],
            "categories": {c: len(self.get_facts_by_category(c))
                          for c in ("system", "explore", "network", "general")},
            "latest_discovery": self._current_discovery,
        }
