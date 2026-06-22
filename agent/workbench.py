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


class Workbench:
    """工作栏: 事实存储 + 自动提取 + 状态摘要"""

    def __init__(self, max_facts: int = 40, meta_learner=None):
        self.facts: dict[str, dict] = {}  # key → {value, source, step, confidence, count}
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
        """添加或更新事实 (置信度递增, P5.4: 追踪提取效用)"""
        if not value or len(value) > 100:
            value = value[:100]
        # P6.1: 拒绝垃圾值
        if not self._is_valid_value(value):
            return

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
        if self.meta and is_new:
            self.meta.register(f"extract_{key}", "extraction",
                              {"key": key, "source": str(source_cmd)[:40]}, step)
            self.meta.record(f"extract_{key}", 1.0, step)

    # ── 查询接口 ──

    def get_current_discovery(self) -> Optional[str]:
        """返回最新发现的关键事实 key"""
        return self._current_discovery

    def get_fact(self, key: str) -> Optional[str]:
        """按 key 查询事实值"""
        return self.facts[key]["value"] if key in self.facts else None

    def get_facts_by_category(self, category: str) -> list[str]:
        """按类别查询所有事实 key"""
        return [k for k, v in self.facts.items() if v.get("category") == category]

    def get_follow_up(self) -> Optional[tuple[str, dict]]:
        """推荐下一步, 并存储到 last_follow_up"""
        result = self._compute_follow_up()
        self.last_follow_up = result
        return result

    def _compute_follow_up(self) -> Optional[tuple[str, dict]]:
        """实际的推荐逻辑 (先尝试配置链, 回退到硬编码)"""

        # P5.3: 先尝试配置驱动的链
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

    def get_curiosity_probe(self, explored_paths: set[str] | None = None) -> Optional[tuple[str, dict]]:
        """
        P6.0: 元学习加权的探针选择

        流程:
          1. 构建候选探针列表 (配置优先级 + fallback)
          2. 从 meta 读取历史效用分
          3. 按 基础分 + 效用×2 - 已探索惩罚 - 近期使用惩罚 排序
          4. 跳过总分 < -0.5 的探针
          5. 新探针首次自动注册到 meta
        """
        if explored_paths is None:
            return None

        probe_cfg = self.rules.get("curiosity_probes", {})
        now = self._step_counter
        candidates = []

        # 1. 配置里的高优先级文件路径
        for rank, p in enumerate(probe_cfg.get("priority", [])):
            cmd = ["cat", p]
            path_key = "cat " + p
            candidates.append({
                "cmd": cmd,
                "cluster": "PROCFS",
                "path_key": path_key,
                "base_score": 1.0 - rank * 0.05,
            })

        # 2. 回退命令
        fallbacks = probe_cfg.get("fallback_commands", [])
        if fallbacks:
            phase = len(explored_paths) % max(len(fallbacks), 1)
            for offset in range(len(fallbacks)):
                idx = (phase + offset) % len(fallbacks)
                cmd = list(fallbacks[idx])
                path_key = " ".join(cmd)
                candidates.append({
                    "cmd": cmd,
                    "cluster": "FILE_FIND" if "find" in path_key else "FILE_LIST",
                    "path_key": path_key,
                    "base_score": 0.35 - offset * 0.05,
                })

        # 3. 从元学习器取历史效用
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
            val = fact["value"]
            cmd = fact.get("source_cmd", "") or ""
            
            if key == "kernel":
                lines.append(f"echo '=== Kernel Info ==='")
                lines.append(f"echo 'Kernel: {val}'")
                lines.append("uname -a")
                lines.append("")
            elif key == "cpu_cores":
                lines.append(f"echo '=== CPU ==='")
                lines.append(f"echo 'Online cores: $(nproc)'")
                lines.append("cat /proc/loadavg")
                lines.append("")
            elif key == "mem_total":
                lines.append(f"echo '=== Memory ==='")
                lines.append("free -h")
                lines.append("")
            elif key == "disk_root":
                lines.append(f"echo '=== Disk ==='")
                lines.append(f"df -h | head -5")
                lines.append("")
            elif key == "hostname":
                lines.append(f"echo '=== Host ==='")
                lines.append(f"echo 'Hostname: {val}'")
                lines.append(f"echo 'Hostname (cmd): $(hostname)'")
                lines.append("")
            elif key == "current_user":
                lines.append(f"echo '=== User ==='")
                lines.append(f"echo 'User: {val}'")
                lines.append(f"id")
                lines.append("")
            elif key == "ip_addr":
                lines.append(f"echo '=== Network ==='")
                lines.append(f"echo 'IP: {val}'")
                lines.append("cat /sys/class/net/*/address 2>/dev/null || ip link")
                lines.append("")
            elif key == "os_pretty_name":
                lines.append(f"echo '=== OS ==='")
                lines.append(f"echo '{val}'")
                lines.append("cat /etc/os-release 2>/dev/null | head -3")
                lines.append("")
            else:
                lines.append(f"echo '=== {key} ==='")
                if cmd and cmd.startswith("["):
                    # 从列表表示提取: ['cat', '/etc/os-release']
                    import ast
                    try:
                        cmd_list = ast.literal_eval(cmd)
                        if isinstance(cmd_list, list) and len(cmd_list) == 1:
                            lines.append(self._find_command_for_key(key) or cmd_list[0])
                        elif isinstance(cmd_list, list):
                            lines.append(" ".join(cmd_list))
                    except:
                        lines.append(self._find_command_for_key(key) or cmd)
                else:
                    lines.append(self._find_command_for_key(key) or cmd)
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
        """根据事实 key 推荐一个可执行的 shell 命令"""
        cmd_map = {
            "kernel": "uname -a",
            "node_name": "uname -a",
            "architecture": "uname -a",
            "hostname": "cat /etc/hostname",
            "hostname_cmd": "hostname",
            "current_user": "whoami",
            "uid_info": "id",
            "cpu_cores": "cat /proc/cpuinfo | grep processor | wc -l",
            "cpu_model": "cat /proc/cpuinfo | grep 'model name' | head -1",
            "mem_total": "free -h | grep Mem",
            "swap_total": "free -h | grep Swap",
            "disk_root": "df -h /",
            "ip_addr": "ip addr show | grep 'inet '",
            "users": "cat /etc/passwd | cut -d: -f1",
            "etchosts_hosts": "cat /etc/hosts",
            "os_pretty_name": "cat /etc/os-release",
            "os_name": "cat /etc/os-release | grep '^NAME='",
            "os_version": "cat /etc/os-release | grep '^VERSION_ID='",
            "os_version_id": "cat /etc/os-release | grep '^VERSION_ID='",
            "os_id": "cat /etc/os-release | grep '^ID='",
            "os_version_codename": "cat /etc/os-release | grep '^VERSION_CODENAME='",
        }
        return cmd_map.get(key, "")

    # ── 状态摘要 ──

    def get_state_summary(self, max_keys: int = 4) -> str:
        """生成简短文本摘要 → 注入 StateEncoder.fact_summary"""
        if not self.facts:
            return "无"

        # 按 (置信度, 最近使用) 排序
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
        """持久化工作栏状态到文件 (含事实 + 链状态)"""
        if not path:
            # 默认: 项目 data/persistent/ 目录 (host 可写)
            import os
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "data", "persistent", "workbench_snapshot.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = {
                "facts": {k: {sk: sv for sk, sv in v.items()}
                         for k, v in self.facts.items()},
                "chain_step": self.chain_step,
                "chain_completed_at": self.chain_completed_at,
                "_current_discovery": self._current_discovery,
            }
            import json
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️ 工作栏保存失败: {e}")

    def load(self, path: str = ""):
        """从文件恢复工作栏状态"""
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
        return {
            "n_facts": len(self.facts),
            "categories": {c: len(self.get_facts_by_category(c))
                          for c in ("system", "explore", "network", "general")},
            "latest_discovery": self._current_discovery,
        }
