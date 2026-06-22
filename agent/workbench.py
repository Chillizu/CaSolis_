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

    def __init__(self, max_facts: int = 40):
        self.facts: dict[str, dict] = {}  # key → {value, source, step, confidence, count}
        self.max_facts = max_facts
        self._current_discovery: Optional[str] = None  # 最新关键事实 key
        self._step_counter = 0

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

        # ── 命令名辅助的提取 (单命令 / CUSTOM) ──

        # hostname (裸命令)
        if lower.strip() == "hostname":
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

    def _add_fact(self, key: str, value: str, source_intent: str,
                  source_cmd: str, step: int, confidence: float = 1.0,
                  category: str = "general"):
        """添加或更新事实 (置信度递增)"""
        if not value or len(value) > 100:
            value = value[:100]

        if key in self.facts:
            old = self.facts[key]
            old["value"] = value
            old["confidence"] = min(old["confidence"] + 0.15, 1.0)
            old["step"] = step
            old["count"] = old.get("count", 1) + 1
            old["category"] = category
        else:
            if len(self.facts) >= self.max_facts:
                # 替换最旧的
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
        """
        基于工作栏事实, 推荐下一步的行动

        Returns:
          (intent_name, params) 或 None (无建议)
        """
        # 知道 hostname → 验证 hostname 命令
        if "hostname" in self.facts and "hostname_cmd" not in self.facts:
            return ("CUSTOM", {"custom_args": ["hostname"], "cluster": "SYSTEM"})
        # 知道 hostname → 查 hosts 文件
        if "hostname" in self.facts and "etchosts_hosts" not in self.facts:
            return ("READ", {"path": "/etc/hosts"})
        # 知道内核 → 查发行版
        if "kernel" in self.facts and "os_pretty_name" not in self.facts:
            if "os_release" not in [k for k in self.facts]:
                return ("READ", {"path": "/etc/os-release"})
        # 知道内存 → 查磁盘
        if "mem_total" in self.facts and "disk_root" not in self.facts:
            # 用 CUSTOM 的 df 而不是 INFO (INFO 不传参)
            return ("CUSTOM", {"custom_args": ["df", "-h"], "cluster": "SYSTEM"})
        # 有目录内容 → 读其中一个文件
        dir_facts = [k for k in self.facts if k.startswith("dir_")]
        for dk in sorted(dir_facts, key=lambda k: -self.facts[k]["confidence"]):
            if dk not in ("dir_tmp", "dir_root"):
                val = self.facts[dk]["value"]
                first_file = val.split(",")[0].strip()
                if first_file and first_file not in (".", ".."):
                    # 尝试 /etc/{file}
                    path = f"/etc/{first_file}" if dk == "dir_etc" else first_file
                    return ("CUSTOM", {"custom_args": ["head", "-5", path], "cluster": "FILE_READ"})

        return None

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
