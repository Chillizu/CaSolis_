"""
KnowledgeMapper — 知识拓展引擎

系统性地探索 --network none Docker 沙箱内所有可到达的信息源,
将新发现的事实自动注入 FactGraph。

探索阶段:
  A: 静态清单 — 枚举所有可发现的资源
  B: 命令分类 — 了解每个命令做什么
  C: 子系统探测 — 扫描 /proc /sys /etc 的深层结构
  D: 只读执行 — 运行探测命令收集动态状态
  E: 能力推断 — "这台机能做什么"

安全:
  - 只读白名单: 禁止 rm/dd/mkfs/iptables/passwd/reboot
  - 每命令 timeout 2s, 输出截断 4KB
  - --network none 天然防外连
"""

import re
import time
from typing import Any, Callable, Optional


# ── 只读白名单 ──
# 禁止执行的命令前缀
BLOCKED_PREFIXES = (
    "rm ", "dd ", "mkfs", "fdisk", "iptables", "reboot", "shutdown",
    "passwd", "chmod", "chown", "kill", "pkill", "mount", "umount",
    "wget", "curl", "nc ", "telnet",
)

# ── 探索阶段配置 ──
PHASES = {
    "A": {
        "name": "静态清单",
        "commands": [
            # 命令枚举
            "ls /usr/bin/ 2>/dev/null | head -200",
            "ls /bin/ 2>/dev/null | head -100",
            "ls /sbin/ 2>/dev/null | head -100",
            "ls /usr/local/bin/ 2>/dev/null | head -50",
            # 包列表
            "dpkg -l 2>/dev/null | wc -l",
            "dpkg -l 2>/dev/null | head -30",  # 仅关键软件
            # 文件系统
            "ls /proc/ 2>/dev/null | head -100",
            "ls /sys/class/ 2>/dev/null",
            "ls /etc/ 2>/dev/null | head -80",
            "ls /dev/ 2>/dev/null | head -50",
            "/etc/os-release 2>/dev/null && cat /etc/os-release",
            # 用户和环境
            "cat /etc/passwd 2>/dev/null | head -10",
            "cat /etc/group 2>/dev/null | head -10",
            "env 2>/dev/null | head -20",
        ],
    },
    "B": {
        "name": "命令分类",
        "commands": [
            "whatis python3 2>/dev/null",
            "whatis gcc 2>/dev/null",
            "whatis git 2>/dev/null",
            "whatis curl 2>/dev/null",
            "whatis perl 2>/dev/null",
            "whatis node 2>/dev/null",
            "whatis make 2>/dev/null",
            "whatis java 2>/dev/null",
            "python3 --version 2>/dev/null",
            "gcc --version 2>/dev/null | head -1",
            "perl --version 2>/dev/null | head -2",
        ],
    },
    "C": {
        "name": "子系统探测",
        "commands": [
            # 网络子系统
            "cat /proc/net/dev 2>/dev/null",
            "cat /proc/net/tcp 2>/dev/null | head -10",
            "cat /proc/net/route 2>/dev/null",
            "ls /sys/class/net/ 2>/dev/null",
            # 硬件子系统
            "cat /proc/cpuinfo 2>/dev/null | head -20",
            "cat /proc/meminfo 2>/dev/null | head -15",
            "cat /proc/uptime 2>/dev/null",
            "cat /proc/loadavg 2>/dev/null",
            "cat /proc/1/cgroup 2>/dev/null | head -10",
            "ls /sys/devices/ 2>/dev/null",
            "cat /proc/modules 2>/dev/null | head -20",
            # 内核参数
            "cat /proc/sys/kernel/hostname 2>/dev/null",
            "cat /proc/sys/kernel/osrelease 2>/dev/null",
            "cat /proc/filesystems 2>/dev/null | head -20",
        ],
    },
    "D": {
        "name": "动态探测",
        "commands": [
            "uname -a 2>/dev/null",
            "hostname 2>/dev/null",
            "df -h 2>/dev/null",
            "df -T 2>/dev/null | head -10",
            "ip addr 2>/dev/null | head -15",
            "ip route 2>/dev/null",
            "ss -tuln 2>/dev/null | head -10",
            "ps aux 2>/dev/null | head -15",
            "ps -eo pid,ppid,cmd 2>/dev/null | head -20",
            "lscpu 2>/dev/null | head -15",
            "lsblk 2>/dev/null | head -10",
            "free -h 2>/dev/null",
            "mount 2>/dev/null | head -10",
            "locale 2>/dev/null",
            "timedatectl 2>/dev/null | head -8",
        ],
    },
    "E": {
        "name": "能力推断",
        "commands": [
            # 语言运行时
            "which python3 2>/dev/null && python3 -c 'import sys; print(\"python3=\"+sys.version[:10])'",
            "which node 2>/dev/null && node --version 2>/dev/null",
            "which perl 2>/dev/null && perl -e 'print \"perl=available\"' 2>/dev/null",
            "which ruby 2>/dev/null && ruby --version 2>/dev/null",
            # 编译工具
            "which gcc 2>/dev/null && gcc --version 2>/dev/null | head -1",
            "which make 2>/dev/null && echo 'make=available'",
            "which cmake 2>/dev/null && echo 'cmake=available'",
            # 脚本 shell
            "which bash 2>/dev/null && bash --version 2>/dev/null | head -1",
            "which python3 2>/dev/null && echo 'can_run_python=true'",
            # 包管理器
            "which apt 2>/dev/null && echo 'apt=available'",
            "which dpkg 2>/dev/null && dpkg --version 2>/dev/null | head -1",
            # 文本处理
            "which awk 2>/dev/null && awk --version 2>/dev/null | head -1",
            "which sed 2>/dev/null && sed --version 2>/dev/null | head -1",
            "which grep 2>/dev/null && grep --version 2>/dev/null | head -1",
            "which jq 2>/dev/null && jq --version 2>/dev/null",
            # 系统工具
            "which systemctl 2>/dev/null && echo 'systemctl=available'",
            "which docker 2>/dev/null && echo 'docker=available'",
            "which git 2>/dev/null && git --version 2>/dev/null",
        ],
    },
}

# ── 事实提取器 (从命令输出中提取结构化事实) ──

EXTRACTORS: dict[str, Callable[[str, str], list[tuple[str, Any, str, float]]]] = {}


def _register(pattern: str):
    """装饰器: 注册提取器"""
    def decorator(fn):
        EXTRACTORS[pattern] = fn
        return fn
    return decorator


# --- Phase A 提取 ---

@_register("dpkg -l")
def extract_packages(cmd: str, output: str) -> list[tuple]:
    """从 dpkg -l 提取已安装包"""
    facts = []
    lines = output.strip().split('\n')
    started = False
    for line in lines:
        if line.startswith('+++'):
            started = True
            continue
        if not started:
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0] == 'ii':
            pkg_name = parts[1]
            facts.append((f"pkg_{pkg_name}", pkg_name, "package", 0.9))
    if len(facts) > 20:
        return [("n_installed_packages", str(len(facts)), "package", 0.9)]
    return facts[:20]


@_register("/etc/os-release")
def extract_os_release(cmd: str, output: str) -> list[tuple]:
    """从 /etc/os-release 提取系统信息"""
    facts = []
    for line in output.strip().split('\n'):
        if '=' in line:
            key, val = line.split('=', 1)
            val = val.strip('"\'')
            k = key.lower().replace('_id', '').replace('_name', 'name')
            facts.append((f"os_{k}", val, "system", 1.0))
    return facts


@_register("ls /usr/bin/")
def extract_commands(cmd: str, output: str) -> list[tuple]:
    """从 ls /usr/bin 提取命令列表"""
    cmds = [c.strip() for c in output.strip().split('\n') if c.strip() and not c.startswith('ls')]
    if len(cmds) > 5:
        return [("n_available_commands", str(len(cmds)), "system", 0.9)]
    return []


# --- Phase C 提取 ---

@_register("cat /proc/cpuinfo")
def extract_cpu(cmd: str, output: str) -> list[tuple]:
    facts = []
    for line in output.strip().split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            k = k.strip().lower().replace(' ', '_')
            v = v.strip()
            if k in ('processor', 'model_name', 'cpu_cores', 'cpu_family', 'vendor_id'):
                facts.append((f"cpu_{k}", v, "system", 0.95))
    return facts


@_register("cat /proc/meminfo")
def extract_mem(cmd: str, output: str) -> list[tuple]:
    facts = []
    for line in output.strip().split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ('memtotal', 'memfree', 'swaptotal', 'swapfree'):
                facts.append((f"mem_{k}", v, "system", 0.95))
    return facts


@_register("cat /proc/net/dev")
def extract_net_dev(cmd: str, output: str) -> list[tuple]:
    facts = []
    for line in output.strip().split('\n')[2:]:  # 跳过表头
        parts = line.split()
        if len(parts) >= 10 and ':' in parts[0]:
            iface = parts[0].rstrip(':')
            rx_bytes = parts[1]
            tx_bytes = parts[9]
            facts.append((f"net_iface_{iface}", f"RX={rx_bytes} TX={tx_bytes}", "network", 0.8))
            facts.append((f"net_ifaces_available", iface, "network", 0.9))
    return facts


@_register("cat /proc/1/cgroup")
def extract_container(cmd: str, output: str) -> list[tuple]:
    if 'docker' in output.lower() or 'lxc' in output.lower():
        return [("is_container", True, "system", 1.0)]
    return []


# --- Phase D 提取 ---

@_register("ip addr")
def extract_ip(cmd: str, output: str) -> list[tuple]:
    facts = []
    for line in output.strip().split('\n'):
        m = re.search(r'inet (\d+\.\d+\.\d+\.\d+/\d+)', line)
        if m:
            facts.append(("ip_address", m.group(1), "network", 0.9))
        m = re.search(r'link/ether ([0-9a-f:]+)', line.lower())
        if m:
            facts.append(("mac_address", m.group(1), "network", 0.9))
    return facts


@_register("uname -a")
def extract_uname(cmd: str, output: str) -> list[tuple]:
    parts = output.strip().split()
    if len(parts) >= 3:
        return [
            ("kernel_version", parts[2].rstrip(','), "system", 1.0),
            ("hostname_info", parts[1], "system", 0.9),
        ]
    return []


@_register("df -h")
def extract_df(cmd: str, output: str) -> list[tuple]:
    facts = []
    for line in output.strip().split('\n')[1:]:
        parts = line.split()
        if len(parts) >= 6 and parts[0].startswith('/'):
            facts.append((f"fs_{parts[0].replace('/', '_').strip('_')}",
                          f"size={parts[1]} used={parts[2]} avail={parts[3]} mount={parts[5]}",
                          "system", 0.85))
    return facts


@_register("ps aux")
def extract_ps(cmd: str, output: str) -> list[tuple]:
    lines = output.strip().split('\n')[1:]
    proc_count = len(lines)
    return [("n_processes", str(proc_count), "system", 0.8)]


# --- Phase E 提取 ---

@_register("which python3")
def extract_capabilities(cmd: str, output: str) -> list[tuple]:
    if 'python3' in output:
        return [(f"capability_python", True, "capability", 1.0)]
    return []


@_register("which gcc")
def extract_capability_gcc(cmd: str, output: str) -> list[tuple]:
    if 'gcc' in output:
        return [(f"capability_compile", True, "capability", 1.0)]
    return []


# ── 知识映射器主类 ──

class KnowledgeMapper:
    """知识拓展引擎"""

    def __init__(self, sandbox, workbench):
        self.sandbox = sandbox
        self.workbench = workbench
        self.completed_phases: set[str] = set()
        self._phase_results: dict[str, list[dict]] = {}  # phase → [discovered facts]
        self._phase_steps: dict[str, int] = {}  # phase → step completed

    def is_phase_done(self, phase: str) -> bool:
        return phase in self.completed_phases

    def run_phase(self, phase: str, step: int) -> int:
        """
        执行一个探索阶段

        Args:
            phase: A/B/C/D/E
            step: 当前步数

        Returns:
            新发现的事实数
        """
        if phase in self.completed_phases:
            return 0

        phase_config = PHASES.get(phase)
        if not phase_config:
            return 0

        print(f"  [KNOWLEDGE] Phase {phase}: {phase_config['name']}")
        total_new = 0

        for cmd_template in phase_config["commands"]:
            # 安全检查
            blocked = False
            for prefix in BLOCKED_PREFIXES:
                if cmd_template.lstrip().startswith(prefix):
                    blocked = True
                    break
            if blocked:
                continue

            # 执行命令
            result = self.sandbox.execute(
                cmd_template,
                timeout=3,
            )

            if not result or result.exit_code != 0:
                continue

            output = result.stdout.strip()[:4096]  # 截断 4KB
            if not output:
                continue

            # 提取事实
            new_facts = self._extract_facts(cmd_template, output)

            # 注入 Workbench/FactGraph
            for key, value, category, confidence in new_facts:
                wb = self.workbench
                step_src = f"km_phase_{phase}"
                # 检查是否已存在 (用 fact_history)
                existing = None
                # 确保值是字符串 (FactGraph 需要)
                str_value = str(value)

                if hasattr(wb, 'facts') and key in wb.facts:
                    existing = wb.facts[key]
                if hasattr(wb, 'graph') and key in wb.graph.nodes:
                    existing = True

                if existing:
                    continue  # 跳过已存在事实

                # 添加到 workbench facts
                if hasattr(wb, 'add_fact'):
                    wb.add_fact(key, str_value, category=category, confidence=confidence,
                                step=step, source_cmd=step_src)
                elif hasattr(wb, 'facts'):
                    wb.facts[key] = {
                        "value": str_value,
                        "category": category,
                        "confidence": confidence,
                        "step": step,
                        "source_cmd": step_src,
                    }

                # 添加到 FactGraph
                if hasattr(wb, 'graph'):
                    wb.graph.add_node(key, str_value, category=category,
                                      confidence=confidence, step=step, source_cmd=step_src)

                total_new += 1

            # 记录原始输出 (用于 schema 推断)
            self._record_raw_output(phase, cmd_template, output)

        # 标记阶段完成
        self.completed_phases.add(phase)
        self._phase_steps[phase] = step

        if total_new > 0:
            print(f"    → 新增 {total_new} 个事实")

        # 自动 schema 扩展
        self._auto_extend_schema()

        return total_new

    def _extract_facts(self, cmd: str, output: str) -> list[tuple]:
        """从命令输出中提取事实"""
        all_facts = []

        # 1. 使用注册的提取器
        for pattern, extractor in EXTRACTORS.items():
            if pattern in cmd:
                try:
                    facts = extractor(cmd, output)
                    all_facts.extend(facts)
                except Exception:
                    pass

        # 2. 通用提取: 对 Phase D/E 做 key=value 解析
        if all_facts:
            return all_facts

        # 3. 非常简单的默认提取: 对探测结果的摘要
        for line in output.strip().split('\n')[:3]:
            line = line.strip()
            if not line:
                continue
            # 试试 key=value 格式
            if '=' in line:
                k, v = line.split('=', 1)
                clean_key = k.strip().lower().replace(' ', '_').replace('-', '_')
                if clean_key and v.strip():
                    all_facts.append((clean_key, v.strip(), "general", 0.5))

        return all_facts

    def _record_raw_output(self, phase: str, cmd: str, output: str):
        """记录原始输出 (用于 schema 推断)"""
        if phase not in self._phase_results:
            self._phase_results[phase] = []
        self._phase_results[phase].append({
            "cmd": cmd,
            "output_length": len(output),
            "output_sample": output[:200],
        })

    def _auto_extend_schema(self):
        """自动检测新类别并扩展 FactGraph schema"""
        if not hasattr(self.workbench, 'graph'):
            return

        graph = self.workbench.graph
        # 获取当前所有 category
        categories = set()
        for node in graph.nodes.values():
            categories.add(node.category)

        # 已有 schemas
        existing_schemas = set(graph.schemas.keys()) if hasattr(graph, 'schemas') else set()

        # 推断新 schema
        for cat in categories:
            if cat not in existing_schemas and cat != "general":
                # 自动创建 schema
                if hasattr(graph, 'schemas'):
                    graph.schemas[cat] = {
                        "required": [],
                        "optional": [],
                        "description": f"Auto-detected from KnowledgeMapper (category: {cat})",
                        "_auto_detected": True,
                    }
                    print(f"    [SCHEMA] 自动扩展: 新增 category '{cat}'")

    def get_phase_stats(self) -> dict:
        """获取探索统计"""
        total_raw = sum(len(v) for v in self._phase_results.values())
        return {
            "completed_phases": sorted(self.completed_phases),
            "phases_remaining": [p for p in "ABCDE" if p not in self.completed_phases],
            "n_phase_steps": dict(self._phase_steps),
            "n_raw_outputs": total_raw,
        }

    def infer_capabilities(self) -> list[tuple]:
        """从已收集的事实推断系统能力"""
        facts = {}
        if hasattr(self.workbench, 'facts'):
            facts = self.workbench.facts
        elif hasattr(self.workbench, 'graph'):
            facts = {k: n.value for k, n in self.workbench.graph.nodes.items()}

        capabilities = []

        # 语言运行时
        for lang in ["python3", "node", "perl", "ruby"]:
            if any(lang in k for k in facts):
                capabilities.append((f"can_run_{lang.replace('3', '')}", "True", "capability", 1.0))

        # 编译
        if any("gcc" in k for k in facts):
            capabilities.append(("can_compile_c", "True", "capability", 1.0))
        if any("make" in k for k in facts):
            capabilities.append(("can_run_build", "True", "capability", 1.0))

        # 网络
        if any("iface" in k for k in facts):
            capabilities.append(("has_network", "False", "capability", 0.9))

        # 包管理器
        if any("apt" in k or "dpkg" in k for k in facts):
            capabilities.append(("has_package_manager", "True", "capability", 1.0))

        return capabilities
