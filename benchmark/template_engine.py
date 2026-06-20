"""确定性模板引擎 — 安全的命令执行器

功能:
  输入: 意图名 + 参数 dict → 输出: args 列表 (无 shell=True)
  安全: subprocess.run(args=[...]), 白名单校验
"""

from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from typing import Any


# ── 命令模板 ──────────────────────────────────────────────────

# 意图 → (二进制, [固定参数, 模板参数, ...])
COMMAND_TEMPLATES = {
    "READ":   (["cat"],               ["{path}"]),
    "LIST":   (["ls", "-la"],         ["{path}"]),
    "SEARCH": (["grep"],              ["{pattern}", "{path}"]),
    "INFO":   (None, None),           # 特殊处理
    "COUNT":  (["wc", "-l"],          ["{path}"]),
    "INSPECT":(["which"],             ["{cmd}"]),
    "EXPLORE":(["ls", "-la"],         ["{target}"]),
    "HELP":   (["man"],               ["{cmd}"]),
    "CUSTOM": (None, None),           # 自由命令, CommandSelector 处理
    # 自动发现的意图
    "READ_ETC":  (["cat"],             ["{path}"]),
    "USB_DEVICES":(["lsusb"],           None),
    "DISK_USAGE":(["du", "-sh", "{path}"], None),
}

INFO_CMDS = {
    "cpu":     (["cat", "/proc/cpuinfo"], ["head", "-10"]),
    "mem":     (["cat", "/proc/meminfo"], ["head", "-10"]),
    "disk":    (["df", "-h"], None),
    "uptime":  (["uptime"], None),
    "whoami":  (["whoami"], ["id"]),
    "uname":   (["uname", "-a"], None),
    "date":    (["date"], None),
    "hostname":(["hostname"], None),
    # 新增
    "arch":    (["arch"], None),
    "lscpu":   (["lscpu"], None),
    "lsblk":   (["lsblk"], None),
    "free":    (["free", "-h"], None),
    "ps":      (["ps", "aux", "--sort=-%mem"], ["head", "-15"]),
    "uptime_long":(["uptime", "-p"], None),
    "dmesg":   (["dmesg"], ["tail", "-20"]),
    "ip_addr": (["ip", "addr"], None),
    "ss_conn": (["ss", "-tlnp"], None),
    "du_root": (["du", "-sh", "/*"], ["sort", "-rh", "head", "-10"]),
    # 网络
    "route":   (["ip", "route"], None),
}


# 自由命令 — 没有固定模板, CUSTOM 意图使用
CUSTOM_COMMANDS = {
    # 基础文件操作
    "file":     {"args": ["file", "{path}"], "desc": "查看文件类型"},
    "stat":     {"args": ["stat", "{path}"], "desc": "查看文件详细信息"},
    "du":       {"args": ["du", "-sh", "{path}"], "desc": "查看文件/目录大小"},
    "which":    {"args": ["which", "{cmd}"], "desc": "查找命令路径"},
    "type":     {"args": ["type", "{cmd}"], "desc": "查看命令类型"},
    # 进程
    "pstree":   {"args": ["pstree"], "desc": "进程树"},
    "top_brief":{"args": ["ps", "-eo", "pid,ppid,cmd,%mem,%cpu", "--sort=-%mem"], "desc": "进程列表(按内存)"},
    # 系统
    "lsmod":    {"args": ["lsmod"], "desc": "内核模块"},
    "lspci":    {"args": ["lspci"], "desc": "PCI 设备"},
    "lsusb":    {"args": ["lsusb"], "desc": "USB 设备"},
    "env_vars": {"args": ["env"], "desc": "环境变量"},
    "locale":   {"args": ["locale"], "desc": "区域设置"},
    "timedate": {"args": ["timedatectl"], "desc": "时间设置"},
    # 文件系统
    "mounts":   {"args": ["mount"], "desc": "挂载信息"},
    "inodes":   {"args": ["df", "-i"], "desc": "inode 使用情况"},
    # 网络
    "dns":      {"args": ["cat", "/etc/resolv.conf"], "desc": "DNS 配置"},
    "hosts":    {"args": ["cat", "/etc/hosts"], "desc": "主机映射"},
    "services": {"args": ["cat", "/etc/services"], "desc": "服务端口映射"},
    # Shell
    "completions":{"args": ["compgen", "-c"], "desc": "所有可用命令"},
}


# ── 安全配置 ──────────────────────────────────────────────────

# 允许的路径前缀
SAFE_PATHS = [
    "/proc/", "/etc/", "/tmp/", "/usr/",
    "/var/", "/sys/", "/home/", "/",
]

# 允许的命令
SAFE_COMMANDS = {
    # 基础
    "cat", "ls", "grep", "wc", "which", "man",
    "head", "tail", "sort", "uniq", "cut", "tr",
    "uptime", "whoami", "id", "uname", "df", "date", "hostname",
    # 系统信息 (新增)
    "arch", "lscpu", "lsblk", "free", "ps", "ss", "ip",
    "dmesg", "pstree", "lsmod", "lspci", "lsusb", "env",
    "locale", "timedatectl", "mount",
    # 文件
    "file", "stat", "du", "type",
    # shell
    "compgen", "printf", "echo", "seq",
}

# 危险命令黑名单
BLOCKED_COMMANDS = {
    "rm", "chmod", "chown", "dd", "mkfs",
    "mount", "umount", "shutdown", "reboot",
    "sudo", "su", "passwd",
}


# ── 模板引擎 ────────────────────────────────────────────────

@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


class TemplateEngine:
    """安全的命令生成和执行器"""

    def __init__(self, dry_run: bool = False, timeout: int = 30, sandbox=None):
        self.dry_run = dry_run
        self.timeout = timeout
        self.sandbox = sandbox  # SandboxExecutor (可选, 用于隔离执行)
        self._stats = {"calls": 0, "errors": 0, "blocked": 0}

    def build_args(self, intent: str, params: dict[str, Any]) -> list[str] | None:
        """从意图+参数构建安全的 args list"""
        if intent == "INFO":
            target = params.get("target", "uname")
            template = INFO_CMDS.get(target)
            if template is None:
                return None
            args1, args2 = template
            if args2:
                return args1 + args2
            return args1

        if intent == "CUSTOM":
            # CUSTOM 意图: params['custom_args'] 里已经有完整的 args list
            return params.get("custom_args")

        template = COMMAND_TEMPLATES.get(intent)
        if template is None:
            return None

        base, arg_templates = template
        args = list(base)

        if arg_templates is not None:
            for t in arg_templates:
                key = t.strip("{}")
                value = params.get(key)
                if value is None:
                    return None
                args.append(str(value))

        return args

    def validate(self, args: list[str]) -> tuple[bool, str]:
        """安全检查"""
        if not args:
            return False, "空命令"

        cmd = args[0]

        # 检查命令是否在黑名单
        if cmd in BLOCKED_COMMANDS:
            self._stats["blocked"] += 1
            return False, f"命令被禁止: {cmd}"

        # 允许的命令
        if cmd not in SAFE_COMMANDS:
            self._stats["blocked"] += 1
            return False, f"命令不在白名单: {cmd}"

        # 路径参数检查
        for arg in args[1:]:
            if arg.startswith("/"):
                if not any(arg.startswith(p) for p in SAFE_PATHS):
                    self._stats["blocked"] += 1
                    return False, f"路径不在安全范围内: {arg}"

        return True, ""

    def execute(self, intent: str, params: dict[str, Any]) -> ExecResult:
        """执行意图, 返回命令结果"""
        import time

        args = self.build_args(intent, params)
        if args is None:
            return ExecResult("", f"不支持的意图: {intent}", 1, 0)

        ok, msg = self.validate(args)
        if not ok:
            self._stats["blocked"] += 1
            return ExecResult("", msg, 1, 0)

        self._stats["calls"] += 1

        if self.dry_run:
            return ExecResult(f"[DRY RUN] {' '.join(args)}", "", 0, 0)

        try:
            start = time.monotonic()
            
            if self.sandbox:
                # 在 Docker 沙箱内执行
                cmd_str = " ".join(shlex.quote(a) for a in args)
                r = self.sandbox.execute(cmd_str, timeout=self.timeout)
                elapsed = (time.monotonic() - start) * 1000
                if r.exit_code != 0:
                    self._stats["errors"] += 1
                return ExecResult(
                    stdout=r.stdout or "",
                    stderr=r.stderr or "",
                    exit_code=r.exit_code,
                    duration_ms=elapsed,
                )
            else:
                # 宿主执行 (fallback)
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                if result.returncode != 0:
                    self._stats["errors"] += 1
                return ExecResult(
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    exit_code=result.returncode,
                    duration_ms=elapsed,
                )
        except subprocess.TimeoutExpired:
            self._stats["errors"] += 1
            return ExecResult("", f"超时 ({self.timeout}s)", 124, self.timeout * 1000)
        except Exception as e:
            self._stats["errors"] += 1
            return ExecResult("", str(e), 1, 0)

    def stats(self) -> dict:
        return {**self._stats}


# ── 测试 ────────────────────────────────────────────────────

def test():
    engine = TemplateEngine(dry_run=True)

    test_cases = [
        ("READ", {"path": "/etc/hostname"}),
        ("SEARCH", {"pattern": "root", "path": "/etc/passwd"}),
        ("INFO", {"target": "cpu"}),
        ("LIST", {"path": "/"}),
        ("INSPECT", {"cmd": "python3"}),
    ]

    print("模板引擎测试:")
    for intent, params in test_cases:
        args = engine.build_args(intent, params)
        ok, msg = engine.validate(args)
        status = "✅" if ok else "❌"
        print(f"  {status} {intent:10s} {params!s:40s} → {' '.join(args)}")

    # 测试安全拦截
    bad_cases = [
        ("READ", {"path": "/etc/passwd; rm -rf /"}),
        ("INSPECT", {"cmd": "rm"}),
    ]
    print("\n安全检查测试:")
    for intent, params in bad_cases:
        result = engine.execute(intent, params)
        status = "✅" if result.exit_code != 0 else "❌"
        print(f"  {status} {intent:10s} {params!s:40s} → {result.stderr[:50]}")


if __name__ == "__main__":
    test()
