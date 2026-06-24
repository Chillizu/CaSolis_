"""确定性模板引擎 — 安全的命令执行器

功能:
  输入: 意图名 + 参数 dict → 输出: args 列表 (无 shell=True)
  安全: subprocess.run(args=[...]), 白名单校验
"""

from __future__ import annotations

import subprocess
import shlex
import re
from dataclasses import dataclass
from typing import Any


# ── 命令模板 (从 JSON 配置加载) ──────────────────────────────

# P8.4c: 模板移入 config/command_registry.json
# 仅在 JSON 加载失败时使用这些硬编码后备

_EMBEDDED_COMMAND_TEMPLATES = {
    "READ":   (["cat"],               ["{path}"]),
    "LIST":   (["ls", "-la"],         ["{path}"]),
    "SEARCH": (["grep"],              ["{pattern}", "{path}"]),
    "INFO":   (None, None),
    "COUNT":  (["wc", "-l"],          ["{path}"]),
    "INSPECT":(["sh", "-c"],          ["command -v {cmd} 2>/dev/null || { echo 'not found'; true; }"]),
    "EXPLORE":(["ls", "-la"],         ["{path}"]),
    "HELP":   (["echo"],              ["(HELP disabled) {cmd}"]),
    "CUSTOM": (None, None),
    "READ_ETC":  (["cat"],             ["{path}"]),
    "USB_DEVICES":(["lsusb"],           None),
    "DISK_USAGE":(["du", "-sh", "{path}"], None),
    "LS_TMP":    (["ls", "-la"], ["/tmp"]),
    "ARCH_INFO": (["arch"], None),
    "WRITE":   (["sh", "-c"], ["printf '%s\n' {content} > {path}"]),
    "APPEND":  (["sh", "-c"], ["printf '%s\n' {content} >> {path}"]),
    "CAT":     (["cat"],      ["{path}"]),
}

_EMBEDDED_INFO_CMDS = {
    "cpu":     (["cat", "/proc/cpuinfo"], ["head", "-10"]),
    "mem":     (["cat", "/proc/meminfo"], ["head", "-10"]),
    "disk":    (["df", "-h"], None),
    "uptime":  (["uptime"], None),
    "whoami":  (["whoami"], ["id"]),
    "uname":   (["uname", "-a"], None),
    "date":    (["date"], None),
    "hostname":(["hostname"], None),
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
    "route":   (["ip", "route"], None),
}

_EMBEDDED_CUSTOM_COMMANDS = {
    "file":     {"args": ["file", "{path}"], "desc": "查看文件类型"},
    "stat":     {"args": ["stat", "{path}"], "desc": "查看文件详细信息"},
    "du":       {"args": ["du", "-sh", "{path}"], "desc": "查看文件/目录大小"},
    "which":    {"args": ["which", "{cmd}"], "desc": "查找命令路径"},
    "type":     {"args": ["type", "{cmd}"], "desc": "查看命令类型"},
    "pstree":   {"args": ["pstree"], "desc": "进程树"},
    "top_brief":{"args": ["ps", "-eo", "pid,ppid,cmd,%mem,%cpu", "--sort=-%mem"], "desc": "进程列表"},
    "lsmod":    {"args": ["lsmod"], "desc": "内核模块"},
    "lspci":    {"args": ["lspci"], "desc": "PCI 设备"},
    "lsusb":    {"args": ["lsusb"], "desc": "USB 设备"},
    "env_vars": {"args": ["env"], "desc": "环境变量"},
    "locale":   {"args": ["locale"], "desc": "区域设置"},
    "timedate": {"args": ["timedatectl"], "desc": "时间设置"},
    "mounts":   {"args": ["mount"], "desc": "挂载信息"},
    "inodes":   {"args": ["df", "-i"], "desc": "inode 使用情况"},
    "dns":      {"args": ["cat", "/etc/resolv.conf"], "desc": "DNS 配置"},
    "hosts":    {"args": ["cat", "/etc/hosts"], "desc": "主机映射"},
    "services": {"args": ["cat", "/etc/services"], "desc": "服务端口映射"},
    "completions":{"args": ["compgen", "-c"], "desc": "所有可用命令"},
    "hostname": {"args": ["cat", "/etc/hostname"], "desc": "主机名"},
    "wc_hosts": {"args": ["wc", "-l", "/etc/hosts"], "desc": "hosts行数"},
    "sha1sum_hosts":{"args": ["sha1sum", "/etc/hosts"], "desc": "hosts哈希"},
    "lsof_hosts":  {"args": ["lsof", "/etc/hosts"], "desc": "hosts进程"},
    "realpath_hosts":{"args": ["realpath", "/etc/hosts"], "desc": "hosts真实路径"},
    "id_version":  {"args": ["id"], "desc": "用户身份"},
    "mount_info":  {"args": ["mount"], "desc": "挂载点"},
    "groups":     {"args": ["groups"], "desc": "用户组"},
    "kernel_modules":{"args": ["lsmod"], "desc": "内核模块"},
    "disk_usage":  {"args": ["du", "-sh", "/"], "desc": "根目录大小"},
}

_EMBEDDED_MULTI_COMMANDS = {
    "INFO": [
        (["cat", "/proc/cpuinfo"], ["head", "-5"]),
        (["free", "-h"], None),
        (["uname", "-a"], None),
        (["uptime"], None),
        (["df", "-h"], None),
    ],
    "READ": [
        (["cat", "{path}"], None),
        (["wc", "-l", "{path}"], None),
        (["stat", "{path}"], None),
    ],
    "SEARCH": [
        (["grep", "-i", "{pattern}", "{path}"], None),
        (["grep", "-c", "{pattern}", "{path}"], None),
        (["grep", "-n", "{pattern}", "{path}"], ["head", "-10"]),
    ],
    "LIST": [
        (["ls", "-la", "{path}"], None),
        (["ls", "-la", "{path}/"], ["head", "-10"]),
    ],
    "COUNT": [
        (["wc", "-l", "{path}"], None),
        (["wc", "-c", "{path}"], None),
    ],
    "INSPECT": [
        (["command", "-v", "{cmd}"], None),
        (["type", "{cmd}"], None),
        (["which", "{cmd}"], None),
    ],
    "READ_ETC": [
        (["cat", "{path}"], None),
        (["wc", "-l", "{path}"], None),
        (["head", "-20", "{path}"], None),
    ],
    "DISK_USAGE": [
        (["du", "-sh", "{path}"], None),
        (["df", "-h"], None),
        (["df", "-i"], None),
    ],
    "USB_DEVICES": [
        (["lsusb"], None),
        (["lsusb", "-v"], ["head", "-30"]),
    ],
    "LS_TMP": [
        (["ls", "-la", "/tmp"], None),
        (["ls", "-la", "/tmp/"], ["head", "-10"]),
    ],
    "ARCH_INFO": [
        (["arch"], None),
        (["uname", "-m"], None),
    ],
    "WRITE": [
        (["sh", "-c"], ["printf '%s\n' {content} > {path}"]),
        (["cat", "{path}"], None),
        (["wc", "-l", "{path}"], None),
    ],
    "APPEND": [
        (["sh", "-c"], ["printf '%s\n' {content} >> {path}"]),
        (["cat", "{path}"], None),
    ],
}


# ── 安全配置 ──────────────────────────────────────────────────

# 允许的路径前缀
SAFE_PATHS = [
    "/proc/", "/etc/", "/tmp/", "/usr/",
    "/var/", "/sys/", "/home/", "/",
    "/workspace/",  # 容器内工作区
]

# 开放: 所有命令默认允许 (容器只读+零能力+无网络)
SAFE_COMMANDS: set[str] | None = None

BLOCKED_COMMANDS = {"sudo", "shutdown", "reboot", "poweroff", "halt"}

# 危险标志参数 — 放宽: 容器无法真正破坏
# 只保留 clearly destructive 的格式操作
DANGEROUS_FLAGS = {
    "--mkfs", "--format",  # 磁盘格式化 (但没用, 没设备)
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

    def __init__(self, dry_run: bool = False, timeout: int = 30, sandbox=None,
                 registry_path: str = ""):
        self.dry_run = dry_run
        self.timeout = timeout
        self.sandbox = sandbox
        self._stats = {"calls": 0, "errors": 0, "blocked": 0}

        # P8.4c: 从 JSON 加载命令模板, 失败时用内联后备
        self.command_templates = dict(_EMBEDDED_COMMAND_TEMPLATES)
        self.info_cmds = dict(_EMBEDDED_INFO_CMDS)
        self.custom_commands = dict(_EMBEDDED_CUSTOM_COMMANDS)
        self.multi_commands = dict(_EMBEDDED_MULTI_COMMANDS)
        self._load_registry(registry_path)

    def _load_registry(self, path: str):
        """从 JSON 配置加载命令注册表"""
        if not path:
            import os
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "config", "command_registry.json")
        try:
            import json
            with open(path) as f:
                registry = json.load(f)

            ct = registry.get("command_templates", {})
            for intent, cfg in ct.items():
                if cfg.get("special"):
                    self.command_templates[intent] = (None, None)
                else:
                    self.command_templates[intent] = (cfg["base"], cfg.get("args"))

            ic = registry.get("info_cmds", {})
            for target, cfg in ic.items():
                self.info_cmds[target] = (cfg["base"], cfg.get("pipe"))

            cc = registry.get("custom_commands", {})
            for key, cfg in cc.items():
                self.custom_commands[key] = {"args": cfg["args"], "desc": cfg.get("desc", "")}

            mc = registry.get("multi_commands", {})
            for intent, cmds in mc.items():
                seq = []
                for c in cmds:
                    if c.get("pipe"):
                        seq.append((c["base"], c["pipe"]))
                    else:
                        seq.append((c["base"], None))
                self.multi_commands[intent] = seq

        except Exception as e:
            print(f"  [TemplateEngine] 加载 {path} 失败: {e}, 使用内联后备")

    def register_template(self, intent: str, base: list[str],
                          args: list[str] | None = None,
                          multi: list[tuple[list[str], list[str] | None]] | None = None):
        """
        P8.4c: 运行时注册新意图的命令模板
        用于新意图自动接入 (P6.4) 后立即注册模板
        """
        self.command_templates[intent] = (base, args)
        if multi:
            self.multi_commands[intent] = multi

    def build_args(self, intent: str, params: dict[str, Any]) -> list[str] | None:
        """从意图+参数构建安全的 args list"""
        if intent == "INFO":
            target = params.get("target", "uname")
            template = self.info_cmds.get(target)
            if template is None:
                return None
            args1, args2 = template
            if args2:
                return args1 + args2
            return args1

        if intent == "CUSTOM":
            return params.get("custom_args")

        # P0: 安全写入 — base64 + python3, 避免 shell 注入
        if intent in ("WRITE", "APPEND"):
            import base64 as _b64
            content = params.get("content", "")
            path = params.get("path", "/tmp/output.txt")
            op = ">" if intent == "WRITE" else ">>"
            encoded = _b64.b64encode(content.encode()).decode()
            shell_cmd = (
                f"python3 -c \"import base64; "
                f"data=base64.b64decode('{encoded}'); "
                f"f=open('{path}','a' if '{op}' == '>>' else 'wb'); "
                f"f.write(data); f.close(); "
                f"print(len(data))\""
            )
            return ["sh", "-c", shell_cmd]

        template = self.command_templates.get(intent)
        if template is None:
            return None

        base, arg_templates = template
        args = list(base)

        if arg_templates is not None:
            for t in arg_templates:
                if "{" in t and "}" in t:
                    # 模板参数替换: 支持 {key} 嵌入在字符串中的情况
                    filled = t
                    for key, value in params.items():
                        placeholder = "{" + key + "}"
                        if placeholder in filled:
                            filled = filled.replace(placeholder, str(value))
                    # 检查是否还有未替换的 {key} 占位符 (排除 shell 语法的大括号)
                    import re
                    unreplaced = re.findall(r'\{[a-zA-Z_]\w*\}', filled)
                    if unreplaced:
                        return None
                    args.append(filled)
                else:
                    # 字面参数: 直接追加
                    args.append(t)

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

        # 允许的命令 (None = 全部允许, 仅黑名单过滤)
        if SAFE_COMMANDS is not None and cmd not in SAFE_COMMANDS:
            self._stats["blocked"] += 1
            return False, f"命令不在白名单: {cmd}"

        # 路径参数检查
        for arg in args[1:]:
            if arg.startswith("/"):
                if not any(arg.startswith(p) for p in SAFE_PATHS):
                    self._stats["blocked"] += 1
                    return False, f"路径不在安全范围内: {arg}"

        # 危险标志检查 (拒绝任何含破坏性标志的命令)
        for arg in args[1:]:
            if arg in DANGEROUS_FLAGS:
                self._stats["blocked"] += 1
                return False, f"危险标志被禁止: {arg}"

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

    def _build_multi_cmd(self, intent: str, params: dict[str, Any], index: int) -> tuple[list[str] | None, str | None]:
        """
        构建多命令序列中第 index 条命令.
        Returns: (args_list, shell_cmd) — 二选一, 有 pipe 时用 shell_cmd
        """
        cmds = self.multi_commands.get(intent)
        if not cmds or index >= len(cmds):
            return None, None

        args_template, pipe_template = cmds[index]
        args = list(args_template)

        # 模板参数替换 (支持 {path}/ 这种带后缀的写法)
        args_str = " ".join(args)
        for key in list(params.keys()):
            placeholder = "{" + key + "}"
            if placeholder in args_str:
                args_str = args_str.replace(placeholder, str(params[key]))
        args = shlex.split(args_str)
        # 检查是否所有占位符都已替换
        if "{" in args_str and "}" in args_str:
            return None, None

        if pipe_template:
            # pipe_template 是 args 列表, 如 ["head", "-5"]
            # 拼成 shell pipe 目标: "head -5"
            pipe_dest = " ".join(shlex.quote(p) for p in pipe_template)
            main_part = " ".join(shlex.quote(a) for a in args)
            shell_cmd = f"{main_part} | {pipe_dest}"
            return args, shell_cmd

        return args, None

    def execute_multi(self, intent: str, params: dict[str, Any], depth: int = 2) -> list[ExecResult]:
        """
        执行多命令序列, 返回结果列表
        depth: 执行前 depth 条命令 (1-3)
        """
        import time
        results = []
        n_cmds = min(depth, 3)

        for i in range(n_cmds):
            args, shell_cmd = self._build_multi_cmd(intent, params, i)
            if args is None:
                results.append(ExecResult("", f"序列第{i+1}条: 不支持的参数", 1, 0))
                break

            ok, msg = self.validate(args)
            if not ok:
                self._stats["blocked"] += 1
                results.append(ExecResult("", msg, 1, 0))
                break

            self._stats["calls"] += 1

            if self.dry_run:
                display = shell_cmd if shell_cmd else " ".join(args)
                results.append(ExecResult(f"[DRY RUN] {display}", "", 0, 0))
                continue

            try:
                start = time.monotonic()
                if self.sandbox:
                    cmd_str = shell_cmd if shell_cmd else " ".join(shlex.quote(a) for a in args)
                    r = self.sandbox.execute(cmd_str, timeout=self.timeout)
                    elapsed = (time.monotonic() - start) * 1000
                    if r.exit_code != 0:
                        self._stats["errors"] += 1
                    results.append(ExecResult(
                        stdout=r.stdout or "",
                        stderr=r.stderr or "",
                        exit_code=r.exit_code,
                        duration_ms=elapsed,
                    ))
                else:
                    if shell_cmd:
                        result = subprocess.run(
                            ["/bin/sh", "-c", shell_cmd],
                            capture_output=True, text=True, timeout=self.timeout,
                        )
                    else:
                        result = subprocess.run(
                            args,
                            capture_output=True, text=True, timeout=self.timeout,
                        )
                    elapsed = (time.monotonic() - start) * 1000
                    if result.returncode != 0:
                        self._stats["errors"] += 1
                    results.append(ExecResult(
                        stdout=result.stdout or "",
                        stderr=result.stderr or "",
                        exit_code=result.returncode,
                        duration_ms=elapsed,
                    ))
            except subprocess.TimeoutExpired:
                self._stats["errors"] += 1
                results.append(ExecResult("", f"超时 ({self.timeout}s)", 124, self.timeout * 1000))
                break
            except Exception as e:
                self._stats["errors"] += 1
                results.append(ExecResult("", str(e), 1, 0))
                break

        return results

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
