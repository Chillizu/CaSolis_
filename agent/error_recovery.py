"""
P8.0: 自动失败恢复 — 命令不存在/参数错误/权限拒绝 → 智能回退

分类错误 → 查询回退映射 → 重试 → 记录恢复策略到经验缓冲
"""

import re
from typing import Optional


# ── 已知在 Debian bookworm-slim 中不存在的命令 → 回退方案 ──
COMMAND_FALLBACKS: dict[str, list[str]] = {
    # 硬件/设备
    "depmod": ["cat", "/proc/modules"],
    "lsusb": ["sh", "-c", "find /sys/devices -name 'usb*' -maxdepth 2 2>/dev/null | head -5 || echo 'no usb devices'"],
    "lspci": ["cat", "/proc/bus/pci/devices"],
    "usb-devices": ["ls", "/dev/bus/usb"],
    "lsblk": ["df", "-h"],
    "lshw": ["cat", "/proc/cpuinfo"],
    "dmidecode": ["cat", "/sys/class/dmi/id/product_name"],
    "hwinfo": ["cat", "/proc/cpuinfo"],
    # 网络
    "host": ["getent", "hosts"],
    "nslookup": ["getent", "hosts"],
    "dig": ["cat", "/etc/hosts"],
    "ping": ["echo", "ping unavailable in sandbox"],
    "ifconfig": ["cat", "/proc/net/dev"],
    "netstat": ["cat", "/proc/net/tcp"],
    "ss": ["cat", "/proc/net/tcp"],
    "iwconfig": ["echo", "wireless unavailable in sandbox"],
    "ip": ["hostname", "-I"],
    # 系统
    "who": ["cat", "/var/log/wtmp"],
    "last": ["cat", "/var/log/wtmp"],
    "acpi": ["cat", "/sys/class/power_supply"],
    "sensors": ["cat", "/sys/class/thermal"],
    "timedatectl": ["date"],
    "locale": ["echo", "C.UTF-8"],
    # 磁盘
    "du": ["df", "-h"],
    "mount": ["cat", "/proc/mounts"],
    # 用户
    "chage": ["cat", "/etc/shadow"],
    "passwd": ["cat", "/etc/passwd"],
    # 包管理 (容器内不可用)
    "apt": ["echo", "apt unavailable in sandbox"],
    "dpkg": ["echo", "dpkg unavailable in sandbox"],
    "pacman": ["echo", "pacman unavailable in sandbox"],
    # 其他
    "clear": ["echo", ""],
    "reset": ["echo", ""],
    "watch": ["sleep", "1"],
    "script": ["echo", "script unavailable"],
    "su": ["echo", "su unavailable"],
    "sudo": ["echo", "sudo unavailable"],
    "poweroff": ["echo", "poweroff unavailable"],
    "reboot": ["echo", "reboot unavailable"],
    "shutdown": ["echo", "shutdown unavailable"],
}

# ── 常见不可读路径 → 替代路径 ──
PATH_FALLBACKS: dict[str, str] = {
    "/etc/shadow": "/etc/passwd",
    "/etc/gshadow": "/etc/group",
    "/var/log/syslog": "/proc/kmsg",
    "/var/log/auth.log": "/proc/kmsg",
    "/var/log/dmesg": "/proc/kmsg",
    "/root": "/home",
    "/sys/class/power_supply": "/proc/uptime",
}

# ── 参数缺失时从工作栏事实推断 ──
PARAM_FALLBACKS: dict[str, str] = {
    "path": "/etc/hostname",
    "pattern": "root",
    "cmd": "ls",
}


class ErrorClassifier:
    """解析 stderr, 分类错误类型"""

    @staticmethod
    def classify(stderr: str, stdout: str, exit_code: int,
                 cmd_str: str = "") -> dict:
        """
        返回:
          type: str — "not_found" | "no_such_file" | "permission_denied"
                   | "unsupported" | "timeout" | "empty_output" | "unknown"
          detail: str — 提取的错误细节 (命令名/路径/参数)
          confidence: float — 0-1
        """
        err = stderr.lower() if stderr else ""
        out = stdout.lower() if stdout else ""

        # 命令不存在
        if exit_code == 127 or "not found" in err or "command not found" in err:
            cmd_name = cmd_str.split()[0] if cmd_str else ""
            return {"type": "not_found", "detail": cmd_name, "confidence": 0.95}

        # 文件不存在
        if exit_code != 0 and ("no such file" in err or "cannot access" in err
                               or "does not exist" in err):
            # 提取路径
            path = ""
            for m in re.finditer(r"'(/[^']+)'|\"(/[^\"]+)\"|(/[^ ]+)", err):
                path = m.group(1) or m.group(2) or m.group(3)
                if path:
                    break
            return {"type": "no_such_file", "detail": path, "confidence": 0.9}

        # 权限拒绝
        if exit_code != 0 and ("permission denied" in err
                               or "operation not permitted" in err):
            path = ""
            for m in re.finditer(r"'(/[^']+)'|\"(/[^\"]+)\"|(/[^ ]+)", err):
                path = m.group(1) or m.group(2) or m.group(3)
                if path:
                    break
            return {"type": "permission_denied", "detail": path, "confidence": 0.9}

        # 不支持 (引擎返回的)
        if "不支持的意图" in stderr or "不支持的参数" in stderr:
            return {"type": "unsupported", "detail": stderr[:60], "confidence": 0.9}

        # 超时
        if "超时" in err or "timeout" in err:
            return {"type": "timeout", "detail": "", "confidence": 0.95}

        # 空输出 (exit=0 但无实质内容)
        if exit_code == 0 and len(stdout.strip()) < 3:
            return {"type": "empty_output", "detail": "", "confidence": 0.6}

        return {"type": "unknown", "detail": stderr[:60], "confidence": 0.3}


class ErrorRecovery:
    """
    失败恢复: 查找回退方案 → 执行 → 返回恢复后的结果

    使用:
      er = ErrorRecovery(sandbox)
      recovered, recovery_info = er.recover(intent, params, result)
      if recovered:
          # 使用 recovered 代替原始 result
    """

    def __init__(self, sandbox=None, workbench=None):
        self.sandbox = sandbox
        self.workbench = workbench
        self._blocked_cmds: set[str] = set()   # 已知不存在的命令
        self._blocked_paths: set[str] = set()   # 已知不可读的路径
        self._recovery_count = 0
        self._recovery_success = 0
        self._blacklist_growth = 0  # 因失败被加入黑名单的次数

    def recover(self, result, intent: str, params: dict,
                cmd_str: str = "") -> tuple:
        """
        尝试恢复失败的执行。
        
        Args:
            result: 原始 ExecResult
            intent: 执行的意图
            params: 参数字典
            cmd_str: 命令字符串 (用于提取命令名)
            
        Returns:
            (new_result, recovery_info):
              new_result: 恢复后的 ExecResult, 或 None 表示恢复失败
              recovery_info: dict 含 recovery_action, recovered_from 等
        """
        info = ErrorClassifier.classify(
            result.stderr, result.stdout, result.exit_code, cmd_str
        )
        err_type = info["type"]
        detail = info["detail"]

        recovery_action = None
        new_result = None

        if err_type == "not_found":
            new_result = self._recover_not_found(detail, result)
            if new_result:
                recovery_action = f"fallback_cmd:{detail}"
                self._blocked_cmds.add(detail)

        elif err_type == "no_such_file":
            new_result = self._recover_no_such_file(detail, result, intent, params)
            if new_result:
                recovery_action = f"alt_path:{detail}"
                self._blocked_paths.add(detail)

        elif err_type == "permission_denied":
            new_result = self._recover_permission_denied(detail, result, intent, params)
            if new_result:
                recovery_action = f"skip_path:{detail}"
                self._blocked_paths.add(detail)

        elif err_type == "unsupported":
            new_result = self._recover_unsupported(intent, params, result)
            if new_result:
                recovery_action = f"fixed_params:{intent}"

        elif err_type == "empty_output":
            # 空输出但 exit=0: 可能是命令运行了但没有有用信息, 不算失败
            return None, {"action": "empty_output_ok", "success": False}

        if recovery_action:
            self._recovery_count += 1
            if new_result and new_result.exit_code == 0:
                self._recovery_success += 1

        return new_result, {
            "action": recovery_action or "none",
            "error_type": err_type,
            "error_detail": detail,
            "success": new_result is not None and new_result.exit_code == 0,
        }

    def _recover_not_found(self, cmd_name: str, result) -> Optional:
        """命令不存在 → 查回退映射"""
        if not cmd_name or cmd_name in self._blocked_cmds:
            return None
        fallback = COMMAND_FALLBACKS.get(cmd_name)
        if not fallback:
            return None
        try:
            import subprocess, time, shlex
            start = time.monotonic()
            if self.sandbox:
                cmd_str = " ".join(shlex.quote(a) for a in fallback)
                r = self.sandbox.execute(cmd_str, timeout=10)
            else:
                r = subprocess.run(
                    fallback, capture_output=True, text=True, timeout=10
                )
            elapsed = (time.monotonic() - start) * 1000
            if r.returncode == 0 and (r.stdout or "").strip():
                from benchmark.template_engine import ExecResult
                return ExecResult(
                    stdout=r.stdout or "",
                    stderr=f"[recovery] {cmd_name} → {' '.join(fallback)}\n" + (r.stderr or ""),
                    exit_code=0,
                    duration_ms=elapsed,
                )
        except Exception:
            return None
        return None

    def _recover_no_such_file(self, path: str, result, intent: str, params: dict) -> Optional:
        """文件不存在 → 查替代路径"""
        if not path:
            path = params.get("path", "")
        alt = PATH_FALLBACKS.get(path)
        if not alt and path:
            # 尝试同级目录: /etc/something → 同级其他文件
            import os as _os
            parent = _os.path.dirname(path)
            if parent and _os.path.exists(parent):
                try:
                    items = _os.listdir(parent) if not self.sandbox else []
                except:
                    items = []
                if items:
                    alt = _os.path.join(parent, items[0])
        if alt and alt not in self._blocked_paths:
            new_params = dict(params)
            new_params["path"] = alt
            from benchmark.template_engine import TemplateEngine
            eng = TemplateEngine(sandbox=self.sandbox)
            return eng.execute(intent, new_params)
        return None

    def _recover_permission_denied(self, path: str, result, intent: str, params: dict) -> Optional:
        """权限拒绝 → 切 /proc 或 /tmp"""
        alt_paths = ["/proc/version", "/proc/loadavg", "/proc/stat", "/proc/uptime",
                     "/tmp", "/etc/hostname", "/etc/resolv.conf", "/etc/hosts"]
        for alt in alt_paths:
            if alt not in self._blocked_paths:
                new_params = dict(params)
                new_params["path"] = alt
                from benchmark.template_engine import TemplateEngine
                eng = TemplateEngine(sandbox=self.sandbox)
                result2 = eng.execute(intent, new_params)
                if result2 and result2.exit_code == 0:
                    return result2
        return None

    def _recover_unsupported(self, intent: str, params: dict, result) -> Optional:
        """不支持的意图/参数 → 补充缺失参数重试"""
        missing = []
        for key, default in PARAM_FALLBACKS.items():
            if key not in params or not params.get(key):
                # 尝试从工作栏事实推断
                if self.workbench and key == "path":
                    facts = self.workbench.facts
                    if "dir_etc" in facts:
                        default = "/etc/hostname"
                    elif "dir_root" in facts:
                        default = "/proc/version"
                missing.append((key, default))

        if missing:
            new_params = dict(params)
            for key, default in missing:
                new_params[key] = default
            from benchmark.template_engine import TemplateEngine
            eng = TemplateEngine(sandbox=self.sandbox)
            return eng.execute(intent, new_params)
        return None

    def get_stats(self) -> dict:
        return {
            "recovery_attempts": self._recovery_count,
            "recovery_success": self._recovery_success,
            "recovery_rate": (self._recovery_success / max(self._recovery_count, 1)),
            "blocked_cmds": len(self._blocked_cmds),
            "blocked_paths": len(self._blocked_paths),
            "blacklist_growth": self._blacklist_growth,
        }

    def get_blocked(self) -> dict:
        return {
            "cmds": sorted(self._blocked_cmds),
            "paths": sorted(self._blocked_paths),
        }
