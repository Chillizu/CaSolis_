"""
SandboxExecutor — Docker 沙箱执行器

替代直接在宿主上跑 subprocess.run, 所有命令在容器内执行。
防止 GUI 弹窗、系统副作用等问题。
"""

import subprocess
import os
import time
import json
from typing import Optional
from dataclasses import dataclass

SANDBOX_IMAGE = "folunar-sandbox:latest"  # P9: 预装 python3+curl
CONTAINER_NAME = "folunar-sandbox"


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


# P5.3: 命令→包名映射 (auto-install 用)
COMMAND_PACKAGE_MAP = {
    "python3": "python3", "python": "python3",
    "pip": "python3-pip", "pip3": "python3-pip",
    "git": "git", "curl": "curl", "wget": "wget",
    "vim": "vim", "nano": "nano",
    "htop": "htop", "jq": "jq", "tree": "tree",
    "make": "make", "gcc": "gcc", "g++": "g++",
    "node": "nodejs", "npm": "npm", "ruby": "ruby",
    "perl": "perl", "lua": "lua5.4",
    "ifconfig": "net-tools",
    "zip": "zip", "unzip": "unzip",
    "bc": "bc", "lsof": "lsof", "strace": "strace",
    "nc": "netcat-openbsd", "socat": "socat",
    "rsync": "rsync",
    "screen": "screen", "tmux": "tmux",
}


class SandboxExecutor:
    """
    Docker 容器执行器

    启动一个持久容器, 所有命令通过 docker exec 执行。
    容器:
      - 非特权
      - 只读根文件系统
      - 无网络 (可选)
      - 自动清理
    """

    def __init__(self, image: str = SANDBOX_IMAGE, name: str = CONTAINER_NAME):
        self.image = image
        self.name = name
        self._installed_packages: set[str] = set()
        self._unavailable_commands: set[str] = set()
        self._ensure_image()
        self._ensure_container()

    def _ensure_image(self):
        """确保镜像存在, 不存在则拉取"""
        result = subprocess.run(
            ["docker", "images", "-q", self.image],
            capture_output=True, text=True, timeout=30
        )
        if not result.stdout.strip():
            print(f"  拉取镜像 {self.image}...")
            subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True, timeout=120
            )

    def _ensure_container(self):
        """确保容器运行"""
        # 检查容器是否已存在且运行
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={self.name}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10
        )
        if self.name in result.stdout.strip().split("\n"):
            return  # 已经在运行

        # 删除旧容器
        subprocess.run(
            ["docker", "rm", "-f", self.name],
            capture_output=True, timeout=10
        )

        # 启动新容器 (P9: 非只读, 预装python3+curl)
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.name,
                "--tmpfs", "/tmp",
                "--tmpfs", "/workspace:exec",  # P9: 允许执行脚本
                "-v", f"{os.getcwd()}/data/persistent:/persistent:rw",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--network", "none",
                "--rm",
                self.image,
                "sleep", "infinity",
            ],
            capture_output=True, timeout=30
        )

    def execute(self, cmd: str, timeout: int = 10) -> ExecResult:
        """在容器内执行命令 (自动检测缺失命令并安装)"""
        try:
            result = subprocess.run(
                [
                    "docker", "exec", "-i",
                    self.name,
                    "/bin/sh", "-c", cmd,
                ],
                capture_output=True, text=True,
                timeout=timeout,
                errors='replace',
            )
            er = ExecResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.returncode,
            )
            # P5.3: 检测缺失命令并自动安装
            if er.exit_code != 0 and er.stderr:
                self._try_auto_install(cmd, er)
            return er
        except subprocess.TimeoutExpired:
            return ExecResult("", f"TIMEOUT ({timeout}s)", -1)
        except Exception as e:
            return ExecResult("", str(e), -1)

    def _try_auto_install(self, cmd: str, result: ExecResult):
        """检测 stderr 中 'not found' 并记录不可用命令
        (容器只读+无网络, apt-get 不可用, 仅跟踪)"""
        stderr = result.stderr
        for cmd_name, pkg in COMMAND_PACKAGE_MAP.items():
            if cmd_name in self._unavailable_commands:
                continue
            if f"{cmd_name}: not found" in stderr or f"{cmd_name}: command not found" in stderr:
                self._unavailable_commands.add(cmd_name)

    def execute_list(self, args: list[str], timeout: int = 10) -> ExecResult:
        """在容器内执行命令 (参数列表形式)"""
        try:
            cmd = ["docker", "exec", "-i", self.name] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, errors='replace',
            )
            return ExecResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult("", f"TIMEOUT ({timeout}s)", -1)
        except Exception as e:
            return ExecResult("", str(e), -1)

    def install_packages(self, packages: list[str]):
        """在容器内安装包 (Debian apt)"""
        self.execute("apt-get update -qq", timeout=60)
        for pkg in packages:
            print(f"  安装 {pkg}...")
            self.execute(f"apt-get install -y -qq {pkg}", timeout=60)

    def close(self):
        """停止并删除容器"""
        subprocess.run(
            ["docker", "rm", "-f", self.name],
            capture_output=True, timeout=10
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
