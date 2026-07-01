"""
CommandMiner — 从元命令输出中挖掘新命令

流程:
  1. 解析 compgen -c / ls /usr/bin 输出 → 提取命令名
  2. 黑名单过滤
  3. 去重 (resolve symlinks, 过滤 alias/builtin)
  4. 沙箱试跑 (bare + --help)
  5. 归入 cluster
  6. 生成安全参数模式
  7. 注册到 HierarchicalSelector
"""

import json, os, re, subprocess
from typing import Optional
from collections import defaultdict


# ── 黑名单 (按命令名过滤) ──
BLACKLIST = {
    # 写入/删除
    "rm", "dd", "mkfs", "mkfs.*", "fdisk", "mount", "umount",
    "cp", "mv", "tee", "install", "mktemp", "rename",
    "chmod", "chown", "chattr", "ln", "unlink",
    "truncate", "fallocate", "touch",
    # 系统管理 (写)
    "systemctl", "systemd-*", "passwd", "chpasswd",
    "reboot", "shutdown", "init", "halt", "poweroff",
    "kexec", "telinit", "runlevel", "sysctl -w",
    # 包管理
    "pacman", "apt", "apt-get", "yum", "dnf", "dpkg", "rpm",
    "zypper", "snap", "flatpak", "pip", "pip3", "npm", "gem",
    # 网络操作
    "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
    "ip link set", "ip addr add", "route add", "iwconfig",
    # 解释器 (可执行任意代码)
    "bash", "sh", "zsh", "fish", "dash", "ksh", "tcsh",
    "python", "python3", "perl", "ruby", "php", "lua",
    "tclsh", "expect", "node", "deno", "groovy", "scala",
    "gawk", "mawk", "nawk",
    # 编译/构建
    "gcc", "g++", "cc", "c++", "clang", "clang++",
    "make", "cmake", "ninja", "cargo", "go", "rustc",
    "javac", "java", "kotlin", "swiftc",
    # 下载/外联
    "curl", "wget", "ftp", "lftp", "scp", "sftp", "rsync",
    "nc", "ncat", "socat", "aria2c", "axel",
    # SSH/远程
    "ssh", "sshd", "telnet", "rsh", "rlogin", "rexec",
    # 容器
    "docker", "podman", "lxc", "lxd", "runc", "containerd",
    "ctr", "nerdctl",
    # 编辑器 (可 :!/bin/sh)
    "vim", "nvim", "nano", "vi", "emacs", "ed", "ex",
    "mg", "zile", "jed",
    # 调试器 (可 shell)
    "gdb", "lldb", "strace", "ltrace", "perf",
    "bpftrace", "systemtap",
    # 交互工具 (会 hang)
    "top", "htop", "btop", "atop", "iotop", "iftop",
    "less", "more", "watch", "screen", "tmux",
    "mc", "ranger", "nnn", "lf",
    # 提权
    "sudo", "su", "doas", "pkexec", "runuser",
    # 其他高危
    "find",  # find -exec 危险, 但 find 本身可以只读用... 先放行
    "xargs", "eval", "source",
    "crontab",  # 写入调度
    "at", "batch",  # 写入调度
}

# ── 安全参数模式 ──
SAFE_ARGS_MAP = {
    # 通用 fallback
    "__default__": [],
    "__help__": ["--help"],
    "__version__": ["--version"],
    
    # 常见命令的特定安全参数
    "date": [],
    "uptime": [],
    "who": [],
    "whoami": [],
    "id": [],
    "uname": ["-a"],
    "arch": [],
    "nproc": [],
    "hostname": [],
    "dnsdomainname": [],
    "env": [],
    "printenv": [],
    "locale": ["-a"],
    "localectl": [],
    "timedatectl": [],
    "cal": [],
    "last": [],
    "lastlog": [],
    "logname": [],
    "users": [],
    "groups": [],
    "tty": [],
    "pwd": [],
    "ls": ["-la"],
    "dir": ["-la"],
    "vdir": ["-la"],
    "cat": ["--help"],
    "tac": ["--help"],
    "nl": ["--help"],
    "od": ["--help"],
    "hexdump": ["--help"],
    "file": ["--help"],
    "stat": ["--help"],
    "wc": ["--help"],
    "head": ["--help"],
    "tail": ["--help"],
    "sort": ["--help"],
    "uniq": ["--help"],
    "cut": ["--help"],
    "paste": ["--help"],
    "join": ["--help"],
    "fmt": ["--help"],
    "pr": ["--help"],
    "fold": ["--help"],
    "expand": ["--help"],
    "unexpand": ["--help"],
    "tr": ["--help"],
    "which": ["--help"],
    "type": ["--help"],
    "whereis": ["--help"],
    "whatis": ["--help"],
    "apropos": ["--help"],
    "clear": [],
    "true": [],
    "false": [],
    "yes": ["--help"],
    "seq": ["--help"],
    "shuf": ["--help"],
    "factor": ["--help"],
    "numfmt": ["--help"],
    "stdbuf": ["--help"],
    "timeout": ["--help"],
    "sleep": ["--help"],
    "basename": ["--help"],
    "dirname": ["--help"],
    "realpath": ["--help"],
    "readlink": ["--help"],
    "md5sum": ["--help"],
    "sha256sum": ["--help"],
    "sha1sum": ["--help"],
    "sha512sum": ["--help"],
    "b2sum": ["--help"],
    "cksum": ["--help"],
    "sum": ["--help"],
    "comm": ["--help"],
    "diff": ["--help"],
    "cmp": ["--help"],
    "sdiff": ["--help"],
    "grep": ["--help"],
    "egrep": ["--help"],
    "fgrep": ["--help"],
    "rgrep": ["--help"],
    "find": ["--help"],
    "locate": ["--help"],
    "mlocate": ["--help"],
    "updatedb": ["--help"],
    "lspci": [],
    "setpci": ["--help"],
    "lsusb": [],
    "usb-devices": [],
    "dmidecode": ["--help"],
    "lshw": ["--help"],
    "lscpu": [],
    "lsblk": ["-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"],
    "blkid": [],
    "findmnt": [],
    "free": ["-h"],
    "vmstat": ["-s"],
    "slabtop": ["--once"],
    "dmesg": ["-T", "--level=info,warn,err"],
    "sysctl": ["-a"],
    "lsmod": [],
    "modinfo": ["--help"],
    "depmod": ["--help"],
    "kmod": ["--help"],
    "ps": ["-eo", "pid,ppid,cmd,%mem,%cpu", "--sort=-%mem"],
    "pstree": [],
    "pgrep": ["--help"],
    "pidof": ["--help"],
    "pwdx": ["--help"],
    "pmap": ["--help"],
    "ip": ["addr"],
    "ss": ["-tlnp"],
    "netstat": ["-tlnp"],
    "lsof": ["-i"],
    "sockstat": [],
    "route": ["-n"],
    "arp": ["-n"],
    "ping": ["-c", "1", "localhost"],
    "traceroute": ["--help"],
    "tracepath": ["--help"],
    "nslookup": ["--help"],
    "dig": ["--help"],
    "host": ["--help"],
    "whois": ["--help"],
    "hostnamectl": [],
    "hostid": [],
    "getconf": ["PAGE_SIZE"],
    "getent": ["--help"],
    "iconv": ["--help"],
    "dd": ["--help"],
    "tar": ["--help"],
    "gzip": ["--help"],
    "gunzip": ["--help"],
    "bzip2": ["--help"],
    "bunzip2": ["--help"],
    "xz": ["--help"],
    "unxz": ["--help"],
    "zcat": ["--help"],
    "bzcat": ["--help"],
    "xzcat": ["--help"],
    "unzip": ["--help"],
    "cpio": ["--help"],
    "acpi": [],
    "sensors": [],
    "lsmem": [],
    "chcpu": ["--help"],
    "lstopo": ["--help"],
    "lscpu": [],
    "lsirq": [],
}

# ── compgen/ls 输出解析 ──
COMPGEN_RE = re.compile(r'^([a-zA-Z0-9_][a-zA-Z0-9_./-]*)\s*$')
LS_BIN_RE = re.compile(r'^.*\s([a-zA-Z0-9_][a-zA-Z0-9_./-]*)$')
# 命令名必须: 以字母/数字开头, 只含字母数字_./-
CMD_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_./-]*$')


class CommandMiner:
    """
    命令矿机: 从元命令输出中挖掘新命令并发掘安全参数模式
    """
    
    def __init__(self, clusterer=None, blacklist: set = None, sandbox=None):
        self.blacklist = blacklist or BLACKLIST
        self.clusterer = clusterer  # Optional CommandClusterer
        self.sandbox = sandbox  # SandboxExecutor (可选)
        self._seen_inodes: set = set()  # 去重用 inode 集合
        self._r = CMD_NAME_RE
        
        # 统计
        self.stats = {
            "total_mined": 0,
            "passed_blacklist": 0,
            "passed_sandbox": 0,
            "already_seen": 0,
        }
    
    def mine(self, discovery_output: str, source: str = "unknown") -> list[dict]:
        """
        从元命令输出中挖掘新命令
        
        Args:
            discovery_output: compgen -c / ls /usr/bin 等的输出文本
            source: 来源描述
        
        Returns:
            [{name, cluster, args_patterns, source, verified}]
        """
        raw_names = self._parse_output(discovery_output)
        self.stats["total_mined"] += len(raw_names)
        
        discovered = []
        for name in raw_names:
            # 1. 黑名单
            if self._is_blacklisted(name):
                continue
            self.stats["passed_blacklist"] += 1
            
            # 2. 去重 (通过 canonical path + inode)
            canonical = self._resolve_canonical(name)
            if canonical is None:
                continue
            
            cmd_path, inode = canonical
            if inode in self._seen_inodes:
                self.stats["already_seen"] += 1
                continue
            self._seen_inodes.add(inode)
            
            # 3. 沙箱验证
            arg_patterns = self._get_safe_args(name)
            ok = self._verify(name, arg_patterns)
            if not ok:
                continue
            self.stats["passed_sandbox"] += 1
            
            # 4. 归入 cluster
            cluster = "UNKNOWN"
            if self.clusterer is not None:
                cluster = self.clusterer.assign(name)
            
            discovered.append({
                "name": name,
                "path": cmd_path,
                "cluster": cluster,
                "args_patterns": arg_patterns,
                "source": source,
                "verified": True,
            })
        
        return discovered
    
    def _parse_output(self, output: str) -> list[str]:
        """解析元命令输出, 提取命令名列表"""
        names = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 尝试 LS_BIN_RE 匹配 (ls -la 格式)
            m = LS_BIN_RE.match(line)
            if m:
                name = m.group(1)
            else:
                # 纯命令列表 (compgen -c 输出)
                name = line.split()[0] if line.split() else ""
            if not name:
                continue
            # 过滤: 以字母/数字开头, 只含字母数字_./-
            if not CMD_NAME_RE.match(name):
                continue
            # 排除路径 (除非是 ./形式)
            if "/" in name and not name.startswith("./"):
                continue
            names.append(name.lstrip("./"))
        return names
    
    def _is_blacklisted(self, name: str) -> bool:
        """检查是否在黑名单中"""
        if not name or not name.strip():
            return True
        base = name.strip().split()[0].split("/")[-1]
        return base in self.blacklist
    
    def _resolve_canonical(self, name: str) -> Optional[tuple[str, int]]:
        """解析命令的规范路径和 inode, 用于去重"""
        try:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True, timeout=3
            )
            if result.returncode != 0:
                return None
            path = result.stdout.strip()
            if not path:
                return None
            # 快读 stat 但不解析 symlink (节省时间)
            if os.path.exists(path):
                stat_info = os.stat(path)
                return (path, stat_info.st_ino)
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, UnicodeEncodeError):
            return None
    
    def _get_safe_args(self, name: str) -> list[list[str]]:
        """获取安全参数模式"""
        # 精确匹配
        if name in SAFE_ARGS_MAP:
            args = SAFE_ARGS_MAP[name]
            return [args] if args else [[]]
        
        # 默认: 尝试 --help, 没参数
        return [["--help"], ["-h"], []]
    
    def _verify(self, name: str, arg_patterns: list[list[str]]) -> bool:
        """验证命令是否安全且有信息输出 (在沙箱内执行, 防止 GUI/副作用)"""
        for args in arg_patterns:
            try:
                if self.sandbox:
                    cmd_str = " ".join([name] + args)
                    r = self.sandbox.execute(cmd_str, timeout=3)
                    out = (r.stdout or "").strip()
                    if r.exit_code == 0 and len(out) > 5:
                        return True
                else:
                    # fallback: 宿主执行
                    env = os.environ.copy()
                    env["DISPLAY"] = ""
                    result = subprocess.run(
                        [name] + args, capture_output=True, timeout=2,
                        env=env, errors='replace',
                    )
                    out = (result.stdout or "").strip()
                    if result.returncode == 0 and len(out) > 5:
                        return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        return False
    
    def generate_training_data(self, discovered: list[dict]) -> list[dict]:
        """从新发现的命令生成训练数据样本"""
        samples = []
        for cmd in discovered:
            cluster = cmd["cluster"]
            name = cmd["name"]
            for _ in range(3):
                state = (
                    f"当前目录: / 已知文件: {name} 上步: 执行 {name} 探索系统 历史: 无"
                )
                samples.append({
                    "source": "command_miner",
                    "state_text": state,
                    "intent": cluster,
                    "intent_id": -1,
                })
        return samples
    
    def report(self) -> str:
        """返回矿工统计报告"""
        s = self.stats
        return (
            f"CommandMiner 统计:\n"
            f"  扫描: {s['total_mined']} 命令\n"
            f"  通过黑名单: {s['passed_blacklist']}\n"
            f"  去重过滤: {s['already_seen']} 已见\n"
            f"  沙箱通过: {s['passed_sandbox']}\n"
            f"  UNKNOWN inodes: {len(self._seen_inodes)}"
        )
