"""
CommandClusterer: 将 Linux 命令按信息类型聚类

用于分层采样:
  外层 IntentClassifier 决定 "要不要走 CUSTOM"
  内层 CustomPolicy: cluster → command → args
  
聚类依据:
  1. 命令名 whatis 输出文本
  2. man 章节
  3. 包的分类
  4. 实际 state_text 关键词分布 (运行时积累)
"""

import json, os, re, subprocess
from typing import Optional
from collections import defaultdict


# ── 种子 cluster: 按 Agent 能发现的信息类型划分 ──────────────
# 
# 不是 Linux 命令分类法, 而是 "Agent 执行这个命令能得到什么信息"
#
# 命名规则: 信息类型_子类型
#   例如 FILE_READ  → Agent 读取文件内容
#        FILE_LIST  → Agent 列出目录
#        FILE_STAT  → Agent 获取文件元信息

SEED_CLUSTERS: dict[str, list[str]] = {
    # ── 文件系统 (File System) ──
    "FILE_READ": [
        "cat", "head", "tail", "tac", "nl", "od", "hexdump", "rev",
        "cut", "paste", "join", "sort", "uniq", "fmt", "pr", "fold",
        "expand", "unexpand", "tr", "sed", "awk",
    ],
    "FILE_LIST": [
        "ls", "exa", "tree", "dir", "vdir",
    ],
    "FILE_FIND": [
        "find", "locate", "mlocate", "updatedb", "grep",
        "egrep", "fgrep", "rgrep", "rg",
    ],
    "FILE_STAT": [
        "stat", "file", "wc", "namei", "readlink", "realpath",
        "basename", "dirname",
    ],
    "FILE_DIFF": [
        "diff", "cmp", "comm", "sdiff", "diff3", "patch",
    ],
    "FILE_CHECKSUM": [
        "md5sum", "sha256sum", "sha1sum", "sha512sum",
        "cksum", "sum", "b2sum", "xxhsum",
    ],
    
    # ── 存储 (Storage) ──
    "STORAGE_USAGE": [
        "du", "df", "df", "du -sh", "df -h", "du -h",
    ],
    "STORAGE_DEVICE": [
        "lsblk", "blkid", "findmnt", "mount", "df",
        "fdisk -l", "parted -l", "sfdisk -l",
    ],
    
    # ── 进程 (Process) ──
    "PROCESS_LIST": [
        "ps", "pstree", "pgrep", "pidof", "pwdx",
    ],
    "PROCESS_MEM": [
        "pmap", "smem", "memstat",
    ],
    
    # ── CPU ──
    "CPU_INFO": [
        "lscpu", "nproc", "getconf", "cpuid", "arch",
    ],
    
    # ── 内存 (Memory) ──
    "MEMORY_INFO": [
        "free", "vmstat", "slabtop",
    ],
    
    # ── 硬件 (Hardware) ──
    "HW_PCI": [
        "lspci", "setpci",
    ],
    "HW_USB": [
        "lsusb", "usb-devices",
    ],
    "HW_DMI": [
        "dmidecode", "lshw",
    ],
    
    # ── 网络 (Network) ──
    "NET_LINK": [
        "ip", "ip link", "ip addr", "ip route",
        "ifconfig", "route", "arp",
    ],
    "NET_SOCKET": [
        "ss", "netstat", "lsof -i", "sockstat", "fuser",
    ],
    "NET_DIAG": [
        "ping", "traceroute", "tracepath", "mtr",
        "nslookup", "dig", "host", "whois", "hostname",
    ],
    
    # ── 内核 (Kernel) ──
    "KERNEL_MODULES": [
        "lsmod", "modinfo", "depmod",
    ],
    "KERNEL_DMESG": [
        "dmesg",
    ],
    "KERNEL_SYSCTL": [
        "sysctl",
    ],
    
    # ── 环境 (Environment) ──
    "ENV_VARS": [
        "env", "printenv", "export", "declare", "set",
    ],
    "SYS_INFO": [
        "uname", "hostnamectl", "hostname", "domainname",
        "machine-info", "arch",
    ],
    "LOCALE_INFO": [
        "locale", "localectl", "localedef",
    ],
    "TIME_INFO": [
        "date", "timedatectl", "uptime", "cal", "timeout",
        "hwclock", "ntpq", "chronyc",
    ],
    
    # ── 用户 (User) ──
    "USER_LIST": [
        "who", "w", "users", "logname", "last", "lastlog",
        "utmpdump", "finger",
    ],
    "USER_ID": [
        "whoami", "id", "groups",
    ],
    
    # ── 命令发现 (Command Discovery) ──
    "CMD_LOCATE": [
        "which", "type", "command", "whereis",
    ],
    "CMD_DESC": [
        "whatis", "apropos", "man", "man -k", "man -f",
        "help",
    ],
    "CMD_LIST_ALL": [
        "compgen -c", "ls /usr/bin", "ls /bin",
        "ls /usr/sbin", "ls /usr/local/bin",
    ],
    
    # ── 包管理查询 (read-only) ──
    "PACKAGE_DEB": [
        "dpkg -l", "dpkg -L", "dpkg -s", "dpkg-query",
    ],
    "PACKAGE_RPM": [
        "rpm -qa", "rpm -ql", "rpm -qi",
    ],
    "PACKAGE_ARCH": [
        "pacman -Q", "pacman -Ql", "pacman -Qi", "pacman -Qo",
    ],
    
    # ── 调度器 (Scheduler) ──
    "SCHEDULER": [
        "crontab -l", "at -l", "systemctl list-timers",
    ],
    
    # ── 安全 (Security) ──
    "SECURITY_STATUS": [
        "getenforce", "sestatus", "aa-status",
        "apparmor_status",
    ],
    
    # ── 伪文件系统 (Procfs) ──
    "PROCFS": [
        "cat /proc/*", "ls /proc", "cat /proc/cpuinfo",
        "cat /proc/meminfo", "cat /proc/version",
        "cat /proc/uptime", "cat /proc/loadavg",
        "cat /proc/stat", "cat /proc/devices",
        "cat /proc/filesystems", "cat /proc/partitions",
    ],
    
    # ── 系统日志 (System Log) ──
    "SYSLOG": [
        "dmesg", "cat /var/log/*", "logname",
    ],
    
    # ── 系统资源 (Resource) ──
    "RESOURCE_MON": [
        "free", "vmstat", "iostat", "mpstat", "sar", "nmon",
        "top -bn1", "htop --no-color",
    ],
}


class CommandClusterer:
    """
    命令聚类器
    
    能力:
    - 将新命令自动归入最匹配的 seed cluster
    - 跟踪每个 cluster 的命令数, 为动态分裂做准备
    - UNKNOWN cluster 兜底
    """
    
    def __init__(self):
        self.clusters: dict[str, list[str]] = {k: list(v) for k, v in SEED_CLUSTERS.items()}
        self.clusters["UNKNOWN"] = []
        self._build_index()
        
    def _build_index(self):
        """构建命令→cluster 查找索引"""
        self.cmd_to_cluster: dict[str, str] = {}
        for cluster_name, cmds in self.clusters.items():
            for cmd in cmds:
                base = cmd.split()[0]  # "dpkg -l" → "dpkg"
                self.cmd_to_cluster[base] = cluster_name
                
    def assign(self, cmd_name: str) -> str:
        """
        将命令名归入 cluster
        
        策略:
        1. 精确匹配命令名 → 归入对应 cluster
        2. whatis 输出关键词匹配 → 归入最匹配 cluster
        3. 以上都不行 → UNKNOWN
        """
        # 精确匹配
        base = cmd_name.split()[0]
        if base in self.cmd_to_cluster:
            return self.cmd_to_cluster[base]
        
        # whatis 匹配
        try:
            desc = subprocess.run(
                ["whatis", base], capture_output=True, text=True, timeout=5
            ).stdout.lower()
            
            whatis_keywords = {
                "FILE_READ":     ["display", "output", "file content", "concatenate", "text"],
                "FILE_LIST":     ["directory", "list", "directory contents"],
                "FILE_FIND":     ["search", "find", "pattern", "grep"],
                "FILE_STAT":     ["file status", "file type", "stat"],
                "PROCESS_LIST":  ["process", "process status", "process tree"],
                "HW_PCI":        ["pci", "pci devices", "bus"],
                "HW_USB":        ["usb", "usb devices", "universal serial bus"],
                "NET_LINK":      ["network", "ip", "interface", "route", "address"],
                "NET_SOCKET":    ["socket", "network", "connection", "port"],
                "KERNEL_MODULES":["kernel module", "module"],
                "MEMORY_INFO":   ["memory", "free", "memory usage"],
                "TIME_INFO":     ["time", "date", "clock", "calendar"],
                "USER_LIST":     ["user", "login", "who", "logged"],
                "CMD_DESC":      ["manual", "whatis", "describe", "help"],
                "PACKAGE_*":     ["package", "deb", "rpm", "dpkg", "pacman"],
            }
            
            best_cluster = "UNKNOWN"
            best_score = 0
            for cluster_name, keywords in whatis_keywords.items():
                score = sum(1 for kw in keywords if kw in desc)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_name
            
            if best_score > 0:
                self._add_to_cluster(best_cluster, cmd_name)
                return best_cluster
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # 命令名前缀启发式
        prefix_map = {
            "ls": "FILE_LIST", "find": "FILE_FIND", "grep": "FILE_FIND",
            "ps": "PROCESS_LIST", "ip": "NET_LINK", "ss": "NET_SOCKET",
            "dpkg": "PACKAGE_DEB", "rpm": "PACKAGE_RPM", "pacman": "PACKAGE_ARCH",
            "ls": "HW_", "dmidecode": "HW_DMI",
        }
        for prefix, cluster_base in prefix_map.items():
            if base.startswith(prefix):
                matched = [c for c in self.clusters if c.startswith(cluster_base)]
                if matched:
                    self._add_to_cluster(matched[0], cmd_name)
                    return matched[0]
        
        # 回退: 按 man 章节
        try:
            man_section = subprocess.run(
                ["man", "-w", base], capture_output=True, text=True, timeout=3
            ).stdout.strip()
            # man section → cluster 映射
            if "man1" in man_section:
                self._add_to_cluster("CMD_DESC", cmd_name)
                return "CMD_DESC"
        except:
            pass
        
        self._add_to_cluster("UNKNOWN", cmd_name)
        return "UNKNOWN"
    
    def _add_to_cluster(self, cluster: str, cmd: str):
        """向 cluster 添加命令并更新索引"""
        if cmd not in self.clusters.get(cluster, []):
            self.clusters.setdefault(cluster, []).append(cmd)
            base = cmd.split()[0]
            self.cmd_to_cluster[base] = cluster
    
    def get_clusters(self) -> list[str]:
        """返回所有 cluster 名 (排除空的)"""
        return [k for k, v in self.clusters.items() if v]
    
    def cluster_size(self, cluster: str) -> int:
        """返回 cluster 内命令数"""
        return len(self.clusters.get(cluster, []))
    
    def needs_split(self, cluster: str, max_size: int = 30) -> bool:
        """检查 cluster 是否需要分裂"""
        return self.cluster_size(cluster) > max_size
    
    def save(self, path: str):
        """持久化"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.clusters, f, indent=2)
    
    def load(self, path: str):
        """加载"""
        with open(path) as f:
            self.clusters = json.load(f)
        self._build_index()
