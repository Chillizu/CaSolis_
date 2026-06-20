"""
分层 CUSTOM 命令选择器

架构:
  cluster 层: UCB (Upper Confidence Bound) 选择 cluster
  command 层: softmax + 厌腻感 (基于成功率和新颖度)
  args 层: 预定义模式 (Phase 2 再扩展到动态)

参数:
  cluster_temp: 0.3-0.5 (低温度, 让 agent 在熟悉的 cluster 深耕)
  cmd_temp: 0.8-1.2 (高温度, 给长尾命令机会)
  cluster_ucb_c: 1.0-1.5
  n_min: 5 (new command cold-start quota)
  satiation_decay: 0.03/step
  per_cluster_baseline: True (不同 cluster 本底成功率不同)
"""

import json
import os
import random
import math
import re
from collections import defaultdict
from typing import Optional


class HierarchicalSelector:
    """
    两层分层命令选择:
      Level 1: cluster (UCB + temp)
      Level 2: command (softmax + satiation + novelty)
    """
    
    def __init__(
        self,
        cluster_temp: float = 0.4,
        cmd_temp: float = 1.0,
        cluster_ucb_c: float = 1.2,
        n_min: int = 5,
        satiation_penalty: float = 0.85,
        satiation_recovery: float = 0.03,
        novelty_weight: float = 0.3,
        discovery_rate: float = 0.15,  # 每次选 CUSTOM 时, 强制探索元命令的概率
        jitter: float = 0.2,
    ):
        self.cluster_temp = cluster_temp
        self.cmd_temp = cmd_temp
        self.cluster_ucb_c = cluster_ucb_c
        self.n_min = n_min
        self.satiation_penalty = satiation_penalty
        self.satiation_recovery = satiation_recovery
        self.novelty_weight = novelty_weight
        self.discovery_rate = discovery_rate
        self.jitter = jitter
        
        # ── Cluster 级统计 ──
        self.clusters: dict[str, list[str]] = {}  # cluster_name → [cmd1, cmd2, ...]
        self.cluster_stats: dict[str, dict] = {}  # cluster_name → {n, success, total_reward}
        
        # ── Command 级统计 ──
        self.cmd_stats: dict[str, dict] = {}  # cmd → {n, success, total_novelty, satiation}
        
        # ── 发现命令池 (元命令) ──
        self.discovery_commands = {
            "compgen -c": ["compgen", "-c"],
            "ls /usr/bin": ["ls", "/usr/bin"],
            "ls /bin": ["ls", "/bin"],
        }
    
    def register_cluster(self, name: str, commands: list[str]):
        """注册一个 cluster 及其命令"""
        self.clusters[name] = list(commands)
        if name not in self.cluster_stats:
            self.cluster_stats[name] = {"n": 0, "success": 0, "total_reward": 0.0}
        for cmd in commands:
            self._ensure_cmd(cmd)
    
    def _ensure_cmd(self, cmd: str):
        """确保命令的统计条目存在"""
        if cmd not in self.cmd_stats:
            self.cmd_stats[cmd] = {
                "n": 0, "success": 0, "total_novelty": 0.0,
                "satiation": 0.0, "total_reward": 0.0,
            }
    
    def add_command(self, cluster: str, cmd: str):
        """向 cluster 添加新命令"""
        if cluster not in self.clusters:
            self.clusters[cluster] = []
            self.cluster_stats[cluster] = {"n": 0, "success": 0, "total_reward": 0.0}
        if cmd not in self.clusters[cluster]:
            self.clusters[cluster].append(cmd)
        self._ensure_cmd(cmd)
    
    def is_discovery_command(self, cmd_args: list[str]) -> bool:
        """检查是否是元命令 (输出可解析出新命令)"""
        cmd_str = " ".join(cmd_args)
        return cmd_str in self.discovery_commands
    
    def select_discovery(self) -> tuple[str, list[str]]:
        """强制选择一条发现命令 (不走 UCB, 保底探索)"""
        # 只选 self.discovery_commands 里的, 确保 is_discovery_command 匹配
        import random as _r
        key = _r.choice(list(self.discovery_commands.keys()))
        cmd_args = list(self.discovery_commands[key])
        return ("CMD_LIST_ALL", cmd_args)
    
    def select(self) -> tuple[str, list[str]]:
        """
        选择命令
        
        Returns:
            (cluster_name, [cmd, arg1, ...])
        """
        # ── 保底发现: 每次有 discovery_rate 概率强制跑发现命令 ──
        if random.random() < self.discovery_rate:
            cluster, cmd_args = self.select_discovery()
            return cluster, cmd_args
        
        # ── Step 1: 选 cluster (UCB) ──
        cluster = self._select_cluster()
        
        # ── Step 2: 选命令 (softmax + satiation) ──
        cmd = self._select_command(cluster)
        
        # ── Step 3: 生成参数 ──
        args = self._build_args(cmd)
        
        return cluster, [cmd] + args
    
    def _select_cluster(self) -> str:
        """UCB 选择 cluster"""
        active_clusters = [c for c in self.clusters if self.clusters[c]]
        if not active_clusters:
            return list(self.clusters.keys())[0] if self.clusters else "UNKNOWN"
        
        scores = {}
        for c in active_clusters:
            stats = self.cluster_stats[c]
            n = max(stats["n"], 1)
            avg_reward = stats["total_reward"] / n
            
            # UCB bonus
            total_n = sum(s["n"] for s in self.cluster_stats.values())
            ucb = self.cluster_ucb_c * math.sqrt(math.log(max(total_n, 2)) / n)
            
            # Per-cluster baseline 归一化
            # 不同 cluster 本底成功率不同, 归一化到 [0,1]
            success_rate = stats["success"] / max(n, 1) if n > 0 else 0.5
            
            scores[c] = avg_reward + ucb
        
        # Softmax over clusters
        temp = self.cluster_temp
        keys = list(scores.keys())
        vals = [scores[k] for k in keys]
        vals = [v / temp for v in vals]
        max_v = max(vals)
        exp_vals = [math.exp(v - max_v) for v in vals]  # numerical stability
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]
        
        # 采样 (不直接用概率也行, 加一点随机)
        r = random.random()
        cumulative = 0.0
        for k, p in zip(keys, probs):
            cumulative += p
            if r <= cumulative:
                return k
        return keys[-1]
    
    def _select_command(self, cluster: str) -> str:
        """从 cluster 中选择命令 (softmax + satiation)"""
        cmds = self.clusters.get(cluster, [])
        if not cmds:
            return ""
        
        scores = []
        for cmd in cmds:
            stats = self.cmd_stats.get(cmd, {})
            n = stats.get("n", 0)
            success = stats.get("success", 0)
            
            # 冷启动配额: n < n_min 时给先验成功率
            if n < self.n_min:
                base_score = 0.7 + (success / max(n, 1)) * 0.3
            else:
                base_score = success / max(n, 1)
            
            satiation = stats.get("satiation", 0.0)
            
            # 厌腻感: 越久没选恢复越多
            satiation_penalty = 1.0 - satiation * self.satiation_penalty
            
            # 新颖度 bonus
            novelty_bonus = stats.get("total_novelty", 0) / max(n, 1) * self.novelty_weight
            
            # 随机 jitter
            j = 1.0 + (random.random() - 0.5) * self.jitter * 2
            
            score = base_score * satiation_penalty * j + novelty_bonus
            scores.append(score)
        
        # Softmax with temperature
        temp = self.cmd_temp
        vals = [s / temp for s in scores]
        max_v = max(vals)
        exp_vals = [math.exp(v - max_v) for v in vals]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]
        
        r = random.random()
        cumulative = 0.0
        for cmd, p in zip(cmds, probs):
            cumulative += p
            if r <= cumulative:
                return cmd
        return cmds[-1]
    
    def _build_args(self, cmd: str) -> list[str]:
        """生成命令参数"""
        # 简单参数生成, Phase 2 再扩展
        if cmd in ("ls",):
            return ["-la"]
        elif cmd in ("du",):
            return ["-sh", "/"]
        elif cmd in ("df",):
            return ["-h"]
        elif cmd in ("ps",):
            return ["-eo", "pid,ppid,cmd,%mem,%cpu", "--sort=-%mem"]
        elif cmd in ("free",):
            return ["-h"]
        elif cmd in ("uname",):
            return ["-a"]
        elif cmd in ("ip",):
            return ["addr"]
        elif cmd in ("cat",):
            # 选取 /etc 下常见文件
            targets = ["/etc/passwd", "/etc/hostname", "/etc/hosts", "/etc/os-release", "/etc/resolv.conf"]
            return [random.choice(targets)]
        elif cmd in ("dmesg",):
            return ["-T", "--level=info,warn,err"]
        elif cmd in ("systemctl",):
            return ["list-units", "--type=service", "--state=running", "--no-pager"]
        elif cmd in ("journalctl",):
            return ["--no-pager", "--lines=20"]
        elif cmd in ("ping",):
            return ["-c", "1", "localhost"]
        elif cmd in ("traceroute",):
            return ["localhost"]
        elif cmd in ("which", "type", "whereis"):
            return ["bash"]
        elif cmd in ("dmidecode",):
            return ["-t", "system"]
        elif cmd in ("lspci",):
            return []  # no args
        elif cmd in ("lsusb",):
            return []  # no args
        elif cmd in ("lscpu",):
            return []  # no args
        elif cmd in ("lsmod",):
            return []  # no args
        elif cmd in ("whatis",):
            return ["ls"]
        elif cmd in ("date", "uptime", "cal", "who", "whoami", "id", "env", "arch"):
            return []  # no args
        elif cmd in ("locale",):
            return ["-a"]
        elif cmd in ("hostname",):
            return []  # no args
        elif cmd in ("timedatectl",):
            return []  # no args
        elif cmd in ("groups", "users", "logname"):
            return []  # no args
        elif cmd in ("nproc",):
            return []
        elif cmd in ("getconf",):
            return ["PAGE_SIZE"]
        elif cmd in ("vmstat",):
            return ["-s"]  # summary
        elif cmd in ("slabtop",):
            return ["--once"]
        elif cmd in ("ss",):
            return ["-tlnp"]  # tcp listening
        elif cmd in ("netstat",):
            return ["-tlnp"]
        elif cmd in ("lsof",):
            return ["-i"]
        elif cmd in ("host",):
            return ["localhost"]
        elif cmd in ("nslookup",):
            return ["localhost"]
        elif cmd in ("dig",):
            return ["localhost"]
        elif cmd in ("blkid",):
            return []  # no args
        elif cmd in ("findmnt",):
            return []  # no args
        elif cmd in ("stat",):
            return ["/"]
        elif cmd in ("file",):
            return ["/bin/ls"]
        elif cmd in ("wc",):
            return ["-l", "/etc/passwd"]
        elif cmd in ("head", "tail"):
            return ["-5", "/etc/passwd"]
        elif cmd in ("mlocate", "locate"):
            return ["bash"]
        elif cmd in ("crontab",):
            return ["-l"]
        elif cmd in ("last",):
            return []  # no args
        elif cmd in ("lastlog",):
            return []  # no args
        elif cmd in ("getenforce",):
            return []
        elif cmd in ("rpm",):
            return ["-qa"]
        elif cmd in ("dpkg",):
            return ["-l"]
        elif cmd in ("pacman",):
            return ["-Q"]
        elif cmd in ("sysctl",):
            return ["-a"]
        elif cmd in ("depmod",):
            return []  # no args
        elif cmd in ("sar",):
            return ["-u", "1", "1"]  # CPU usage, 1 sample
        elif cmd in ("iostat",):
            return ["1", "1"]
        elif cmd in ("mpstat",):
            return ["1", "1"]
        elif cmd in ("htop",):
            return ["--no-color"]
        elif cmd in ("top",):
            return ["-bn1"]
        elif cmd in ("tty",):
            return []
        elif cmd in ("readlink",):
            return ["-f", "/bin/sh"]
        elif cmd in ("realpath",):
            return ["/"]
        elif cmd in ("namei",):
            return ["/bin/ls"]
        elif cmd in ("hostnamectl",):
            return []  # no args
        elif cmd in ("localectl",):
            return []  # no args
        elif cmd in ("rhash",):
            return ["--help"]
        elif cmd in ("reset",):
            return []
        elif cmd in ("run-parts", "--list"):
            return ["--list", "/etc/cron.daily"]
        elif cmd in ("lsblk",):
            return ["-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"]
        elif cmd in ("fdisk",):
            return ["-l"]
        elif cmd in ("parted",):
            return ["-l"]
        elif cmd in ("sfdisk",):
            return ["-l"]
        else:
            return []  # no args, let it run bare
    
    def record_result(
        self,
        cluster: str,
        cmd: str,
        success: bool,
        novelty: float = 0.0,
        reward: float = 0.0,
    ):
        """记录执行结果"""
        # Cluster 统计
        cs = self.cluster_stats.setdefault(cluster, {"n": 0, "success": 0, "total_reward": 0.0})
        cs["n"] += 1
        if success:
            cs["success"] += 1
        cs["total_reward"] += reward
        
        # Command 统计
        self._ensure_cmd(cmd)
        s = self.cmd_stats[cmd]
        s["n"] += 1
        if success:
            s["success"] += 1
        s["total_novelty"] += novelty
        s["total_reward"] += reward
        
        # 厌腻感衰减 (被选了就增加, 然后整体恢复)
        for c in self.cmd_stats:
            # 整体恢复 (每步固定减, clamp 到 0)
            self.cmd_stats[c]["satiation"] = max(0.0, self.cmd_stats[c]["satiation"] - self.satiation_recovery)
        # 被选中的命令增加 satiation
        s["satiation"] = min(1.0, s["satiation"] + self.satiation_penalty * (1.0 - s["satiation"]))
    
    def get_cluster_distribution(self) -> dict[str, int]:
        """返回每个 cluster 的选择次数"""
        return {c: s["n"] for c, s in self.cluster_stats.items() if s["n"] > 0}
    
    def get_command_distribution(self, cluster: str) -> dict[str, int]:
        """返回 cluster 内命令的选择次数"""
        dist = {}
        for cmd in self.clusters.get(cluster, []):
            n = self.cmd_stats.get(cmd, {}).get("n", 0)
            if n > 0:
                dist[cmd] = n
        return dist
    
    def save(self, path: str):
        """持久化"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "clusters": self.clusters,
            "cluster_stats": self.cluster_stats,
            "cmd_stats": self.cmd_stats,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """加载"""
        with open(path) as f:
            data = json.load(f)
        self.clusters = data["clusters"]
        self.cluster_stats = data["cluster_stats"]
        self.cmd_stats = data["cmd_stats"]
