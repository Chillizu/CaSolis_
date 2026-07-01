# Phase 2 计划：自举命令宇宙 (v2)

## 核心理念

Agent 通过执行元命令 (`compgen -c`, `ls /usr/bin`, `whatis`, `apropos`)
发现新命令 → 黑名单过滤 → 沙箱试跑验证 → 自动加入 CUSTOM 池 → 探索 → 发现新意图 → 池子继续变大

```
CUSTOM 探索 → 元命令执行 (compgen -c)
                ↓
           CommandMiner: 解析输出 → 黑名单过滤 → 沙箱试跑
                ↓
           CommandClusterer: 按功能聚类
                ↓
           新命令加入 CUSTOM 池 (分层结构)
                ↓
           CUSTOM 探索继续 → 新命令被选中
                ↓
           IntentDiscoverer 发现新模式
                ↓
           生成训练数据 → 重训分类器
                ↓
           Agent 认知空间扩大 → 继续探索 → 回到元命令
```

## 模块设计

### 1. CommandMiner (`agent/command_miner.py`)

从元命令输出中提取新命令名。

```python
class CommandMiner:
    """
    执行发现命令 → 解析输出 → 黑名单过滤 → 沙箱验证 → 返回安全命令列表
    """
    
    # 元命令池 (Agent 通过 CUSTOM 执行这些来发现新命令)
    DISCOVERY_COMMANDS = [
        "compgen -c",       # 所有可执行命令 (bash)
        "ls /usr/bin",      # 二进制文件列表
        "ls /usr/sbin",     # 系统管理命令
        "ls /bin",          # 基础命令
        "ls /usr/local/bin",# 本地安装的命令
    ]
    
    # 静态黑名单 (明显危险/破坏性的直接拒)
    BLACKLIST = {
        # 写入/删除
        "rm", "dd", "mkfs*", "fdisk", "mount", "umount",
        "cp", "mv", "tee", "install", "mktemp", "rename",
        "chmod", "chown", "chattr", "ln",
        # 系统管理
        "systemctl", "systemd-*", "journalctl", "passwd",
        "reboot", "shutdown", "init", "halt", "poweroff",
        # 包管理 (写入系统)
        "pacman", "apt", "yum", "dnf", "dpkg", "rpm", "zypper",
        # 网络操作
        "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
        "ip link", "ip addr add", "route add",
        # 脚本解释器 (可执行任意代码)
        "bash", "sh", "zsh", "fish", "dash", "python*",
        "perl", "ruby", "php", "lua", "tclsh", "expect",
        # 编译
        "gcc", "g++", "cc", "c++", "clang", "make", "cmake", "cargo",
        # 下载/外联
        "curl", "wget", "ftp", "scp", "sftp", "rsync", "nc", "ncat",
        # SSH
        "ssh", "sshd", "telnet", "socat",
        # 容器
        "docker", "podman", "lxc", "runc", "containerd",
        # 编辑器 (可 :!/bin/sh)
        "vim", "nano", "vi", "emacs", "ed", "ex",
        # 调试器
        "gdb", "lldb", "strace", "ltrace",
        # 交互工具 (会 hang)
        "top", "htop", "atop", "less", "more", "watch", "tail -f",
        # 提权
        "sudo", "su", "doas", "pkexec", "login",
        # 其他
        "find -exec", "xargs", "eval", "source", ". ",
    }
    
    def mine(self, discovery_output: str) -> list[dict]:
        """解析 compgen/ls 输出, 返回 [{name, cluster, verified}]"""
    
    def verify_in_sandbox(self, cmd: str) -> bool:
        """在 Docker 沙箱中试跑, 检查 exit_code + 超时 + 输出大小"""
```

### 2. CommandClusterer (`agent/command_clusterer.py`)

将命令按功能聚类，为分层采样做准备。

```python
class CommandClusterer:
    """
    将命令按功能聚类:
    - 基于 man 章节 (1=用户命令, 8=管理命令)
    - 基于命令名前缀/位置
    - whatis 文本关键词匹配
    - MiniLM 嵌入作为补充
    """
    
    SEED_CLUSTERS = {
        "FILESYSTEM_READ":  ["cat", "head", "tail", "tac", "nl", "od", "hexdump", "rev", "cut", "paste", "join", "sort", "uniq"],
        "FILESYSTEM_LIST":  ["ls", "exa", "tree", "find", "locate", "mlocate"],
        "FILESYSTEM_STAT":  ["stat", "du", "df", "file", "wc", "lsblk", "blkid", "findmnt"],
        "PROCESS":          ["ps", "pstree", "pgrep", "pidof"],
        "HARDWARE":         ["lspci", "lsusb", "lscpu", "lsblk", "lshw", "dmidecode"],
        "NETWORK":          ["ss", "ip", "hostname", "nslookup", "dig", "traceroute", "ping"],
        "TIME":             ["date", "uptime", "timedatectl", "cal", "hwclock", "timeout"],
        "ENV":              ["env", "printenv", "locale", "arch", "uname", "hostnamectl", "arch", "nproc"],
        "KERNEL":           ["lsmod", "modinfo", "modprobe -l", "kmod", "dmesg"],
        "RESOURCE":         ["free", "vmstat", "iostat", "mpstat", "sar", "nmon"],
        "USER":             ["who", "w", "whoami", "id", "groups", "users", "finger", "last", "lastlog"],
        "CMD_LOCATE":       ["which", "type", "command -v", "whereis", "apropos", "whatis"],
        "MOUNT":            ["mount", "df", "du", "lsblk", "blkid", "findmnt"],
        "PACKAGE_QUERY":    ["dpkg -l", "dpkg -L", "dpkg -s", "pacman -Q", "pacman -Ql", "rpm -qa"],
    }
    
    def cluster(self, commands: list[str]) -> dict[str, list[str]]:
        """将新命令归入现有 cluster, 或创建新 cluster"""
```

### 3. 分层 CommandSelector (`agent/command_selector_v2.py`)

从 flat softmax 改为两层采样：cluster → command。

```python
class HierarchicalSelector:
    """
    分层命令选择:
    第一层: cluster 选择 (~15-30 个 cluster, UCB/成功率+新颖度)
    第二层: cluster 内命令选择 (~5-20 条, 厌腻感 + softmax)
    args: 从预定义 safe patterns 中选择 (不学)
    """
    
    def select(self) -> tuple[str, list[str]]:
        """返回 (cluster_name, [cmd, arg1, ...])"""
    
    def record_result(self, cluster: str, cmd: str, success: bool, novelty: float):
        """更新统计"""
```

### 4. 集成到 OnlineAgent

```python
# step() 中 CUSTOM 处理扩展
if intent == "CUSTOM":
    cluster, cmd_args = selector.select()
    
    # 如果是元命令, 解析输出发现新命令
    if selector.is_discovery_command(cmd_args):
        output = bash(cmd_args)
        new_cmds = miner.mine(output)
        for cmd in new_cmds:
            cluster_name = clusterer.assign(cmd)
            selector.add_command(cluster_name, cmd)
    
    # 正常执行
    result = bash(cmd_args)
    selector.record_result(cluster, cmd_args, result.success, novelty)
    discoverer.add_trajectory(...)
```

## 文件清单

| 文件 | 功能 |
|------|------|
| `agent/command_miner.py` | 元命令解析 + 黑名单过滤 + 沙箱验证 |
| `agent/command_clusterer.py` | 功能聚类 + seed cluster 管理 |
| `agent/command_selector_v2.py` | 分层采样选择器 |
| `data/command_pool.json` | 持久化命令池 + cluster 映射 |
| `scripts/setup_command_pool.py` | 初始化命令池 |

## 执行顺序

```
Step 1: 写 CommandClusterer + 种子 cluster → 验证聚类效果
Step 2: 写 HierarchicalSelector → 替代 flat CommandSelector
Step 3: 写 CommandMiner → 验证元命令发现新命令
Step 4: 集成到 OnlineAgent → 跑 1000 步验证自举
```

---

## Kimi 评审结论 (第2轮, 排除安全约束)

### 关键结构修正

1. **分层采样不是银弹**：cluster 温度 0.3–0.5（求稳），command 温度 0.8–1.2（给长尾机会）。cluster UCB c=1.0–1.5。**per-cluster baseline 归一化**必须做（不同 cluster 本底成功率不同）。

2. **分类器拆两层**：
   - 外层：IntentClassifier(11类) → 稳定，不关心命令池大小
   - 内层：CUSTOM Policy(bandit/UCB) → cluster → command → args，快变，不怕非稳态

3. **14 clusters 不够**：FILESYSTEM_* 会爆炸到上百条。建议：
   - 至少 40+ 顶层
   - 或保持 14 个但支持动态分裂（>30命令 + embedding内聚度低时自动拆）
   - 必须有 UNKNOWN cluster

4. **Miner 自动聚类只能 bootstrap**：靠命令名前缀 + whatis + man 章节只能处理 obvious 命令。多义词（ip, find）会错。UNKNOWN 必须有探索预算。

5. **核心坑**：
   - args 也是动作的一部分（`ls` vs `ls -R /` 完全不同）
   - 成功定义不能只看 exit_code（`compgen -c` 返回 0 但没解决任务）
   - 去重（aliases, builtins, 同二进制多名字）
   - 新命令冷启动配额
   - 经验回放加动作空间版本 tag
   - 元命令本身在 CUSTOM pool 里（不写死在 miner）

### 修正后的执行顺序

```
Phase 1 (这次的): 
  a. 分类器拆两层: 外层 IntentClassifier(11) + 内层 CustomPolicy(bandit)
  b. CommandClusterer: 14→40 seed clusters + UNKNOWN + 动态分裂准备
  c. HierarchicalSelector: 替换 flat CommandSelector
  d. 跑 1000 步, 看分层采样和 cluster 稳定性

Phase 2 (下一轮):
  a. CommandMiner: 元命令发现 + 自动聚类
  b. 冷启动配额 + args 探索
  c. 自举环完整集成

Phase 3 (更远):
  a. 动态分裂生效
  b. 经验回放版本 tag
  c. 发现奖励 hacking 防制
```
