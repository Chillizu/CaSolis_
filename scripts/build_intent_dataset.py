"""
数据整合工具 — 合成训练数据集

来源:
  A) folunar_intent_dataset.csv    2400 条 (Kimi 生成)
  B) checkpoints/clean-v3/         1452 条 (历史真实命令, parse_intent 打标)
  C) 混淆样本 (ChatGPT 设计)        自定义生成

输出: data/intent_train.jsonl      ~3500-4000 条训练样本
"""

import json
import csv
import random
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

# ── 配置 ─────────────────────────────────────────────────

CSV_PATH = "Response/folunar_intent_dataset.csv"
CLEAN_V3_PATH = "checkpoints/clean-v3/train_data.jsonl"
OUTPUT_PATH = "data/intent_train.jsonl"

INTENT_MAP = {
    "READ": 0, "LIST": 1, "SEARCH": 2, "INFO": 3,
    "INSPECT": 4, "COUNT": 5, "EXPLORE": 6, "HELP": 7,
}
INTENT_NAMES = list(INTENT_MAP.keys())

# ── A) 加载 CSV ─────────────────────────────────────────

def load_csv_data(path: str) -> list[dict]:
    """加载 Kimi 生成的 CSV 数据"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            intent = row.get("intent", "").upper().strip()
            if intent not in INTENT_MAP:
                continue
            state_text = row.get("state_text", "").strip()
            if not state_text:
                continue
            rows.append({
                "source": "csv",
                "state_text": state_text,
                "intent": intent,
                "intent_id": INTENT_MAP[intent],
            })
    print(f"  CSV: {len(rows)} 条")
    print(f"    分布: {dict(Counter(r['intent'] for r in rows).most_common())}")
    return rows


# ── B) 从 clean-v3 解析意图 ─────────────────────────────

def parse_intent_from_cmd(cmd: str) -> tuple[str | None, dict | None]:
    """从命令字符串解析意图 (简化版, 与 intent_translator.py 一致)"""
    cmd = cmd.strip()
    cmd_lower = cmd.lower()

    # 显式匹配模式
    patterns = [
        (r"^\s*cat\s+", "READ"),
        (r"^\s*ls\s+", "LIST"),
        (r"^\s*grep\s+", "SEARCH"),
        (r"^\s*wc\s+", "COUNT"),
        (r"^\s*head\s+", "READ"),
        (r"^\s*tail\s+", "READ"),
        (r"^\s*which\s+", "INSPECT"),
        (r"^\s*type\s+", "INSPECT"),
        (r"^\s*man\s+", "HELP"),
        (r"--help", "HELP"),
    ]

    for pattern, intent in patterns:
        if re.search(pattern, cmd_lower):
            return intent, {}

    # 系统命令
    sys_cmds = {
        "whoami", "id", "uptime", "date", "uname", "df",
        "free", "ps", "top", "env", "hostname", "dmesg",
        "lscpu", "lsblk", "lsusb", "lspci", "ip",
    }
    base = cmd_lower.split()[0] if cmd_lower.split() else ""
    if base in sys_cmds:
        return "INFO", {}

    return None, None


def load_clean_v3_data(path: str) -> list[dict]:
    """加载 clean-v3, 解析出意图标签"""
    rows = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            cmd = data.get("cmd", "")
            text = data.get("text", "")

            intent, _ = parse_intent_from_cmd(cmd)
            if intent is None or intent not in INTENT_MAP:
                continue

            # 从 [THOUGHT] 和 [OBS] 构建状态文本
            thought = ""
            obs = ""
            if m := re.search(r"\[THOUGHT\](.*?)\[CMD\]", text):
                thought = m.group(1).strip()
            if m := re.search(r"\[OBS\](.*)", text):
                obs = m.group(1)[:200].strip()

            state_parts = []
            state_parts.append(f"当前目录: /")
            state_parts.append(f"命令: {cmd}")
            if thought:
                state_parts.append(f"意图: {thought}")
            if obs:
                state_parts.append(f"输出: {obs[:100]}")

            rows.append({
                "source": "clean-v3",
                "state_text": " ".join(state_parts),
                "intent": intent,
                "intent_id": INTENT_MAP[intent],
            })

    print(f"  clean-v3: {len(rows)}/{total} 条可解析")
    print(f"    分布: {dict(Counter(r['intent'] for r in rows).most_common())}")
    return rows


# ── C) 混淆样本 (ChatGPT 设计) ──────────────────────────

DIRS = ["/", "/etc", "/var/log", "/proc", "/tmp", "/home", "/opt", "/dev", "/sys", "/usr/bin"]
FILES_BY_DIR = {
    "/":       ["hosts", "fstab", "passwd", "crontab"],
    "/etc":    ["passwd", "shadow", "hostname", "hosts", "ssh/sshd_config", 
                "nginx/nginx.conf", "apt/sources.list", "services", "protocols",
                "group", "shells", "environment", "resolv.conf"],
    "/var/log":["syslog", "auth.log", "dmesg", "boot.log", "kern.log",
                "nginx/access.log", "messages"],
    "/proc":   ["cpuinfo", "meminfo", "uptime", "loadavg", "version",
                "cmdline", "stat", "filesystems", "partitions"],
    "/tmp":    ["output.txt", "test.sh", "cache.json", "temp.log", "data.csv"],
    "/home":   ["user/.bashrc", "user/.profile", "user/documents/report.txt"],
    "/opt":    ["app/config.yaml", "app/config.json", "scripts/backup.sh"],
    "/dev":    ["null", "zero", "random", "sda", "tty"],
    "/sys":    ["class/net/eth0/address", "devices/system/cpu/online",
                "kernel/hostname"],
    "/usr/bin":["python3", "git", "node", "curl", "docker"],
}


def generate_confusion_samples(n_per_group: int = 50) -> list[dict]:
    """
    ChatGPT 建议的混淆样本生成器
    
    核心: 同目录+同文件 → 不同目标 → 不同意图
    """
    rows = []

    # 混淆组: 同一文件, 不同意图
    confusion_groups = [
        # (/etc/passwd → READ / SEARCH / COUNT)
        {
            "dir": "/etc",
            "file": "passwd",
            "targets": [
                ("查看 /etc/passwd 的内容", "READ"),
                ("在 /etc/passwd 中找出所有包含 root 的行", "SEARCH"),
                ("统计 /etc/passwd 有多少行", "COUNT"),
                ("从 /etc/passwd 中提取用户名列表", "SEARCH"),
            ],
        },
        # (/var/log/syslog → READ / SEARCH / COUNT)
        {
            "dir": "/var/log",
            "file": "syslog",
            "targets": [
                ("查看 syslog 的内容", "READ"),
                ("在 syslog 中搜索 error 关键词", "SEARCH"),
                ("统计 syslog 的行数", "COUNT"),
                ("过滤 syslog 中的 failed 行", "SEARCH"),
            ],
        },
        # (/proc/cpuinfo → INFO / READ)
        {
            "dir": "/proc",
            "file": "cpuinfo",
            "targets": [
                ("查看 CPU 型号和核心数", "INFO"),
                ("查看 /proc/cpuinfo 的内容", "READ"),
                ("统计 CPU 核心数", "COUNT"),
            ],
        },
        # (/etc/hostname → READ / INFO)
        {
            "dir": "/etc",
            "file": "hostname",
            "targets": [
                ("查看主机名", "READ"),
                ("获取系统主机名信息", "INFO"),
            ],
        },
        # (/tmp/output.txt → READ / COUNT / SEARCH)
        {
            "dir": "/tmp",
            "file": "output.txt",
            "targets": [
                ("查看 output.txt 中有哪些内容", "READ"),
                ("统计 output.txt 的行数", "COUNT"),
                ("在 output.txt 中搜索 error", "SEARCH"),
            ],
        },
        # (/ → LIST / READ / EXPLORE)
        {
            "dir": "/",
            "file": "hosts",
            "targets": [
                ("查看 hosts 文件内容", "READ"),
                ("查看根目录下有什么文件", "LIST"),
                ("看看有没有未探索的目录", "EXPLORE"),
            ],
        },
        # (/usr/bin/git → INSPECT / HELP)
        {
            "dir": "/usr/bin",
            "file": "git",
            "targets": [
                ("检查 git 是否存在", "INSPECT"),
                ("查看 git 的帮助文档", "HELP"),
                ("查看 git 命令版本", "INSPECT"),
            ],
        },
    ]

    for group in confusion_groups:
        for _ in range(n_per_group // len(group["targets"]) + 1):
            for goal, intent in group["targets"]:
                state_text = (
                    f"当前目录: {group['dir']} 已知文件: {group['dir']}/{group['file']} "
                    f"上步: {goal} 历史: 无"
                )
                rows.append({
                    "source": "confusion",
                    "state_text": state_text,
                    "intent": intent,
                    "intent_id": INTENT_MAP[intent],
                })

    # 随机打乱 + 截取
    random.shuffle(rows)

    # 根据 ChatGPT 的要求: 避免"passwd → READ"的死记硬背
    # 确保每类有足够的混淆样本
    print(f"  混淆样本(基础): {len(rows)}")
    print(f"    分布: {dict(Counter(r['intent'] for r in rows).most_common())}")
    return rows


def generate_explore_samples(n: int = 200) -> list[dict]:
    """针对 EXPLORE 和 HELP 的专门样本 (这两个在 CSV 中可能偏少)"""
    rows = []
    explore_goals = [
        "不知道该干嘛，随便转转",
        "当前线索全断了，需要找新方向",
        "看看有没有隐藏的后门或者奇怪的文件",
        "探索一个新的目录看看有什么",
        "看看系统中还有什么有趣的命令",
    ]
    help_goals = [
        "想知道这个命令有哪些选项",
        "看看这个命令怎么用",
        "查看命令的帮助文档",
        "学习新命令的用法",
    ]

    for _ in range(n // 2):
        d = random.choice(DIRS)
        g = random.choice(explore_goals)
        state = f"当前目录: {d} 已知文件: 未知 上步: {g} 历史: ls"
        rows.append({"source": "explore_gen", "state_text": state, "intent": "EXPLORE", "intent_id": 6})

    for _ in range(n // 2):
        d = random.choice(DIRS)
        cmd = random.choice(["git", "docker", "python3", "curl", "find", "grep", "sed", "awk", "tar", "zip"])
        g = random.choice(help_goals)
        state = f"当前目录: {d} 已知文件: {cmd} 上步: 想查阅 {cmd} 文档 历史: {g}"
        rows.append({"source": "help_gen", "state_text": state, "intent": "HELP", "intent_id": 7})

    random.shuffle(rows)
    return rows


# ── 整合 ─────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  训练数据集整合")
    print("=" * 50)

    all_rows = []

    # A) CSV
    print("\n[A] 加载 CSV 数据...")
    csv_rows = load_csv_data(CSV_PATH)
    all_rows.extend(csv_rows)

    # B) clean-v3
    print("\n[B] 解析 clean-v3 数据...")
    v3_rows = load_clean_v3_data(CLEAN_V3_PATH)
    all_rows.extend(v3_rows)

    # C) 混淆样本 (ChatGPT)
    print("\n[C] 生成混淆样本 (同文件不同意图)...")
    confusion_rows = generate_confusion_samples(n_per_group=60)
    all_rows.extend(confusion_rows)

    # D) 补 EXPLORE/HELP
    print("\n[D] 补 EXPLORE/HELP 样本...")
    extra_rows = generate_explore_samples(n=200)
    all_rows.extend(extra_rows)

    # ── 最终处理 ──
    random.shuffle(all_rows)

    # 写入
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 统计
    total = len(all_rows)
    intent_counts = Counter(r["intent"] for r in all_rows)
    source_counts = Counter(r["source"] for r in all_rows)

    print(f"\n{'=' * 50}")
    print(f"  最终数据集: {total} 条")
    print(f"  {'=' * 30}")
    print(f"  [来源分布]")
    for src, cnt in source_counts.most_common():
        print(f"    {src:15s}  {cnt:4d} 条  {cnt/total*100:5.1f}%")
    print(f"  {'=' * 30}")
    print(f"  [意图分布]")
    for intent, cnt in intent_counts.most_common():
        print(f"    {intent:10s}  {cnt:4d} 条  {cnt/total*100:5.1f}%")

    print(f"\n  输出: {OUTPUT_PATH}")
    print(f"  样本示例:")
    for i in range(min(3, len(all_rows))):
        r = all_rows[i]
        print(f"    [{r['intent']}] {r['state_text'][:80]}...")

    return all_rows


if __name__ == "__main__":
    main()
