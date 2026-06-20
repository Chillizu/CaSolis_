"""
生成基准测试格式的训练数据 (扩充版)

目标:  ~1600 条 (每类 ~200 条)
格式: "当前目录: {dir} 已知文件: {file} 上步: {desc} 历史: {history}"
"""

import json
import random

random.seed(42)

# ── 所有可能的填充值 ─────────────────────────────────

DIRS = ["/", "/etc", "/var/log", "/proc", "/tmp", "/home", "/opt", "/dev", "/usr/bin", "/sys"]
FILES = ["passwd", "hostname", "hosts", "syslog", "cpuinfo", "meminfo", "shadow", "group", 
         "fstab", "crontab", "services", "protocols", "shells", "environment", "resolv.conf"]
CMDS = ["python3", "git", "docker", "curl", "node", "gcc", "java", "nginx", "ssh", "vim",
        "ruby", "perl", "make", "cmake", "pandoc", "tmux", "screen", "ffmpeg", "nmap"]


def make_templates():
    templates = []

    # ── READ (200) ──
    for file in FILES:
        templates.append(("READ", f"读取 /etc/{file} 的内容", "/etc", file))
        templates.append(("READ", f"查看 /etc/{file}", "/etc", file))
        templates.append(("READ", f"打开 /etc/{file} 看看", "/etc", file))
        templates.append(("READ", f"显示 /etc/{file} 的所有行", "/etc", file))
        templates.append(("READ", f"阅读 /etc/{file} 文件", "/etc", file))
    # 备用路径
    for d in ["/proc", "/var/log", "/tmp", "/home"]:
        templates.append(("READ", f"查看 {d}/cpuinfo", d, "cpuinfo"))
        templates.append(("READ", f"读取 {d}/meminfo", d, "meminfo"))
        templates.append(("READ", f"阅读 {d}/fstab", d, "fstab"))
    # 查看版本文件
    for ver in ["version", "release", "issue"]:
        templates.append(("READ", f"查看系统 {ver}", "/etc", ver))

    # ── INFO (200) ──
    info_targets = [
        ("CPU", "cpu"), ("内存", "mem"), ("磁盘", "disk"), ("时间", "uptime"),
        ("当前用户", "whoami"), ("系统信息", "uname"), ("内核版本", "uname"),
        ("主机名", "hostname"), ("运行时间", "uptime"), ("CPU 型号", "cpu"),
        ("CPU 核心数", "cpu"), ("磁盘使用情况", "disk"), ("内存总量", "mem"),
        ("根目录磁盘", "disk"), ("内核", "uname"), ("硬件架构", "uname"),
    ]
    for target_name, target in info_targets:
        templates.append(("INFO", f"查看{target_name}", "/", ""))
        templates.append(("INFO", f"获取{target_name}信息", "/", ""))
        templates.append(("INFO", f"查看系统{target_name}", "/", ""))
        templates.append(("INFO", f"了解{target_name}情况", "/", ""))
        templates.append(("INFO", f"看看{target_name}是什么", "/", ""))
    # 组合
    templates.append(("INFO", "查看 CPU 型号和核心数", "/", ""))
    templates.append(("INFO", "查看 CPU 型号和内存总量", "/", ""))
    templates.append(("INFO", "查看主机名和系统运行时间", "/", ""))

    # ── LIST (200) ──
    for d in DIRS:
        dname = d.lstrip("/") or "root"
        templates.append(("LIST", f"列出 {d} 目录下的内容", d, dname))
        templates.append(("LIST", f"看看 {d} 里有什么文件", d, dname))
        templates.append(("LIST", f"查看 {d} 目录中有哪些文件", d, dname))
        templates.append(("LIST", f"列举 {d} 中的所有内容", d, dname))
        templates.append(("LIST", f"展示 {d} 的目录列表", d, dname))
    # 双重路径
    for d in ["/etc/ssh", "/etc/nginx", "/proc/sys", "/dev/disk"]:
        templates.append(("LIST", f"查看 {d} 的内容", d, ""))

    # ── SEARCH (200) ──
    keywords = ["root", "admin", "nobody", "www-data", "error", "failed", "denied",
                 "warning", "localhost", "127.0.0.1", "systemd", "user", "port",
                 "22", "80", "443", "debug"]
    for file in FILES[:8]:  # 用前 8 个文件
        for kw in keywords[:4]:
            templates.append(("SEARCH", f"在 /etc/{file} 中搜索 {kw}", "/etc", file))
            templates.append(("SEARCH", f"找出 /etc/{file} 中包含 {kw} 的行", "/etc", file))
    # 在 syslog 中搜 error
    for kw in ["error", "failed", "warning", "denied", "critical"]:
        templates.append(("SEARCH", f"在 /var/log/syslog 中搜索 {kw}", "/var/log", "syslog"))
        templates.append(("SEARCH", f"过滤 /var/log/syslog 中的 {kw} 行", "/var/log", "syslog"))

    # ── COUNT (200) ──
    for file in FILES:
        templates.append(("COUNT", f"统计 /etc/{file} 的行数", "/etc", file))
        templates.append(("COUNT", f"数一数 /etc/{file} 有几行", "/etc", file))
        templates.append(("COUNT", f"计算 /etc/{file} 有多少行", "/etc", file))
        templates.append(("COUNT", f"看看 /etc/{file} 有多少行内容", "/etc", file))
        templates.append(("COUNT", f"对 /etc/{file} 进行行数计数", "/etc", file))

    # ── INSPECT (200) ──
    for cmd in CMDS:
        templates.append(("INSPECT", f"检查 {cmd} 是否安装", "/usr/bin", cmd))
        templates.append(("INSPECT", f"确认 {cmd} 是否存在", "/usr/bin", cmd))
        templates.append(("INSPECT", f"看看 {cmd} 命令能不能用", "/usr/bin", cmd))
        templates.append(("INSPECT", f"查看 {cmd} 命令的位置", "/usr/bin", cmd))
        templates.append(("INSPECT", f"检查 {cmd} 的安装状态", "/usr/bin", cmd))

    # ── HELP (200) ──
    for cmd in CMDS:
        templates.append(("HELP", f"查看 {cmd} 的帮助", "/usr/bin", cmd))
        templates.append(("HELP", f"了解 {cmd} 的用法", "/usr/bin", cmd))
        templates.append(("HELP", f"查看 {cmd} 有哪些选项", "/usr/bin", cmd))
        templates.append(("HELP", f"学习 {cmd} 的详细用法", "/usr/bin", cmd))
        templates.append(("HELP", f"查阅 {cmd} 的手册", "/usr/bin", cmd))

    # ── EXPLORE (200) ──
    for d in DIRS:
        dname = d.lstrip("/") or "root"
        templates.append(("EXPLORE", f"探索 {d} 目录", d, dname))
        templates.append(("EXPLORE", f"看看 {d} 里有没有有趣的东西", d, dname))
        templates.append(("EXPLORE", f"在 {d} 中寻找有趣的文件", d, dname))
        templates.append(("EXPLORE", f"浏览 {d} 目录结构", d, dname))
        templates.append(("EXPLORE", f"随便看看 {d} 里有什么", d, dname))
        templates.append(("EXPLORE", f"查看 {d} 目录的整体情况", d, dname))

    return templates


def main():
    templates = make_templates()
    print(f"原始模板数: {len(templates)}")

    intents = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP"]
    intent_map = {n: i for i, n in enumerate(intents)}

    from collections import Counter
    raw_dist = Counter(t[0] for t in templates)
    print(f"模板分布: {dict(raw_dist.most_common())}")

    history_opts = ["无", "cd /etc → ls", "ls → cat passwd", "前面用了 grep root", 
                    "刚查了 df -h", "刚看了 /etc/hostname", "检查过 python3 了",
                    "前面执行了 whoami", "刚看了系统信息", "之前用了 cat /etc/passwd"]

    rows = []
    for intent, desc, dir_path, known_file in templates:
        # 无历史版本 (单步)
        state_text = f"当前目录: {dir_path} 已知文件: {known_file} 上步: {desc} 历史: 无"
        rows.append({"source": "benchformat", "state_text": state_text, "intent": intent, "intent_id": intent_map[intent]})

        # 有历史版本 (多步 ~30%)
        if random.random() < 0.3:
            hist = random.choice([h for h in history_opts if h != "无"])
            state_text = f"当前目录: {dir_path} 已知文件: {known_file} 上步: {desc} 历史: {hist}"
            rows.append({"source": "benchformat", "state_text": state_text, "intent": intent, "intent_id": intent_map[intent]})

    random.shuffle(rows)

    # 确保每类 ~200 条 (截取)
    final = []
    per_class = {n: [] for n in intents}
    for r in rows:
        per_class[r["intent"]].append(r)

    target = 200
    for n in intents:
        sampled = per_class[n][:target]
        final.extend(sampled)
        print(f"  {n:10s}: {len(per_class[n]):4d} → {len(sampled):4d} 条")

    random.shuffle(final)

    # 输出
    output_path = "data/intent_benchformat.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n输出: {output_path}")
    print(f"总计: {len(final)} 条")
    print(f"\n前 3 条:")
    for r in final[:3]:
        print(f"  [{r['intent']}] {r['state_text'][:80]}...")

    return final


if __name__ == "__main__":
    main()
