#!/usr/bin/env python3
"""生成大规模离线训练数据 — Docker 跑所有命令多次收集输出"""
import subprocess, os, random, json

DATA = []
CMDS = []

# 带参数变化的命令池
for path in ["/", "/tmp", "/etc", "/var", "/home", "/bin", "/usr", "/root", "/dev", "/proc"]:
    CMDS.append(f"ls {path} 2>/dev/null | head -10")
    CMDS.append(f"ls -la {path} 2>/dev/null | head -10")
for flag in ["", "-h", "-b", "-m"]:
    CMDS.append(f"free {flag} 2>/dev/null")
    CMDS.append(f"df {flag} / 2>/dev/null | head -5")
for flag in ["-a", "-r", "-s", "-n", "-v"]:
    CMDS.append(f"uname {flag}")
for u in ["", "root", "daemon", "nobody", "bin"]:
    CMDS.append(f"id {u} 2>/dev/null")
for d in ["/tmp", "/", "/etc", "/var", "/home"]:
    CMDS.append(f"du -sh {d} 2>/dev/null")
    CMDS.append(f"ls -la {d} 2>/dev/null | head -8")
for msg in ["hello", "test", "abc", "123", "hello world", "hi there", "HELLO"]:
    CMDS.append(f"echo {msg}")
CMDS += [
    "pwd", "whoami", "hostname", "who -b", "who -r", "who -d",
    "date", "date -u", "date +%s", "date +%Y-%m-%d",
    "uptime", "uptime -s 2>/dev/null",
    "cat /etc/hostname", "cat /etc/hosts | head -10",
    "cat /etc/os-release 2>/dev/null | head -10",
    "cat /etc/issue 2>/dev/null",
    "dmesg 2>/dev/null | tail -5",
    "ls /var/log 2>/dev/null | head -10",
    "ls /bin | head -15", "ls /usr/bin | head -15",
    "which ls which cat which bash which python3",
    "env | head -10", "locale",
    "ls /sys/class/net 2>/dev/null | head -5",
    "mount | head -10 2>/dev/null",
    "ip addr 2>/dev/null | head -10",
    "lscpu 2>/dev/null | head -10",
    "cat /proc/meminfo 2>/dev/null | head -10",
    "cat /proc/cpuinfo 2>/dev/null | head -10",
    "lsusb 2>/dev/null | head -5",
    "lspci 2>/dev/null | head -5",
]
CMDS = list(set(CMDS))
random.shuffle(CMDS)
print(f"命令池: {len(CMDS)} 条变体")

# 启动容器
subprocess.run(["docker", "kill", "data-gen"], capture_output=True)
subprocess.run(["docker", "rm", "data-gen"], capture_output=True)
subprocess.run(
    ["docker", "run", "-d", "--name", "data-gen", "--rm", "ubuntu:22.04", "sleep", "3600"],
    capture_output=True, timeout=30
)

def run(cmd):
    r = subprocess.run(
        ["docker", "exec", "-i", "data-gen", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=15
    )
    return r.stdout or ""

# 跑 5 轮
for epoch in range(5):
    random.shuffle(CMDS)
    for i, cmd in enumerate(CMDS):
        try:
            out = run(cmd)
            if out and len(out) > 5:
                DATA.append({
                    "cmd": cmd,
                    "output": out.strip()[:2000],
                })
        except:
            pass
    print(f"  轮 {epoch+1}: {len(DATA)} 条")

subprocess.run(["docker", "kill", "data-gen"], capture_output=True)

os.makedirs("data", exist_ok=True)

# 保存 raw
with open("data/offline_raw.jsonl", "w") as f:
    for d in DATA:
        f.write(json.dumps(d) + "\n")

# 保存纯文本（给 tokenizer 用）
all_texts = []
for d in DATA:
    all_texts.append(d["output"])
    # 同时也存 "cmd\noutput" 格式让模型学对应关系
    all_texts.append(f"$ {d['cmd']}\n{d['output']}")

with open("data/offline_corpus.txt", "w") as f:
    for t in all_texts:
        f.write(t + "\n\n")

chars = sum(len(t) for t in all_texts)
unique_cmds = len(set(d["cmd"] for d in DATA))
print(f"\n✅ 数据生成完成")
print(f"   总条目: {len(DATA)}")
print(f"   不同命令: {unique_cmds}")
print(f"   总字符: {chars:,}")
