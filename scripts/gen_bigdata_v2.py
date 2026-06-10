#!/usr/bin/env python3
"""生成大规模离线训练数据 v2 — 5000+ 条"""
import subprocess, os, random, json, sys

name = "data-gen-v2"
subprocess.run(["docker", "kill", name], capture_output=True)
subprocess.run(["docker", "rm", name], capture_output=True)
subprocess.run(["docker", "run", "-d", "--name", name, "--rm", "ubuntu:22.04", "sleep", "600"], capture_output=True, timeout=30)

def run(cmd):
    r = subprocess.run(["docker", "exec", "-i", name, "bash", "-c", cmd], capture_output=True, text=True, timeout=15)
    return r.stdout or ""

# 超丰富命令池
all_cmds = []
for base in ["ls", "ls -la"]:
    for p in ["/", "/tmp", "/etc", "/var", "/home", "/bin", "/usr", "/boot", "/dev", "/proc", "/sys", "/root", "/run"]:
        all_cmds.append(f"{base} {p} 2>/dev/null | head -10")
for flag in ["-h", "-b", "-m", "-k"]:
    for cmd in [f"free {flag}", f"df {flag} /", f"df {flag} /tmp"]:
        all_cmds.append(f"{cmd} 2>/dev/null | head -5")
for flag in ["-a", "-r", "-s", "-n", "-v", "-o", "-m"]:
    all_cmds.append(f"uname {flag}")
for u in ["", "root", "daemon", "nobody", "bin", "sys", "www-data", "mail"]:
    all_cmds.append(f"id {u} 2>/dev/null")
    all_cmds.append(f"grep ^{u} /etc/passwd 2>/dev/null")
for d in ["/", "/tmp", "/etc", "/var", "/home", "/bin", "/usr", "/boot"]:
    all_cmds.append(f"du -sh {d} 2>/dev/null")
    all_cmds.append(f"ls -la {d} 2>/dev/null | head -10")
for msg in ["hello", "test", "123", "hi", "world", "foo", "bar", "abc", "the quick brown fox"]:
    all_cmds.append(f"echo {msg}")
for file in ["/etc/hostname", "/etc/hosts", "/etc/issue", "/etc/os-release", "/etc/passwd", "/etc/group", "/etc/mtab"]:
    all_cmds.append(f"cat {file} 2>/dev/null | head -10")
    all_cmds.append(f"wc {file} 2>/dev/null")
all_cmds.extend(["pwd", "whoami", "hostname", "date", "date -u", "date +%s", "uptime",
    "who", "who -b", "who -r", "who -d", "ls /tmp", "ls -la /tmp", "dmesg 2>/dev/null | tail -5",
    "df -h /", "df -h /tmp", "df -h /etc", "free -h", "free -m",
    "locale", "cat /etc/timezone 2>/dev/null"])
all_cmds.extend([
    "ls /proc | head -20", "ls /sys | head -10", "ls /dev | head -20",
    "head -20 /etc/services", "head -20 /etc/protocols 2>/dev/null",
    "find /etc -maxdepth 1 -type f 2>/dev/null | head -10",
    "find /etc -name '*.conf' 2>/dev/null | head -10",
    "ls /bin | head -20", "ls /usr/bin | head -20",
    "which ls which cat which bash which sh which grep",
    "env | head -15", "seq 1 5",
])
all_cmds = list(set(all_cmds))
random.shuffle(all_cmds)
print(f"命令池: {len(all_cmds)} 条")

data = []
for epoch in range(3):
    random.shuffle(all_cmds)
    for cmd in all_cmds:
        out = run(cmd)
        if out and len(out) > 3:
            data.append({"cmd": cmd, "output": out.strip()[:2000]})
    print(f"  轮 {epoch+1}: {len(data)} 条")

subprocess.run(["docker", "kill", name], capture_output=True)

os.makedirs("data", exist_ok=True)
with open("data/offline_v2.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")

all_texts = []
for d in data:
    all_texts.append(f"$ {d['cmd']}\n{d['output']}\n")
with open("data/offline_v2_corpus.txt", "w") as f:
    for t in all_texts:
        f.write(t + "\n")

chars = sum(len(t) for t in all_texts)
unique_cmds = len(set(d["cmd"] for d in data))
print(f"\n✅ 完成! {len(data)} 条, {unique_cmds} 不同命令, {chars:,} 字符")
