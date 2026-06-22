#!/usr/bin/env python3
"""
高质量数据生成器 v2 — 大规模

策略:
  1. 不限制 `compgen -c`（全量抓，~2500+ 命令）
  2. 每个命令跑 --help / --version → 保证有输出
  3. 安全命令跑随机参数
  4. 管道组合 + 文件搜索
  5. 目标: 2000+ 条经过 Docker 验证的数据
"""

import os, sys, json, subprocess, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTAINER = "folunar_gen_v2"
DOCKER_IMAGE = "ubuntu:22.04"

# 启动容器
subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
subprocess.run(["docker", "run", "-d", "--name", CONTAINER,
                "--rm", DOCKER_IMAGE, "sleep", "infinity"],
               capture_output=True)
subprocess.run(["docker", "exec", CONTAINER,
                "bash", "-c",
                "apt-get update -qq && apt-get install -y -qq "
                "bash-completion coreutils procps util-linux "
                "grep sed awk findutils moreutils man-db "
                "> /dev/null 2>&1"], capture_output=True)
print("🐳 容器就绪", flush=True)

def run(cmd, timeout=10):
    try:
        r = subprocess.run(
            ["docker", "exec", CONTAINER, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except:
        return "", "ERROR", -1

samples = []
seen_cmds = set()
thoughts = [
    "checking", "looking at", "exploring", "testing",
    "viewing", "reading", "inspecting", "querying",
]

# ─── 1. 全量命令抓取 ────────────────────────
print("📋 获取系统命令列表...", flush=True)
out, _, _ = run("compgen -c 2>/dev/null | sort -u | head -500")
all_cmds = [c.strip() for c in out.split("\n") if c.strip() and len(c.strip()) > 1]
print(f"   共 {len(all_cmds)} 个命令", flush=True)

dangerous = {"rm", "dd", "mkfs", "shutdown", "reboot", "halt", "poweroff",
             "kill", "chmod", "chown", "mount", "umount", "passwd", "su",
             "sudo", "apt-get", "dpkg", "adduser", "userdel", "groupadd",
             "apt", "make", "gcc", "cc", "as", "ld"}

# ─── 2. 每个命令跑 help + version ───────────
print("🧪 验证所有命令...", flush=True)
for cmd in all_cmds:
    if cmd in dangerous:
        continue

    # Try --help first (most reliable)
    for flag in ["--help 2>&1 | head -10", "--version 2>&1 | head -5", ""]:
        test_cmd = f"command -v {cmd} && {cmd} {flag}"
        out, _, rc = run(test_cmd, timeout=5)
        if rc == 0 and len(out) >= 5:
            thought = random.choice(thoughts)
            cmd_line = f"{cmd} {flag.split('2>&1')[0].strip()}" if flag else cmd
            text = f"[THOUGHT] {thought} {cmd}\n[CMD] {cmd_line}\n[OBS] {out[:300]}"
            samples.append({"text": text, "cmd": cmd_line, "output": out[:300]})
            seen_cmds.add(cmd)
            break

print(f"   ✅ 基础数据: {len(samples)} 条", flush=True)

# ─── 3. 参数变体（常见的带参数命令）─────────
print("🎲 生成参数变体...", flush=True)
param_map = {
    "ls": ["-la", "-lh /tmp", "/var/log", "/etc", "-R /etc 2>/dev/null | head -10"],
    "cat": ["/etc/hostname", "/etc/os-release", "/proc/version", "/proc/uptime",
            "/proc/loadavg", "/proc/meminfo | head -5"],
    "head": ["-5 /etc/passwd", "-3 /proc/cpuinfo", "-10 /proc/meminfo"],
    "tail": ["-3 /etc/group", "-1 /etc/hostname"],
    "echo": ["hello", "$HOME", "$PATH", "$SHELL", "$USER", "test $(date)"],
    "which": ["ls", "cat", "echo", "grep", "head"],
    "whoami": [""],
    "hostname": [""],
    "uname": ["-a", "-r", "-s", "-n", "-m"],
    "date": ["", "+%s", "+%Y-%m-%d"],
    "id": [""],
    "pwd": [""],
    "groups": [""],
    "uptime": [""],
    "hostid": [""],
    "nproc": [""],
    "arch": [""],
    "df": ["-h /", "-h /tmp", "-h /var"],
    "free": ["-h", "-m"],
    "wc": ["-l /etc/passwd", "-w /etc/hostname"],
    "dmesg": ["| head -5"],
    "env": ["| grep PATH", "| grep HOME", "| head -5"],
    "dirs": [""],
    "users": [""],
    "logname": [""],
    "tty": [""],
    "basename": ["/usr/bin/test"],
    "dirname": ["/usr/bin/test"],
    "seq": ["5", "1 10"],
    "yes": ["test | head -3"],
    "factor": ["42", "100"],
    "nice": ["-n 5 echo test 2>&1 | head -5"],
    "sleep": ["0.1 && echo ok"],
    "time": ["ls 2>&1 | head -3"],
    "find": ["/etc -name '*.conf' -type f 2>/dev/null | head -5"],
    "grep": ["root /etc/passwd", "nobody /etc/passwd"],
    "sort": ["/etc/passwd | head -5"],
    "uniq": ["/etc/passwd | head -5"],
    "cut": ["-d: -f1 /etc/passwd | head -3"],
    "tr": ["a-z A-Z < /etc/hostname 2>/dev/null | head -3"],
    "tee": ["/dev/null < /etc/hostname | head -1"],
    "history": ["| tail -5"],
}

for cmd, params in param_map.items():
    for param in params:
        cmd_line = f"{cmd} {param}".strip()
        out, _, rc = run(cmd_line)
        if rc == 0 and len(out) >= 3:
            thought = random.choice(thoughts)
            text = f"[THOUGHT] {thought} {cmd}\n[CMD] {cmd_line}\n[OBS] {out[:300]}"
            samples.append({"text": text, "cmd": cmd_line, "output": out[:300]})

print(f"   ✅ +参数变体: {len(samples)} 条", flush=True)

# ─── 4. 管道组合 ──────────────────────────
print("🔗 管道组合...", flush=True)
pipes = [
    ("cat /etc/passwd | cut -d: -f1 | head -5", ""),
    ("ps aux 2>/dev/null | grep root | head -5", ""),
    ("ls /bin | head -20", ""),
    ("ls /usr/bin | head -20", ""),
    ("ls /sbin 2>/dev/null | head -10", ""),
    ("find /etc -maxdepth 1 -name '*.conf' -type f 2>/dev/null | head -10", ""),
    ("grep -c 'root' /etc/passwd 2>/dev/null", ""),
    ("cat /proc/cpuinfo | grep processor | wc -l", ""),
    ("ls -la /tmp 2>/dev/null | wc -l", ""),
    ("df -h / | tail -1 | awk '{print $4}'", ""),
    ("uname -a | cut -d' ' -f1,3", ""),
    ("whoami | tr a-z A-Z", ""),
    ("echo $((2 + 2))", ""),
    ("cat /proc/meminfo | grep 'MemTotal' | awk '{print $2}'", ""),
    ("ls /proc | grep -E '^[0-9]+$' | head -10", ""),
]

for cmd_line, _ in pipes:
    out, _, rc = run(cmd_line)
    if rc == 0 and len(out) >= 2:
        thought = random.choice(thoughts)
        text = f"[THOUGHT] {thought}\n[CMD] {cmd_line}\n[OBS] {out[:300]}"
        samples.append({"text": text, "cmd": cmd_line, "output": out[:300]})

print(f"   ✅ +管道组合: {len(samples)} 条", flush=True)

# ─── 5. 安全命令随机参数 ──────────────────
print("🎯 安全命令随机参数...", flush=True)
known_paths = ["/tmp", "/etc", "/proc", "/usr", "/bin",
               "/var/log", "/dev", "/etc/default", "/etc/init.d"]
safe_cmds = [c for c in all_cmds if c not in dangerous and len(c) >= 2]

for cmd in safe_cmds[:80]:
    paths = random.sample(known_paths, min(2, len(known_paths)))
    for path in paths:
        cmd_line = f"{cmd} {path} 2>/dev/null | head -5"
        out, _, rc = run(cmd_line)
        if rc == 0 and len(out) >= 3:
            thought = random.choice(thoughts)
            text = f"[THOUGHT] {thought} {cmd}\n[CMD] {cmd_line}\n[OBS] {out[:300]}"
            samples.append({"text": text, "cmd": cmd_line, "output": out[:300]})

print(f"   ✅ +随机参数: {len(samples)} 条", flush=True)

# ─── 去重 ──────────────────────────────
seen = set()
deduped = []
for s in samples:
    key = s["cmd"]
    if key not in seen:
        seen.add(key)
        deduped.append(s)

# 打乱
random.shuffle(deduped)

print(f"\n📊 最终统计:")
print(f"   总样本: {len(deduped)}")
out_lens = [len(s["output"]) for s in deduped]
print(f"   输出长度: min={min(out_lens)}, max={max(out_lens)}, avg={sum(out_lens)//len(out_lens)}")
print(f"   唯一命令数: {len(set(s['cmd'] for s in deduped))}")

# 保存
import os as _os
_os.makedirs("checkpoints/clean-v2", exist_ok=True)
with open("checkpoints/clean-v2/train_data.jsonl", "w") as f:
    for s in deduped:
        f.write(json.dumps(s) + "\n")

print(f"\n✅ 保存: checkpoints/clean-v2/train_data.jsonl ({len(deduped)} 条)")

# 清理
subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
print("🐳 容器已清理")
