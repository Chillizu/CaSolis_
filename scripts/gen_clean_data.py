#!/usr/bin/env python3
"""
Generate clean [THOUGHT][CMD][OBS] training data
Uses infinite dynamic commands + real Docker output.
No model-generated garbage.
"""

import os, sys, json, subprocess, random, time, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn

DOCKER = "mamba-clean"
def dk(cmd):
    r = subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],
                       capture_output=True, timeout=15)
    return (r.stdout or b"").decode("utf-8", errors="replace").strip()[:1500]

def start_docker():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("echo $RANDOM > /tmp/secret.txt")
    dk("echo $((RANDOM * 65536)) > /tmp/rnd_start.txt")
    dk("mkdir -p /tmp/a/x /tmp/a/y /tmp/b/z")
    for i in range(1, 11):
        dk(f"echo line_{i}_$(date +%N) > /tmp/f{i}.txt")


def random_cmd():
    R = random
    n = R.randint(1, 9999)
    m = R.randint(1, 256)
    ch = R.choice("abcdefghijklmnopqrstuvwxyz")
    ch2 = R.choice("0123456789abcdef")
    charclass = R.choice(["[a-z]", "[0-9]", "[a-f0-9]", "[a-m]", "[n-z]"])
    depth = R.randint(1, 5)

    cmds = ["ls","cat","echo","head","tr","date","whoami","pwd","id",
            "uname","seq","ps","free","wc","sort","grep","find","shuf",
            "printf","cp","mv","rm","mkdir","touch","chmod","od","env",
            "cut","tee","basename","dirname","xargs","fmt","fold","nl",
            "which","type","whatis"]
    rcmd = R.choice(cmds)

    templates = [
        f"date +%s.%N",
        f"cat /proc/uptime",
        f"echo $(( {n} * {m} + {R.randint(0,999)} ))",
        f"echo $(( RANDOM * {n} * {m} ))",
        f"cat /proc/loadavg",
        f"free | head -3",
        f"ps aux 2>/dev/null | wc -l",
        f"cat /proc/meminfo | grep -E '^Mem'",
        f"tr -dc '{charclass}' </dev/urandom | head -c{n}",
        f"head -c{n} /dev/urandom | od -An -tx1 | tr -d ' '",
        f"wc -c <(head -c{n} /dev/urandom) 2>/dev/null",
        f"cat /proc/stat | head -{n % 5 + 2}",
        f"cat /proc/cpuinfo | head -{n % 10 + 5}",
        f"echo $RANDOM > /tmp/v3_{n}.txt && cat /tmp/v3_{n}.txt",
        f"ls /tmp/v3_* 2>/dev/null | wc -l",
        f"ls -la /tmp/ | head -{R.randint(3, 15)}",
        f"find /tmp -name '*{ch}*' -type f 2>/dev/null | head -5",
        f"find / -maxdepth {depth} -name '*.txt' 2>/dev/null | head -{m}",
        f"grep -r '{ch}' /tmp/ 2>/dev/null | head -5",
        f"cat /tmp/f{R.randint(1, 30)}.txt",
        f"echo $(( {n} + {m} * {R.randint(100, 999)} ))",
        f"seq {R.randint(1, 50)} {R.randint(51, 100)} | shuf | head -{R.randint(3, 10)}",
        f"printf '%x\\n' $RANDOM$RANDOM",
        f"for i in 1 2 3; do echo $((RANDOM * {n})); done | sort -n",
        # meta-learning
        f"{rcmd} --help 2>&1 | head -10",
        f"man {rcmd} 2>&1 | head -10",
        f"which {rcmd}",
        f"type {rcmd} 2>/dev/null || echo not_found",
        f"command -v {rcmd}",
        f"whatis {rcmd} 2>/dev/null || echo no_entry",
    ]

    if R.random() < 0.10:
        templates += [
            f"ls /usr/bin/ | shuf | head -{m % 20 + 5}",
            f"cat /etc/shells 2>/dev/null",
            f"help 2>&1 | head -15",
        ]

    return R.choice(templates)


def output_entropy(text):
    """粗略多样性度量：不同字符数 / 总字符数"""
    if not text or len(text) < 3:
        return 0.0
    return len(set(text)) / max(len(text), 1)


if __name__ == "__main__":
    start_docker()
    data = []
    seen_outputs = set()

    print("Generating clean training data...\n")
    start = time.time()

    while len(data) < 1500:
        cmd = random_cmd()
        out = dk(cmd) or "(empty)"
        out_short = out[:1000]

        # 跳过空输出
        if not out or out == "(empty)":
            continue

        # 跳过几乎相同的输出（去重）
        sig = out_short[:60]
        if sig in seen_outputs:
            continue
        seen_outputs.add(sig)

        # 计算多样性分数，过滤掉太无聊的
        div = output_entropy(out)
        if div < 0.2 and len(data) > 100:
            continue  # 太单一的输出跳过

        # 构建样本
        text = f"[THOUGHT] checking output of: {cmd}\n[CMD] {cmd}\n[OBS] {out}\n"
        data.append({
            "text": text,
            "cmd": cmd,
            "output_len": len(out),
            "diversity": round(div, 3)
        })

        if len(data) % 200 == 0:
            elapsed = time.time() - start
            rate = len(data) / elapsed
            print(f"  [{len(data):>4}/1500] {rate:.0f} samples/sec | last: {cmd[:30]}... ({len(out)}b, div={div:.2f})")

    os.makedirs("checkpoints/clean-v1", exist_ok=True)
    path = "checkpoints/clean-v1/train_data.jsonl"
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    avg_div = sum(d["diversity"] for d in data) / len(data)
    avg_len = sum(d["output_len"] for d in data) / len(data)
    dur = time.time() - start
    print(f"\n✅ {len(data)} samples saved to {path}")
    print(f"   avg output length: {avg_len:.0f} chars")
    print(f"   avg diversity: {avg_div:.2f}")
    print(f"   took {dur:.0f}s ({len(data)/dur:.0f} samples/sec)")

    subprocess.run(["docker","kill",DOCKER],capture_output=True)
