#!/usr/bin/env python3
"""
数据生成器 v2 — 用 Ollama (Gemma4) 生成思考链数据

每个样本:
 [THOUGHT] 我想看看当前目录有什么
 [CMD] ls -la
 [OBS] total 64
 ...

课程分级:
  L1: 只读 (ls, pwd, echo, whoami, id, date, cat, hostname)
  L2: 文件操作 (touch, mkdir, cp, mv, rm, find)
  L3: 文本处理 (grep, sort, wc, head, tail, cut)
  L4: 组合脚本 (管道, 重定向, 复合命令)
"""

import os, sys, json, random, subprocess, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA_MODEL = "gemma4:e4b"
def ollama_gen(prompt: str, max_tokens: int = 120) -> str:
    """调用本地 Ollama 生成文本"""
    data = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "options": {"num_predict": max_tokens}
    }).encode()
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate",
                                   data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["response"]
    except Exception as e:
        print(f"  Ollama API 错误: {e}")
        return ""

DOCKER = "folunar-ds"
def docker_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]
def docker_stop():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)

# ── 课程定义 ──
LEVELS = {
    1: {"name": "read_only", "cmds": [
        "ls", "pwd", "echo hello", "whoami", "id",
        "date", "cat /etc/hostname", "uname -a",
        "ls /", "echo $HOME", "ls -la /tmp",
    ]},
    2: {"name": "file_ops", "cmds": [
        "touch /tmp/test_$RANDOM",
        "mkdir -p /tmp/d$RANDOM",
        "echo data > /tmp/f$RANDOM && cat /tmp/f$RANDOM",
        "ls /tmp/ | wc -l",
        "cp /etc/hostname /tmp/h && cat /tmp/h",
    ]},
    3: {"name": "text_proc", "cmds": [
        "cat /etc/passwd | grep root",
        "ls -la / | head -5",
        "sort /etc/passwd | tail -3",
        "cut -d: -f1 /etc/passwd | sort | head -5",
    ]},
    4: {"name": "combo", "cmds": [
        "for f in /tmp/*; do echo $f; done",
        "find /etc -name '*.conf' -type f 2>/dev/null | head -5",
        "ps aux 2>/dev/null | head -5 || true",
        "df -h 2>/dev/null | head -5 || true",
    ]},
}

def gen_thought_prompt(cmd: str, output: str) -> str:
    """给 Ollama 的提示，让它生成思考链"""
    return f"""You are a curious AI exploring a Linux Docker container. You just ran this command:

$ {cmd}

Output:
{output[:500]}

Now you think about what you see, and decide what to do next.
Write exactly in this format with no extra text:

[THOUGHT] (your thoughts about the output, what you learned, what you're curious about)
[CMD] (next command you want to run)
"""

def parse_thought_response(text: str) -> tuple[str, str]:
    """从 Ollama 回复中提取 THOUGHT 和 CMD"""
    thought = ""
    cmd = ""
    for line in text.strip().split("\n"):
        if line.startswith("[THOUGHT]"): thought = line[len("[THOUGHT]"):].strip()
        if line.startswith("[CMD]"): cmd = line[len("[CMD]"):].strip()
    return thought, cmd

def generate_samples(level: int, n_per_cmd: int = 2) -> list[dict]:
    """为某个课程级别生成思考链样本"""
    level_data = LEVELS[level]
    samples = []

    print(f"\n📚 课程 L{level}: {level_data['name']}")
    print(f"  命令池: {len(level_data['cmds'])} 个")
    print(f"  每个生成 {n_per_cmd} 次\n")

    for cmd in level_data["cmds"]:
        for i in range(n_per_cmd):
            # 执行命令
            output = dk(cmd)

            # 用 Ollama 生成思考
            prompt = gen_thought_prompt(cmd, output)
            try:
                resp = ollama_gen(prompt, max_tokens=120)
                thought, next_cmd = parse_thought_response(resp)
            except Exception as e:
                print(f"  Ollama 出错: {e}")
                thought = f"好奇这条命令的输出"
                next_cmd = random.choice(level_data["cmds"])

            if not thought:
                thought = f"运行 {cmd} 看到了: {output[:50]}..."
            if not next_cmd:
                next_cmd = random.choice(level_data["cmds"])

            # 执行下一命令（如果有）
            next_output = dk(next_cmd) if next_cmd else ""

            sample = {
                "level": level,
                "chain": [
                    {"role": "thought", "text": thought},
                    {"role": "cmd", "text": cmd},
                    {"role": "obs", "text": output},
                    {"role": "thought", "text": f"看到结果后，我决定: {thought}"},
                    {"role": "cmd", "text": next_cmd},
                    {"role": "obs", "text": next_output},
                ],
                "text": f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {output}\n[THOUGHT] 看到结果后，{thought}\n[CMD] {next_cmd}\n[OBS] {next_output}\n",
            }
            samples.append(sample)
            print(f"  [{i+1}/{n_per_cmd}] {cmd:30s} → {thought[:40]}...")
            time.sleep(0.5)  # Ollama 冷却

    return samples


def generate_seed_samples() -> list[dict]:
    """生成一批纯种子样本（无 Ollama，用于冷启动）"""
    samples = []
    templates = [
        ("看看目录里有什么", "ls", "bin  dev  home  ..."),
        ("当前路径", "pwd", "/"),
        ("打印问候", "echo hello", "hello"),
    ]
    for thought, cmd, obs in templates:
        samples.append({
            "level": 1,
            "chain": [
                {"role": "thought", "text": thought},
                {"role": "cmd", "text": cmd},
                {"role": "obs", "text": obs},
            ],
            "text": f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {obs}\n",
        })
    return samples


if __name__ == "__main__":
    docker_start()
    os.makedirs("data/thoughts", exist_ok=True)

    all_samples = []

    # 先加种子
    seeds = generate_seed_samples()
    all_samples.extend(seeds)
    print(f"🌱 种子样本: {len(seeds)}")

    # 逐级生成
    for level in [1, 2, 3, 4]:
        try:
            samples = generate_samples(level, n_per_cmd=2)
            all_samples.extend(samples)
            with open(f"data/thoughts/level{level}.jsonl", "w") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"✅ L{level}: {len(samples)} 样本 → data/thoughts/level{level}.jsonl")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ L{level} 出错: {e}")

    # 合并
    with open("data/thoughts/all.jsonl", "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*40}")
    print(f"📦 总计: {len(all_samples)} 个思考链样本")
    print(f"   data/thoughts/all.jsonl")
    docker_stop()
