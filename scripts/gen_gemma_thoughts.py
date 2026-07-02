#!/usr/bin/env python3
"""
Gemma4 协作思考链生成器

用 Ollama 上的 Gemma4 生成真实的"思考→行动→观察→思考→..."链。
每个命令被执行后，Gemma4 看到真实输出，产生下一个想法。

和模板方案的区别：
  - 思考是真实的，不是模板填空
  - 命令选择有上下文（基于前一个输出）
  - 链更自然：像人类探索一样
"""

import os, sys, json, subprocess, random, time, re, urllib.request

DOCKER = "casolis-g4"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    # 制造有趣的测试环境
    dk("echo 'hello world' > /tmp/greet.txt")
    dk("echo 'root:x:0:0:root:/root:/bin/bash' > /tmp/user.txt")
    dk("for i in $(seq 1 5); do echo line_$i > /tmp/f$i.txt; done")
    dk("mkdir -p /tmp/sub/a /tmp/sub/b")
    dk("echo 'port=8080\nhost=localhost\ndebug=true' > /tmp/sub/a/config.cfg")
    dk("echo 'TODO: fix login bug\nTODO: add tests\nDONE: refactor api' > /tmp/todo.txt")

def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:1500]

# ── Ollama API ──
def ask_gemma(prompt: str, max_tokens=150) -> str:
    data = json.dumps({
        "model":"gemma4:e4b","prompt":prompt,"stream":False,
        "options":{"num_predict":max_tokens,"temperature":0.8}
    }).encode()
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate",
                                   data=data, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"]
    except Exception as e:
        print(f"  ⚠️ Gemma4 错误: {e}")
        return ""


def parse_thought(resp: str, prev_cmd: str, prev_out: str) -> tuple[str, str]:
    """从 Gemma4 回复中提取思考内容和下一个命令"""
    thought = ""
    next_cmd = ""
    
    text = resp.strip()
    if not text:
        return f"看看 {prev_cmd} 的结果", "ls"
    
    # 尝试 THOUGHT:/CMD: 格式
    for line in text.split("\n"):
        if "THOUGHT" in line:
            thought = line.split(":", 1)[-1].strip() if ":" in line else text[:100]
        elif "CMD" in line:
            next_cmd = line.split(":", 1)[-1].strip() if ":" in line else ""
    
    if not thought:
        # 去掉命令部分，剩下的就是思考
        # 查找反引号命令 `cmd`
        import re
        cmds_in_text = re.findall(r'[`]\s*([^`]+)\s*[`]', text)
        if cmds_in_text:
            next_cmd = cmds_in_text[0].strip().split()[0]
        # 去掉代码部分作为思考
        thought = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()[:120]
    
    if not thought:
        thought = f"观察 {prev_cmd} 的输出: {prev_out[:60]}..."
    if not next_cmd:
        # 检查回复中是否有看起来像命令的词
        import re
        words = text.split()
        known_cmds = ["ls","cat","pwd","echo","whoami","id","date","find","grep",
                     "head","tail","sort","wc","touch","mkdir","cp","mv","uname","df"]
        for w in words:
            w = w.strip("`'\".,;!?-")
            if w in known_cmds:
                next_cmd = w
                break
    if not next_cmd:
        next_cmd = "ls"
        
    return thought.strip()[:150], next_cmd


def gen_chain(start_cmd="ls", depth=3) -> dict:
    """用 Gemma4 生成思考链"""
    chain = []
    cmd = start_cmd
    full_text = ""
    context = ""

    for step in range(depth):
        out = dk(cmd)
        if not out: out = "(empty)"

        # 构建提示
        prompt = f"""You are exploring a Linux system. Current state:

Context:
{context}
Last command: $ {cmd}
Output: {out[:400]}

What do you think about this output and what would you do next?
First write your thought (starting with THOUGHT:), then the command (starting with CMD:).

Example:
THOUGHT: I see the system has several directories. Let me check what's in /etc.
CMD: ls /etc

Your turn:"""
        
        resp = ask_gemma(prompt)
        thought, next_cmd = parse_thought(resp, cmd, out)

        # 记录
        step_data = f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {out}\n"
        full_text += step_data
        chain.append({"role":"thought","text":thought})
        chain.append({"role":"cmd","text":cmd})
        chain.append({"role":"obs","text":out})

        context += f"$ {cmd}\n{out[:150]}\n"
        if len(context) > 1500:
            context = context[-1000:]

        cmd = next_cmd
        time.sleep(0.5)

    return {"text": full_text, "chain": chain, "depth": depth}


def gen_bootstrap_chains(n=1) -> list[dict]:
    """生成一批 Gemma4 思考链"""
    print(f"\n🧠 Gemma4 协作生成启动...")
    print(f"   模型: gemma4:e4b | 每条 {n} 链\n")

    # 多样化启动命令
    starters = [
        "ls", "pwd", "cat /tmp/greet.txt", "ls /tmp/",
        "cat /etc/hostname", "whoami", "cat /tmp/todo.txt",
        "ls -la /tmp/sub/", "cat /tmp/sub/a/config.cfg",
    ]
    all_chains = []

    for i in range(n):
        start = random.choice(starters)
        depth = random.choice([2, 3, 3, 4])  # 随机深度 2-4

        print(f"\n  [{i+1}/{n}] 启动: {start} (深度 {depth})")
        try:
            chain = gen_chain(start_cmd=start, depth=depth)
            all_chains.append(chain)

            # 打印预览
            print(f"      链长: {len(chain['chain'])} 步")
            for st in chain["chain"][:4]:
                role = st["role"]
                txt = st["text"][:60]
                print(f"      [{role:>7}] {txt}")
            if len(chain["chain"]) > 4:
                print(f"      ... 还有 {len(chain['chain'])-4} 步")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"      ❌ 错误: {e}")
            time.sleep(1)

    return all_chains


if __name__ == "__main__":
    dk_start()
    os.makedirs("data/thoughts", exist_ok=True)

    import sys as _sys
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 else 15

    chains = gen_bootstrap_chains(n)

    # 保存
    with open("data/thoughts/gemma4_chains.jsonl", "w") as f:
        for c in chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"📦 共 {len(chains)} 条 Gemma4 思考链")
    print(f"   保存到 data/thoughts/gemma4_chains.jsonl")

    # 合并到训练集
    with open("data/thoughts/all_with_gemma.jsonl", "w") as f:
        # 先写已有数据
        for lv in [1,2,3,4]:
            lf = f"data/thoughts/lv{lv}.jsonl"
            if os.path.exists(lf):
                for line in open(lf):
                    f.write(line)
        # 再写 Gemma4 数据
        for c in chains:
            d = {"id": f"gemma_{hash(c['text'])%100000:05d}",
                 "text": c["text"], "chain": c["chain"], "level": 0,
                 "source": "gemma4"}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"   合并到 data/thoughts/all_with_gemma.jsonl")
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
