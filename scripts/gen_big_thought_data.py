#!/usr/bin/env python3
"""
大规模思考链数据生成器

用 Docker 执行大量不同命令，生成 [THOUGHT]/[CMD]/[OBS] 格式数据。
"""

import os, sys, json, subprocess, random, time, re, hashlib

DOCKER = "casolis-ds2"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("echo 'hello world' > /tmp/greet.txt")
    dk("echo 'root:x:0:0:root:/root:/bin/bash' > /tmp/user.txt")
    dk("for i in $(seq 1 5); do echo line_$i > /tmp/f$i.txt; done")
    dk("mkdir -p /tmp/sub/a /tmp/sub/b")
    dk("echo 'config_key=value' > /tmp/sub/a/config.cfg")

def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]

# 思考模板
THOUGHTS = [
    "我想看看{t}是什么情况", "检查一下{t}", "好奇{t}的内容",
    "探索{t}", "看看{t}有没有什么有趣的", "查看{t}的状态",
    "确认{t}的信息", "了解一下{t}", "看看{t}里有什么",
    "检查{t}是否正常", "让我看看{t}", "想确认{t}",
    "看看{t}的情况", "{t}应该有点意思", "先看看{t}吧",
    "好奇{t}长什么样", "查一下{t}的信息", "看看{t}是什么样的",
    "检查一下{t}的状态", "看看{t}有什么", "嗯，{t}应该有点用",
    "{t}里面有什么呢", "先检查{t}", "大概看看{t}",
]

REACTIONS = [
    "明白了", "看到了", "了解了", "有点意思",
    "嗯嗯", "好的", "继续探索", "值得往下看",
    "有意思", "原来如此", "知道啦", "继续",
]

CMD_DESC = {
    "ls":["当前目录","文件列表","目录结构"], "pwd":["当前路径","工作目录"],
    "echo":["输出","显示结果"], "whoami":["当前用户","我是谁"],
    "id":["用户信息","用户ID"], "date":["当前时间","日期"],
    "uname":["系统信息","内核版本"], "cat":["文件内容","读取文件"],
    "touch":["创建文件","新文件"], "mkdir":["新目录","创建目录"],
    "cp":["复制文件","文件复制"], "rm":["删除","移除文件"],
    "grep":["搜索文本","过滤内容"], "sort":["排序","排序输出"],
    "wc":["统计","行数统计"], "head":["前几行","头部内容"],
    "tail":["后几行","尾部内容"], "find":["搜索文件","查找文件"],
    "df":["磁盘空间","磁盘"], "ps":["进程列表","运行进程"],
    "for":["循环","遍历文件"],
}

def target(cmd):
    for k,v in CMD_DESC.items():
        if k in cmd: return random.choice(v)
    return "这条命令"

def gen_thought(cmd):
    return random.choice(THOUGHTS).format(t=target(cmd))

def gen_reaction(out):
    if not out: return "没输出"
    if "error" in out.lower() or "not found" in out.lower(): return "出错了"
    return random.choice(REACTIONS)

# 命令列表（用 shell 变量实现随机化）
CMDS = [
    # L1
    "ls", "ls -la /tmp", "ls /", "pwd",
    "echo hello_world", "echo $(date)", "whoami", "id",
    "date", "uname -a", "cat /tmp/greet.txt", "cat /tmp/user.txt",
    "cat /etc/hostname", "cat /tmp/f$((RANDOM % 5 + 1)).txt",
    "ls /tmp/", "ls -la /tmp/sub/", "cat /tmp/sub/a/config.cfg",
    "echo /tmp/*",
    # L2
    "touch /tmp/new_$(date +%s)",
    "mkdir -p /tmp/dir_$RANDOM",
    "echo test_data_$RANDOM > /tmp/td_$RANDOM.txt",
    "cp /tmp/greet.txt /tmp/greet_copy.txt && cat /tmp/greet_copy.txt",
    "ls /tmp/ | wc -l",
    "rm -f /tmp/rmtest_$RANDOM; ls /tmp/",
    # L3
    "cat /tmp/user.txt | grep root",
    "ls -la / | head -$((RANDOM % 5 + 3))",
    "cat /etc/passwd | sort | tail -$((RANDOM % 3 + 2))",
    "ps aux 2>/dev/null | head -$((RANDOM % 5 + 3)) || echo no_ps",
    "df -h 2>/dev/null | head -$((RANDOM % 3 + 2)) || echo no_df",
    "ls -la /tmp/ | sort",
    # L4
    "for f in /tmp/f*.txt; do echo FILE: $f; cat $f; done",
    "find /tmp -name '*.txt' -type f 2>/dev/null",
    "find /etc -name '*.conf' -type f 2>/dev/null | head -5",
    "true && echo ok || echo fail",
    "cat /tmp/user.txt /tmp/greet.txt 2>/dev/null",
]

def level_of(cmd):
    for i, c in enumerate(CMDS):
        if c == cmd:
            if i < 18: return 1
            if i < 24: return 2
            if i < 30: return 3
            return 4
    return 1

def gen_samples(n=500):
    samples = []
    seen = set()
    for i in range(n):
        cmd = random.choice(CMDS)
        if cmd in seen and random.random() < 0.5:
            continue
        seen.add(cmd)

        out = dk(cmd)
        t1 = gen_thought(cmd)
        r = gen_reaction(out)

        next_cmd = random.choice(CMDS)
        while next_cmd == cmd:
            next_cmd = random.choice(CMDS)
        nout = dk(next_cmd) if random.random() < 0.7 else ""

        text = f"[THOUGHT] {t1}\n[CMD] {cmd}\n[OBS] {out}\n"
        text += f"[THOUGHT] {r} {target(next_cmd)}\n[CMD] {next_cmd}\n[OBS] {nout}\n"

        samples.append({
            "id": hashlib.md5(text.encode()).hexdigest()[:8],
            "text": text, "level": level_of(cmd),
        })

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{n}] {len(samples)} 样本")
    return samples


if __name__ == "__main__":
    dk_start()
    os.makedirs("data/thoughts", exist_ok=True)
    print("🚀 大规模思考链数据\n")

    samples = gen_samples(800)

    by_lv = {}
    for s in samples:
        by_lv.setdefault(s["level"], []).append(s)

    total = 0
    for lv in sorted(by_lv):
        fn = f"data/thoughts/lv{lv}.jsonl"
        with open(fn, "w") as f:
            for s in by_lv[lv]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        n = len(by_lv[lv])
        total += n
        print(f"  ✅ L{lv}: {n} → {fn}")

    with open("data/thoughts/all.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n📦 总计: {total} 样本")
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
