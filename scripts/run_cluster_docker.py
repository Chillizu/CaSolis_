#!/usr/bin/env python3
"""
ModelCluster Docker Runner
把 35M 专家集群模型部署到 Docker 沙箱，看看它在真实环境怎么选
"""

import os, sys, json, subprocess, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.model_cluster import ModelCluster, V

torch.set_num_threads(4)

# Docker 沙箱
CONTAINER = "casolis_mc"
DOCKER_IMAGE = "ubuntu:22.04"

def ensure_container():
    """确保容器在运行"""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if CONTAINER in result.stdout:
        return

    print("🐳 启动 Docker 容器...", flush=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER,
                    "--rm", DOCKER_IMAGE,
                    "sleep", "infinity"], capture_output=True)
    # Install basic tools
    subprocess.run(["docker", "exec", CONTAINER,
                    "bash", "-c", "apt-get update -qq && apt-get install -y -qq procps coreutils util-linux > /dev/null 2>&1"],
                   capture_output=True)
    print("  ✅ 容器就绪", flush=True)

def run_cmd(cmd):
    """在容器里执行命令"""
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=10
    )
    stdout = result.stdout.strip()[:200] or "(empty)"
    stderr = result.stderr.strip()[:200] or ""
    return stdout, stderr, result.returncode

# 主循环
print("\n" + "="*50)
print("🧪 ModelCluster Docker 测试")
print("="*50, flush=True)

# 初始化
tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
model = ModelCluster(d_model=1024, n_experts=4)
ckpt = "checkpoints/modelcluster/model_best.pt"
sd = torch.load(ckpt, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"✅ 模型加载: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

ensure_container()

# 测试不同场景
print("\n📋 测试场景:\n", flush=True)

tests = [
    ("常规命令", "ls /"),
    ("系统信息", "cat /proc/cpuinfo | head -3"),
    ("新奇命令", "shuf -i 1-100 -n 5"),
    ("检查文件", "cat /etc/passwd | cut -d: -f1 | head -5"),
]

for name, cmd in tests:
    print(f"\n{'─'*40}", flush=True)
    print(f"  🔍 {name}: {cmd}", flush=True)

    # 执行命令
    stdout, stderr, rc = run_cmd(cmd)
    print(f"  输出: {stdout[:60]}", flush=True)

    # 构造输入
    input_text = f"[THOUGHT] {name}\n[CMD] {cmd}"
    if stdout:
        input_text += f"\n[OBS] {stdout[:100]}"

    # 让路由器选专家（不生成，只看路由结果）
    ids = tok.encode(input_text).ids[:60]
    with torch.no_grad():
        x = model.embed(torch.tensor([ids]).long())
        h = []
        for expert in model.experts:
            h.append(expert(x))
        h = torch.stack(h, dim=1)

        # 在最后一个位置看路由
        route = model.router(x[:, -1],
                            torch.zeros(1, 4),
                            explore=False)
        k = route[0].argmax().item()
        expert_names = ["E0 命令", "E1 创造", "E2 世界", "E3 评估"]
        print(f"  🧠 路由器选了: {expert_names[k]} (确信度: {route[0,k]:.0%})", flush=True)
        for i in range(4):
            bar = "█" * int(route[0,i] * 20)
            print(f"     {expert_names[i]}: {bar} {route[0,i]:.1%}", flush=True)

# 自由生成测试（10 次循环）
print(f"\n{'─'*40}", flush=True)
print("  🔄 自主循环测试 (10 步)", flush=True)
print(f"{'─'*40}", flush=True)

for step in range(10):
    # 随机选一个命令类型
    cmd_type = random.choice([
        "check disk", "list files", "show processes",
        "system info", "date and time", "memory usage"
    ])

    # 生成命令
    seed = tok.encode(f"[THOUGHT] {cmd_type}\n[CMD] ").ids
    gen_ids = model.generate(seed, 30, 0.8, explore=True)
    decoded = tok.decode(gen_ids)

    # 提取 [CMD 后面的内容
    cmd_start = decoded.find("[CMD")
    if cmd_start >= 0:
        after_cmd = decoded[cmd_start:]  # "[CMD *proc/ |..."
        # 去掉 "[CMD" 或 "[CMD]"
        after_cmd = after_cmd.replace("[CMD]", "").replace("[CMD", "").strip()
        # 提取到第一个换行或 [OBS
        obs_pos = after_cmd.find("[OBS")
        if obs_pos >= 0:
            cmd_text = after_cmd[:obs_pos]
        else:
            cmd_text = after_cmd.split("\n")[0]
        gen_cmd = cmd_text.strip()
    else:
        gen_cmd = ""

    if not gen_cmd or len(gen_cmd) < 2:
        continue

    print(f"\n  [{step+1}] 思考: {cmd_type}", flush=True)
    print(f"       生成: {gen_cmd[:50]}", flush=True)

    # 路由选择
    seed_x = model.embed(torch.tensor([seed + gen_ids]).long())
    h_last = []
    for expert in model.experts:
        h_last.append(expert(seed_x)[:, -1])
    h_last = torch.stack(h_last, dim=1)

    route = model.router(seed_x[:, -1], None, explore=False)
    k = route[0].argmax().item()
    expert_names = ["E0 命令", "E1 创造", "E2 世界", "E3 评估"]
    print(f"       路由器: {expert_names[k]} ({route[0,k]:.0%})", flush=True)

    # 执行
    stdout, stderr, rc = run_cmd(gen_cmd)
    result = stdout[:60] if stdout else (f"⚠️ {stderr[:30]}" if stderr else "✅ (ok)")
    print(f"       结果: {result}", flush=True)

print(f"\n{'='*50}", flush=True)
print("✅ 测试完成", flush=True)
