#!/usr/bin/env python3
"""
最终验证 — 通用模型 + 好奇心 + 长期观察

把模型放进 Docker 环境，不打扰它，看它能发展出什么。
"""

import os, sys, json, subprocess, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# ── 加载通用模型 ──
tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
VOCAB = tok.get_vocab_size()
N_ACTION = 16
ACTION_START = VOCAB
BOS_TOKEN = VOCAB + N_ACTION
EOS_TOKEN = VOCAB + N_ACTION + 1
PAD_TOKEN = VOCAB + N_ACTION + 2
TOTAL_VOCAB = VOCAB + N_ACTION + 3

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(TOTAL_VOCAB, 96)
        self.rnn = nn.GRUCell(96, 256)
        self.shared = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())
        self.lm_head = nn.Linear(256, VOCAB)
    def step(self, h, token):
        if token.dim() == 0: token = token.unsqueeze(0)
        h_new = self.rnn(self.token_embed(token), h)
        return h_new, self.lm_head(self.shared(h_new))

def load_model(path="checkpoints/general-v1/model_final.pt"):
    model = BigModel()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model

def perplexity(model, text):
    """模型对文本的困惑度"""
    tokens = tok.encode(text).ids[:100]
    if len(tokens) < 2: return 0.0
    h = torch.zeros(1, 256)
    total = 0.0
    with torch.no_grad():
        for i in range(len(tokens) - 1):
            h, logits = model.step(h, torch.tensor([tokens[i]]))
            total += F.cross_entropy(logits, torch.tensor([tokens[i+1]]), reduction="mean").item()
    return total / (len(tokens) - 1)

def generate(model, prefix="", max_len=60, temp=0.7):
    toks = tok.encode(prefix).ids if prefix else [BOS_TOKEN]
    h = torch.zeros(1, 256)
    with torch.no_grad():
        for t in toks:
            h, _ = model.step(h, torch.tensor([t]))
        out = []; t = toks[-1]
        for _ in range(max_len):
            h, logits = model.step(h, torch.tensor([t])); h = h.detach()
            lp = F.softmax(logits.squeeze(0) / temp, dim=-1)
            t = torch.multinomial(lp, 1).item()
            if t == EOS_TOKEN: break
            if t < VOCAB: out.append(t)
    return tok.decode(out)

# ── Docker ──
DOCKER_NAME = "final-test"
def docker_start():
    subprocess.run(["docker", "kill", DOCKER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", DOCKER_NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", DOCKER_NAME, "--rm", "ubuntu:22.04", "sleep", "3600"], capture_output=True, timeout=30)
def docker_exec(cmd):
    r = subprocess.run(["docker", "exec", "-i", DOCKER_NAME, "bash", "-c", cmd], capture_output=True, timeout=15)
    out = r.stdout.decode("utf-8", errors="replace") if r.stdout else ""
    return out.strip()[:2000]
def docker_stop():
    subprocess.run(["docker", "kill", DOCKER_NAME], capture_output=True)

# ── 命令池 ──
BASIC = [
    "ls", "ls -la", "pwd", "whoami", "hostname", "date",
    "uptime", "echo hello", "uname -a", "id", "df -h /",
    "free -h", "cat /etc/hostname", "who -b", "ls /tmp",
]

WEIRD = [
    "cat /etc/passwd | head -5", "ls /proc | head -10",
    "ls /dev | head -10", "cat /etc/services | head -10",
    "env | head -10", "ls /nonexistent 2>&1",
    "head -20 /etc/group", "ls -la /bin | head -10",
    "grep root /etc/passwd",
    "sl 2>/dev/null; echo not found", "ls /sys | head -10",
    "dmesg 2>/dev/null | tail -5", "cat /etc/issue",
    "cat /etc/os-release | head -5",
]

# ── 训练 ──
def train_step(model, text, opt):
    tokens = tok.encode(text).ids[:150]
    if len(tokens) < 3: return 0.0
    h = torch.zeros(1, 256)
    total = 0.0; n = 0
    for i in range(len(tokens) - 1):
        h, logits = model.step(h, torch.tensor([tokens[i]]))
        loss = F.cross_entropy(logits, torch.tensor([tokens[i+1]]), reduction="mean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item(); n += 1
        h = h.detach()
    return total / max(n, 1)


def run_experiment(cycles=500, log_file="data/final_experiment_log.txt"):
    model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    docker_start()

    known = list(BASIC)
    memory = {}
    visit_counts = {}
    log = []

    with open(log_file, "w") as f:
        f.write(f"最终验证实验 — {cycles} 轮\n{'='*50}\n\n")

    print("🧪 最终验证实验 — 长期观察")
    print(f"   模型: 1.05M params | 词表: {VOCAB}")
    print(f"   轮数: {cycles}\n")

    for cycle in range(cycles):
        # ── 选择行动（好奇心驱动） ──
        curiosities = {}
        for cmd in known:
            history = memory.get(cmd, "")
            ppl = perplexity(model, history[:500]) if history else 5.0
            visits = visit_counts.get(cmd, 0)
            curiosities[cmd] = ppl / (1 + 0.15 * visits)

        avg_c = sum(curiosities.values()) / max(len(curiosities), 1)
        best_cmd = max(curiosities, key=curiosities.get)

        if avg_c < 1.5:
            cmd = random.choice(WEIRD)
            atype = "weird"
        elif random.random() < 0.25:
            cmd = random.choice(WEIRD + known)
            atype = "random"
        else:
            cmd = best_cmd
            atype = "curious"

        # ── 执行 ──
        output = docker_exec(cmd)
        text = f"$ {cmd}\n{output}\n"

        # ── 学习 ──
        loss = train_step(model, text, opt)

        # ── 记录 ──
        memory[cmd] = output
        visit_counts[cmd] = visit_counts.get(cmd, 0) + 1
        if cmd not in known: known.append(cmd)

        if (cycle + 1) % 25 == 0:
            ppl = perplexity(model, output[:500])
            print(f"  [{cycle+1:>4}/{cycles}] {atype:>8} | 损失: {loss:.3f} | 困惑: {ppl:.2f} | {cmd[:30]}")

        if (cycle + 1) % 100 == 0:
            gen_text = generate(model, max_len=60)
            with open(log_file, "a") as f:
                f.write(f"\n--- 轮 {cycle+1} ---\n")
                f.write(f"  命令: {cmd}\n")
                f.write(f"  输出: {output[:200]}\n")
                f.write(f"  生成: {gen_text[:200]}\n")
                f.write(f"  损失: {loss:.4f}\n")
            torch.save(model.state_dict(), f"checkpoints/final-v1/model-{cycle+1}.pt")
            # 好奇度分布快照
            curio_snapshot = {c: curiosities[c] for c in list(curiosities.keys())[:10]}
            log.append(curio_snapshot)

    docker_stop()
    torch.save(model.state_dict(), "checkpoints/final-v1/model_final.pt")

    # ── 最终报告 ──
    print(f"\n✅ {cycles} 轮实验完成")
    print(f"   已知命令: {len(known)}")
    print(f"   总交互: {sum(visit_counts.values())}")
    report = []
    for cmd in ["ls", "date", "whoami", "cat /etc/hostname", "id"]:
        out = memory.get(cmd, "N/A")
        ppl = perplexity(model, out[:500]) if out != "N/A" else 0
        v = visit_counts.get(cmd, 0)
        report.append(f"{cmd:30s} 困惑度:{ppl:.2f} 见过{v}次")
        print(f"   {cmd:30s} 困惑度:{ppl:.2f} 见过{v}次")

    with open(log_file, "a") as f:
        f.write(f"\n\n{'='*50}\n最终报告\n{'='*50}\n")
        f.write(f"已知命令: {len(known)}\n")
        for line in report:
            f.write(line + "\n")

    # ── 最终生成测试 ──
    print("\n🧪 最终生成:")
    for _ in range(3):
        out = generate(model, max_len=80, temp=0.8)
        print(f"  {out[:150]}")
        with open(log_file, "a") as f:
            f.write(f"  生成: {out[:200]}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--output", type=str, default="checkpoints/final-v1")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    run_experiment(cycles=args.cycles)
