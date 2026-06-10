#!/usr/bin/env python3
"""
永续学习者 — 永远在跑的自主 AI

持续循环: 思考 → 行动 → 观察 → 学习 → 思考 → ...
永远不停止，永远在成长。
"""

import os, sys, json, subprocess, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# ── 模型 ──
tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
VOCAB = tok.get_vocab_size()
BOS_TOKEN = VOCAB + 16
EOS_TOKEN = VOCAB + 17
TOTAL_VOCAB = VOCAB + 19

class ForeverLearner(nn.Module):
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


DOCKER_NAME = "forever-agent"

def docker_start():
    subprocess.run(["docker", "kill", DOCKER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", DOCKER_NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", DOCKER_NAME, "--rm",
                    "ubuntu:22.04", "sleep", "86400"], capture_output=True, timeout=30)

def docker_exec(cmd):
    r = subprocess.run(["docker", "exec", "-i", DOCKER_NAME, "bash", "-c", cmd],
                       capture_output=True, timeout=15)
    out = (r.stdout or b"").decode("utf-8", errors="replace").strip()[:2000]
    return out

def docker_stop():
    subprocess.run(["docker", "kill", DOCKER_NAME], capture_output=True)


# ── 命令宇宙 ──
CMDS = {
    "explore": ["ls", "ls -la", "pwd", "whoami", "hostname", "date", "uptime",
                "echo hello", "uname -a", "id", "df -h /", "free -h",
                "cat /etc/hostname", "ls /tmp"],
    "discover": ["ls /proc | head -10", "ls /dev | head -10", "ls /sys | head -10",
                 "cat /etc/passwd | head -5", "cat /etc/services | head -10",
                 "env | head -10", "ls /etc | head -10", "cat /etc/issue",
                 "cat /etc/os-release | head -5"],
    "weird": ["dmesg 2>/dev/null | tail -3", "ls /nonexistent 2>&1",
              "head -20 /etc/group", "ls -la /bin | head -10",
              "grep root /etc/passwd", "ls /var/log | head -10",
              "find /etc -name '*.conf' 2>/dev/null | head -5"],
    "create": ["echo hello > /tmp/x; cat /tmp/x",
               "mkdir -p /tmp/a; ls /tmp/a",
               "touch /tmp/f; ls -la /tmp/f",
               "printf 'a\\nb\\nc' > /tmp/f; wc -l /tmp/f"],
}


class Agent:
    def __init__(self, cp_dir="checkpoints/forever-v1"):
        self.cp_dir = cp_dir
        os.makedirs(cp_dir, exist_ok=True)
        self.model = ForeverLearner()
        self._load()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        self.h = torch.zeros(1, 256)
        self.total = 0
        self.memory = {}
        self.visits = {}
        self.known = list(CMDS["explore"])
        self.thoughts = []
        self.actions = []
        docker_start()

    def _load(self):
        import glob
        pts = sorted(glob.glob(f"{self.cp_dir}/model-*.pt"))
        if pts:
            self.model.load_state_dict(torch.load(pts[-1], map_location="cpu", weights_only=True))
            print(f"  加载: {pts[-1]}")
        else:
            sd = torch.load("checkpoints/general-v1/model_final.pt", map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd, strict=False)
            print(f"  从通用模型初始化")
        print(f"  参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def save(self):
        torch.save(self.model.state_dict(), f"{self.cp_dir}/model-{self.total}.pt")

    @torch.no_grad()
    def think(self, n=8):
        """生成思考文本"""
        h, t = self.h, BOS_TOKEN
        out = []
        for _ in range(n):
            h, logits = self.model.step(h, torch.tensor([t]))
            lp = F.softmax(logits.squeeze(0) / 0.9, dim=-1)
            t = torch.multinomial(lp, 1).item()
            if t == EOS_TOKEN: break
            if t < VOCAB: out.append(t)
            h = h.detach()
        self.h = h
        return tok.decode(out) if out else "..."

    def curiosity(self, text):
        """计算困惑度作为好奇度"""
        toks = tok.encode(text).ids[:80]
        if len(toks) < 2: return 0.0
        h = torch.zeros(1, 256)
        total = 0.0
        with torch.no_grad():
            for i in range(len(toks)-1):
                h, lm = self.model.step(h, torch.tensor([toks[i]]))
                total += F.cross_entropy(lm, torch.tensor([toks[i+1]]), reduction="mean").item()
        return total / (len(toks)-1)

    def pick_action(self):
        """好奇心驱动选行动"""
        curio = {}
        for cmd in self.known:
            out = self.memory.get(cmd, "")
            ppl = self.curiosity(out[:500]) if out else 10.0
            visits = self.visits.get(cmd, 0)
            curio[cmd] = ppl / (1 + 0.2 * visits)
        avg_c = sum(curio.values()) / max(len(curio), 1)

        if avg_c < 1.0:
            cat = random.choice(["discover", "weird", "create"])
            return random.choice(CMDS[cat]), f"bored-{cat}"
        elif random.random() < 0.2:
            all_c = self.known + CMDS["discover"] + CMDS["weird"] + CMDS["create"]
            return random.choice(all_c), "random"
        else:
            return max(curio, key=curio.get), "curious"

    def learn(self, text):
        """从文本学习"""
        toks = tok.encode(text).ids[:150]
        if len(toks) < 3: return 0.0
        h = torch.zeros(1, 256)
        total, n = 0.0, 0
        for i in range(len(toks)-1):
            h, lm = self.model.step(h, torch.tensor([toks[i]]))
            loss = F.cross_entropy(lm, torch.tensor([toks[i+1]]), reduction="mean")
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()
            total += loss.item(); n += 1
            h = h.detach()
        return total / max(n, 1)

    def cycle(self):
        """一个周期"""
        thought = self.think(8)
        self.thoughts.append(thought)
        cmd, atype = self.pick_action()
        output = docker_exec(cmd)
        self.actions.append(cmd)
        text = f"$ {cmd}\n{output}\n"
        loss = self.learn(text)
        self.memory[cmd] = output
        self.visits[cmd] = self.visits.get(cmd, 0) + 1
        if cmd not in self.known: self.known.append(cmd)
        self.total += 1
        return {"thought": thought, "cmd": cmd, "type": atype, "loss": loss}

    @torch.no_grad()
    def generate(self, n=80, temp=0.7):
        """自由生成"""
        h, t = torch.zeros(1, 256), BOS_TOKEN
        out = []
        for _ in range(n):
            h, lm = self.model.step(h, torch.tensor([t]))
            h = h.detach()
            lp = F.softmax(lm.squeeze(0) / temp, dim=-1)
            t = torch.multinomial(lp, 1).item()
            if t == EOS_TOKEN: break
            if t < VOCAB: out.append(t)
        return tok.decode(out)


def main(cp_dir="checkpoints/forever-v1", save_interval=100):
    agent = Agent(cp_dir)
    print(f"\n{'='*50}")
    print("🚀 永续学习者 — 永远运行")
    print(f"{'='*50}\n")

    try:
        while True:
            r = agent.cycle()
            t = agent.total
            if t % 10 == 0:
                print(f"  [{t:>6}] 🤔 {r['thought'][:25]:>25s} | "
                      f"▶️  {r['cmd'][:25]:<25s} | L:{r['loss']:.2f} | {r['type']}")
            if t % save_interval == 0:
                agent.save()
                gen = agent.generate()
                print(f"\n  💾 保存 | 🧪 生成: {gen[:150]}")
                print(f"  📊 命令: {len(agent.known)} | 交互: {sum(agent.visits.values())}\n")
    except KeyboardInterrupt:
        print(f"\n⏹️  停止")
    finally:
        agent.save()
        docker_stop()
        print(f"\n✅ 总周期: {agent.total} | 命令: {len(agent.known)}")
        gen = agent.generate(n=150)
        print(f"🧪 最终生成: {gen[:200]}")
        with open(f"{cp_dir}/report.txt", "w") as f:
            f.write(f"Total cycles: {agent.total}\nKnown commands: {len(agent.known)}\n")
            f.write(f"Final generation:\n{gen}\n")
            f.write(f"Last 10 thoughts:\n" + "\n".join(agent.thoughts[-10:]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cp_dir", default="checkpoints/forever-v1")
    p.add_argument("--save_interval", type=int, default=100)
    args = p.parse_args()
    main(cp_dir=args.cp_dir, save_interval=args.save_interval)
