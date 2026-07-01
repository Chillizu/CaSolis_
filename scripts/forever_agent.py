#!/usr/bin/env python3
"""
永续学习者 v2 — 创造倾向版

在好奇心驱动基础上增加"创造"冲动：
  无聊 → 创造新文件/内容 → 环境改变 → 好奇心满足
"""

import os, sys, json, subprocess, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

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
    return (r.stdout or b"").decode("utf-8", errors="replace").strip()[:2000]
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
}

# 创造计数器
CREATE_N = [0]
def make_create_cmd():
    CREATE_N[0] += 1
    n = CREATE_N[0]
    return random.choice([
        f"echo 'created_{n}' > /tmp/c{n}; cat /tmp/c{n}",
        f"printf 'line{n}\\nline{n+1}' > /tmp/l{n}; cat /tmp/l{n}",
        f"mkdir -p /tmp/d{n}; ls /tmp/d{n}",
        f"touch /tmp/f{n}; ls -la /tmp/f{n}",
        f"echo 'msg_{n}' > /tmp/m{n}; cat /tmp/m{n}",
        f"ls /tmp | tail -10",
    ])


class Agent:
    def __init__(self, cp_dir="checkpoints/forever-v3"):
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
        curio = {}
        for cmd in self.known:
            out = self.memory.get(cmd, "")
            ppl = self.curiosity(out[:500]) if out else 10.0
            visits = self.visits.get(cmd, 0)
            curio[cmd] = ppl / (1 + 0.2 * visits)
        avg_c = sum(curio.values()) / max(len(curio), 1)

        if avg_c < 1.0:
            # 😴 无聊 → 50% 创造 / 50% 探索奇怪
            if random.random() < 0.5:
                return make_create_cmd(), "bored-create"
            else:
                return random.choice(CMDS["discover"] + CMDS["weird"]), "bored-explore"
        elif random.random() < 0.2:
            all_c = self.known + CMDS["discover"] + CMDS["weird"]
            return random.choice(all_c), "random"
        else:
            return max(curio, key=curio.get), "curious"

    def learn(self, text):
        texts = [text]
        if len(self.memory) >= 5:
            replay = random.sample(list(self.memory.keys()), min(3, len(self.memory)))
            for c in replay:
                o = self.memory[c]
                if o: texts.append(f"$ {c}\n{o}\n")
        total_l, total_n = 0.0, 0
        for t in texts:
            toks = tok.encode(t).ids[:120]
            if len(toks) < 3: continue
            h = torch.zeros(1, 256)
            for i in range(len(toks)-1):
                h, lm = self.model.step(h, torch.tensor([toks[i]]))
                loss = F.cross_entropy(lm, torch.tensor([toks[i+1]]), reduction="mean")
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                total_l += loss.item(); total_n += 1
                h = h.detach()
        return total_l / max(total_n, 1)

    def cycle(self):
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


def main(cp_dir="checkpoints/forever-v3", save_interval=100):
    agent = Agent(cp_dir)
    print(f"\n{'='*50}")
    print("🚀 永续学习者 v2 — 创造倾向版")
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cp_dir", default="checkpoints/forever-v3")
    p.add_argument("--save_interval", type=int, default=100)
    args = p.parse_args()
    main(cp_dir=args.cp_dir, save_interval=args.save_interval)
