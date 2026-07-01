#!/usr/bin/env python3
"""
双通道自主代理 — 隐藏状态驱动决策 + 文本通道叙述

架构:
  embed → GRU(1024) → 双头
     ↓               ├── lm_head: 文本生成（自述"想法"）
     │               └── world_head: 预测命令输出（世界模型）
  
流程:
  1. 看到命令输出 → 更新隐藏状态 h ← 这就是"思考"
  2. 从 h 生成"想法"文本 ← 让人看懂
  3. 从 h + 环境状态 决定下一个命令
  4. 执行命令 → 看输出 → 用世界模型损失训练 h
"""

import os, sys, json, subprocess, random, time, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

class DualChannel(nn.Module):
    """隐藏状态（思考）+ 文本叙述 + 世界模型"""

    def __init__(self, embed=256, hidden=1024):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(TV, embed)
        self.rnn = nn.GRUCell(embed, hidden)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        # 头1: 文本生成（自言自語）
        self.lm_head = nn.Linear(hidden, V)
        # 头2: 世界模型（预测输出令牌）
        self.world_head = nn.Linear(hidden, V)

    def think(self, h, obs_tokens):
        """看到输出 → 更新隐藏状态（这就是"思考"）"""
        with torch.no_grad():
            for t in obs_tokens:
                h = self.rnn(self.embed(torch.tensor([t])), h)
        return h.detach()

    def speak(self, h, temp=0.85, max_n=30):
        """从隐藏状态生成"自述"文本"""
        out = []
        with torch.no_grad():
            for _ in range(max_n):
                pv = out[-1] if out else BOS
                h2 = self.rnn(self.embed(torch.tensor([pv])), h)
                lm = self.lm_head(self.shared(h2))
                lp = F.softmax(lm.squeeze(0)/temp, dim=-1)
                nt = torch.multinomial(lp, 1).item()
                if nt >= V: break
                out.append(nt)
        return tok.decode(out) if out else "(静默)"

    def predict_output(self, h, cmd_embed):
        """世界模型：从 h+命令 预测输出"""
        h_cmd = self.rnn(cmd_embed, h)
        return self.world_head(self.shared(h_cmd))

    def curiosity(self, h, obs_tokens):
        """好奇心 = 困惑度：预测输出 vs 实际输出"""
        if len(obs_tokens) < 2: return 0.0
        nll = 0.0; h2 = h.clone()
        with torch.no_grad():
            for i in range(len(obs_tokens)-1):
                h2 = self.rnn(self.embed(torch.tensor([obs_tokens[i]])), h2)
                pred = self.world_head(self.shared(h2))
                nll += F.cross_entropy(pred, torch.tensor([obs_tokens[i+1]])).item()
        return math.exp(nll / (len(obs_tokens)-1))


DOCKER = "dual-agent"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)

def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]


def init_from_v8(model):
    """从 v8 7.6M 模型加载权重"""
    try:
        old = torch.load("checkpoints/curiosity-v8/model_final.pt", map_location="cpu", weights_only=True)
        own = model.state_dict()
        # 复制匹配的层
        for k in ["embed.weight","rnn.weight_ih","rnn.weight_hh","rnn.bias_ih","rnn.bias_hh",
                   "shared.0.weight","shared.0.bias","shared.1.weight","shared.1.bias",
                   "lm_head.weight","lm_head.bias"]:
            if k in old and k in own:
                if old[k].shape == own[k].shape:
                    own[k] = old[k]
                elif old[k].ndim == own[k].ndim:
                    # 取交集维度
                    mins = [min(a,b) for a,b in zip(old[k].shape, own[k].shape)]
                    if own[k].ndim == 1:
                        own[k][:mins[0]] = old[k][:mins[0]]
                    elif own[k].ndim == 2:
                        own[k][:mins[0], :mins[1]] = old[k][:mins[0], :mins[1]]
        # world_head 从 lm_head 初始化
        own["world_head.weight"] = own["lm_head.weight"].clone()
        own["world_head.bias"] = own["lm_head.bias"].clone()
        model.load_state_dict(own)
        print("   ✅ 从 v8 迁移权重")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   ⚠️ 迁移失败: {e}")


class DualAgent:
    """双通道自主代理"""

    def __init__(self):
        self.model = DualChannel()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.h = torch.zeros(1, 1024)
        self.total = 0
        self.cmds_tried = set()
        dk_start()
        print(f"🧠 DualChannel: {sum(p.numel() for p in self.model.parameters()):,} params")

    def train_world_model(self, obs_tokens):
        """训练世界模型：从隐藏状态预测输出令牌"""
        if len(obs_tokens) < 3: return 0.0
        h = self.h.clone()
        loss = 0.0; n = 0
        for i in range(len(obs_tokens)-1):
            h = self.model.rnn(self.model.embed(torch.tensor([obs_tokens[i]])), h)
            pred = self.model.world_head(self.model.shared(h))
            l = F.cross_entropy(pred, torch.tensor([obs_tokens[i+1]]))
            loss += l; n += 1
        loss /= n
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.opt.step()
        return loss.item()

    def cycle(self):
        """一个完整的思考→行动→观察周期"""
        # ── 1. 自述"想法"（从当前隐藏状态生成文本）──
        thought = self.model.speak(self.h, temp=0.85, max_n=25)

        # ── 2. 选择命令 ──
        cmd_pool = ["ls","pwd","whoami","echo hello","date","id",
                    "ls /tmp/","cat /etc/hostname","uname -a"]
        # 好奇心驱动：尝试没做过的命令
        novel = [c for c in cmd_pool if c not in self.cmds_tried]
        if novel and random.random() < 0.4:
            cmd = random.choice(novel)
        else:
            # 否则随机选
            cmd = random.choice(cmd_pool)
        self.cmds_tried.add(cmd)

        # ── 3. 执行命令 ──
        out = dk(cmd)
        if not out: out = "(empty)"

        # ── 4. 思考（更新隐藏状态）──
        obs_tokens = tok.encode(out[:200]).ids[:50]
        self.h = self.model.think(self.h, obs_tokens)

        # ── 5. 训练世界模型 ──
        wm_loss = self.train_world_model(obs_tokens)

        # ── 6. 计算好奇心 ──
        cur = self.model.curiosity(self.h, obs_tokens)

        self.total += 1
        return {"thought": thought, "cmd": cmd, "output": out[:60],
                "wm_loss": wm_loss, "curiosity": cur}


def main(steps=100):
    agent = DualAgent()
    print(f"\n{'='*50}")
    print("🧠 双通道代理 — 隐藏状态思考 + 文本叙述")
    print(f"{'='*50}\n")

    for i in range(steps):
        r = agent.cycle()
        if i % 5 == 0:
            status = f"🧠 {r['thought'][:40]}"
            print(f"  [{i:>3}] {status}")
            print(f"        ▶️  {r['cmd'][:20]} | 好奇心:{r['curiosity']:.1f} | WM:{r['wm_loss']:.2f}")

    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    print(f"\n✅ {agent.total} 周期 | 尝试命令: {len(agent.cmds_tried)}")
    print(f"   最后隐藏状态: {agent.h.norm().item():.2f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    main(steps=args.steps)
