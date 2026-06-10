#!/usr/bin/env python3
"""
微调文本通道 — 让模型学会"自述"自己的隐藏状态

训练方法:
  1. 跑若干轮 dual_agent, 记录每个 (隐藏状态, 命令, 输出)
  2. 用模板生成"自述"目标文本:
     "我看到了... (输出摘要), 感到好奇心=X, 所以决定执行Y"
  3. 训练 lm_head: 给定隐藏状态 h → 生成自述文本
  4. 这样 h 编码了环境信息, 文本通道负责说出来
"""

import os, sys, json, subprocess, random, time, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

# ── 同 dual_agent.py 的架构 ──
class DualChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 256)
        self.rnn = nn.GRUCell(256, 1024)
        self.shared = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 1024), nn.GELU())
        self.lm_head = nn.Linear(1024, V)
        self.world_head = nn.Linear(1024, V)

    def think(self, h, obs_ids):
        with torch.no_grad():
            for t in obs_ids:
                h = self.rnn(self.embed(torch.tensor([t])), h)
        return h.detach()


def load_model(pt):
    m = DualChannel()
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    ms = m.state_dict()
    # 加载匹配的键
    for k in ms:
        if k in sd:
            ms[k] = sd[k]
    # world_head 从 lm_head 初始化
    ms["world_head.weight"] = ms["lm_head.weight"].clone()
    ms["world_head.bias"] = ms["lm_head.bias"].clone()
    m.load_state_dict(ms)
    return m


# ── 数据收集：在 Docker 中运行，记录 h + 命令 + 输出 ──
DOCKER = "dual-train"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]


def gen_narration(cmd: str, output: str, curiosity: float) -> str:
    """生成自述文本：描述看到了什么、好奇心如何、决定做什么"""
    # 输出摘要
    summary = output[:40].replace('\n', ' ')
    if not summary: summary = "空"
    
    # 好奇心描述
    if curiosity < 3:
        cur_desc = "感觉平淡"
    elif curiosity < 8:
        cur_desc = f"有点好奇({curiosity:.1f})"
    else:
        cur_desc = f"很感兴趣!({curiosity:.1f})"
    
    templates = [
        f"看到了 {summary}... {cur_desc}。执行 {cmd}。",
        f"观察到 {summary}。{cur_desc} 试试 {cmd}。",
        f"环境: {summary}。{cur_desc} 运行 {cmd}。",
    ]
    return random.choice(templates)


def collect_data(model, n_episodes=30):
    """收集 (隐藏状态, 命令, 输出) 三元组"""
    dk_start()
    data = []
    h = torch.zeros(1, 1024)
    cmds = ["ls","pwd","whoami","echo hello","date","id",
            "ls /tmp/","cat /etc/hostname","uname -a"]
    tried = set()

    print(f"📡 收集 {n_episodes} 个探索样本...")

    for i in range(n_episodes):
        # 好奇/随机选命令
        novel = [c for c in cmds if c not in tried]
        cmd = random.choice(novel) if novel else random.choice(cmds)
        tried.add(cmd)

        out = dk(cmd) or "(empty)"
        obs_ids = tok.encode(out[:200]).ids[:50]

        # 保存隐藏状态备份（更新前）
        h_before = h.clone()

        # 更新隐藏状态
        h = model.think(h, obs_ids)

        # 计算好奇心（更新后）
        if len(obs_ids) >= 2:
            nll = 0.0; h2 = h.clone()
            with torch.no_grad():
                for j in range(len(obs_ids)-1):
                    h2 = model.rnn(model.embed(torch.tensor([obs_ids[j]])), h2)
                    pred = model.world_head(model.shared(h2))
                    nll += F.cross_entropy(pred, torch.tensor([obs_ids[j+1]])).item()
            cur = math.exp(nll / (len(obs_ids)-1))
        else:
            cur = 0

        # 生成自述
        narration = gen_narration(cmd, out, cur)

        data.append({
            "h": h_before.squeeze(0).tolist(),  # 更新前的状态
            "cmd": cmd,
            "output": out[:100],
            "narration": narration,
            "curiosity": cur,
        })

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n_episodes}] 好奇心:{cur:.1f} | {narration[:50]}...")

    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    print(f"  ✅ 收集 {len(data)} 样本\n")
    return data


def train_narrator(model, data, epochs=20):
    """训练 lm_head: h → 自述文本"""
    opt = torch.optim.AdamW(model.lm_head.parameters(), lr=5e-4)
    
    print(f"🎯 微调文本通道 ({epochs} 轮)...")
    for ep in range(epochs):
        random.shuffle(data)
        total_loss = 0; n = 0
        model.train()

        for d in data:
            # h → narration tokens
            h = torch.tensor([d["h"]])
            n_ids = tok.encode(d["narration"]).ids[:30]
            if len(n_ids) < 3: continue

            loss = 0.0
            h2 = h.clone()
            for i in range(len(n_ids)-1):
                h2 = model.rnn(model.embed(torch.tensor([n_ids[i]])), h2)
                pred = model.lm_head(model.shared(h2))
                loss += F.cross_entropy(pred, torch.tensor([n_ids[i+1]]))
            
            loss /= (len(n_ids)-1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lm_head.parameters(), 0.5)
            opt.step()
            total_loss += loss.item(); n += 1

        if (ep+1)%5==0 or ep==0:
            print(f"  ep{ep+1:>2}/{epochs} | Loss:{total_loss/n:.3f}")

    return model


def test_narration(model):
    """测试自述效果"""
    print("\n🧪 测试自述:")
    test_cases = [
        ("看到目录列表: bin boot dev etc home", 0.5),
        ("文件内容是: hello world", 2.3),
        ("错误: 找不到文件", 9.8),
    ]
    model.eval()
    for obs, cur in test_cases:
        # 构建一个"假设的"隐藏状态（从测试观察出发）
        h = torch.zeros(1, 1024)
        obs_ids = tok.encode(obs).ids[:20]
        for t in obs_ids:
            h = model.rnn(model.embed(torch.tensor([t])), h)
        
        # 生成自述
        with torch.no_grad():
            out = []
            for _ in range(25):
                pv = out[-1] if out else BOS
                h2 = model.rnn(model.embed(torch.tensor([pv])), h)
                lm = model.lm_head(model.shared(h2))
                lp = F.softmax(lm.squeeze(0)/0.7, dim=-1)
                nt = torch.multinomial(lp, 1).item()
                if nt >= V: break
                out.append(nt)
        text = tok.decode(out)
        print(f"  📥 obs: {obs[:30]}...")
        print(f"  📤 自述: {text[:80]}\n")


if __name__ == "__main__":
    os.makedirs("checkpoints/dual-agent", exist_ok=True)

    # 加载模型
    model = load_model("checkpoints/curiosity-v8/model_final.pt")
    print(f"🧠 DualChannel: {sum(p.numel() for p in model.parameters()):,} params\n")

    # 收集数据
    data = collect_data(model, n_episodes=30)
    
    # 保存数据
    with open("data/thoughts/narration_data.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"💾 数据保存到 data/thoughts/narration_data.jsonl\n")

    # 微调文本通道
    model = train_narrator(model, data, epochs=30)

    # 保存微调后的模型
    torch.save(model.state_dict(), "checkpoints/dual-agent/model_final.pt")
    print(f"\n💾 保存到 checkpoints/dual-agent/model_final.pt")

    # 测试
    test_narration(model)
