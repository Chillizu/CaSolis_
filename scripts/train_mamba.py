#!/usr/bin/env python3
"""
Mamba 训练 — 用 Mamba(7.3M) 替代 GRU

训练数据同 curiosity_v8: 模板数据 + 好奇心加权
"""

import os, sys, json, glob, random, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_model import MambaModel, MambaBlock

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

# 带多头的 Mamba 模型（兼容多目标训练）
class MambaThoughtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 768)
        self.mamba = MambaBlock(768, d_state=8, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(768)
        self.shared = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU())
        self.lm_head = nn.Linear(768, V)
        self.role_head = nn.Linear(768, 4)  # 0=none, 1=thought, 2=cmd, 3=obs
        self.cmd_head = nn.Linear(768, 8)   # 命令类型

    def forward(self, h, t):
        """兼容 GRU 接口的单步推理"""
        if t.dim() == 0: t = t.unsqueeze(0)
        if t.dim() == 2: t = t.squeeze(0)
        x = self.embed(t)
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.mamba(x)
        x = self.norm(x.squeeze(1))
        s = self.shared(x)
        return x, self.lm_head(s), self.role_head(s), self.cmd_head(s)

    def forward_seq(self, tokens):
        """批量序列训练"""
        x = self.embed(tokens)
        y = self.mamba(x)
        y = self.norm(y)
        s = self.shared(y)
        return self.lm_head(s), self.role_head(s), self.cmd_head(s)

    def init_h(self):
        return torch.zeros(1, 768)

    @torch.no_grad()
    def perplexity_on(self, tokens):
        if len(tokens) < 2: return 0.0
        x = torch.tensor([tokens[:-1]])
        lm, _, _ = self.forward_seq(x)
        nll = 0.0
        for i in range(len(tokens)-1):
            nll += F.cross_entropy(lm[0, i:i+1], torch.tensor([tokens[i+1]])).item()
        return math.exp(nll / (len(tokens)-1))


def parse_text(text):
    parts = re.split(r'(\[THOUGHT\]|\[CMD\]|\[OBS\])', text)
    rm = {"[THOUGHT]":1,"[CMD]":2,"[OBS]":3}; cur=0; toks=[]; roles=[]; obs=[]
    for p in parts:
        if p in rm: cur=rm[p]
        elif p.strip():
            for tid in tok.encode(p).ids:
                toks.append(tid); roles.append(cur)
                if cur==3: obs.append(tid)
    return toks, roles, obs

def load_data(paths):
    samples=[]
    for p in paths:
        if not os.path.exists(p): continue
        with open(p) as f:
            for line in f:
                d=json.loads(line); toks,roles,obs=parse_text(d["text"])
                if len(toks)>=5: samples.append({"toks":toks,"roles":roles,"obs_toks":obs})
    return samples


if __name__ == "__main__":
    os.makedirs("checkpoints/mamba-v1", exist_ok=True)

    model = MambaThoughtModel()
    print(f"🧠 MambaThoughtModel: {sum(p.numel() for p in model.parameters()):,} params\n")

    # ── 预训练 ──
    print("="*50)
    print("📚 预训练")
    print("="*50)
    samples=load_data(glob.glob("data/thoughts/lv*.jsonl"))
    print(f"   数据: {len(samples)} 样本\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for ep in range(5):
        random.shuffle(samples)
        tl=0; tr=0; tc=0; nb=0; model.train()
        for s in samples:
            tks, roles = s["toks"][:80], s["roles"][:80]
            if len(tks)<5: continue
            t = torch.tensor([tks[:-1]])
            lm, rgt, cgt = model.forward_seq(t)
            lm_l = F.cross_entropy(lm.reshape(-1, V), torch.tensor(tks[1:]))
            rl = sum(F.cross_entropy(rgt[0,i:i+1], torch.tensor([roles[i]]))
                    for i in range(len(roles)-1) if roles[i]>0) or 0
            cl = sum(F.cross_entropy(cgt[0,i:i+1], torch.tensor([0]))
                    for i in range(len(roles)-1) if roles[i]==2) or 0
            loss = lm_l + (0.3*rl if isinstance(rl,torch.Tensor) else 0) + (0.2*cl if isinstance(cl,torch.Tensor) else 0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            tl+=lm_l.item(); nb+=1
            if isinstance(rl,torch.Tensor): tr+=rl.item()
            if isinstance(cl,torch.Tensor): tc+=cl.item()
        print(f"  ep{ep+1}/5 | LM:{tl/nb:.1f}")

    torch.save(model.state_dict(), "checkpoints/mamba-v1/pretrain.pt")
    print(f"\n   💾 pretrain.pt\n")

    # ── 测试生成 ──
    print("🧪 测试生成:")
    model.eval()
    with torch.no_grad():
        for seed in ["[THOUGHT] 看看目录\n[CMD] "]:
            x = tok.encode(seed).ids
            all_ids = list(x)
            for _ in range(60):
                lm, _, _ = model.forward_seq(torch.tensor([all_ids[-30:]]))
                lp = F.softmax(lm[0,-1]/0.85, dim=-1)
                nt = torch.multinomial(lp,1).item()
                if nt >= V: break
                all_ids.append(nt)
            print(f"  '{seed}' → {tok.decode(all_ids[len(x):])[:120]}")

    torch.save(model.state_dict(), "checkpoints/mamba-v1/model_final.pt")
    print(f"\n✅ 完成! checkpoints/mamba-v1/model_final.pt")
