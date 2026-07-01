#!/usr/bin/env python3
"""
多目标思考链训练 v2 — 课程学习 + 多样化数据

改动:
  1. 用大规模数据 (400+ 样本)
  2. 课程学习: L1→L2→L3→L4 逐步训练
  3. 三头损失: LM + Role + CmdType
  4. 每阶段测试生成
"""

import os, sys, json, glob, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

CMD_TYPES = {
    "ls":0,"ll":0,"la":0,"cat":1,"head":1,"tail":1,"echo":2,"printf":2,
    "touch":3,"mkdir":3,"cp":3,"mv":3,"rm":3,
    "grep":4,"sort":4,"wc":4,"cut":4,
    "pwd":5,"whoami":5,"id":5,"date":5,"uname":5,"hostname":5,
    "find":6,"ps":6,"df":6,"du":6,"for":7,"while":7,"if":7,"true":7,
}
N_CMD = 8; N_ROLE = 4

class ThoughtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 96)
        self.rnn = nn.GRUCell(96, 256)
        self.shared = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())
        self.lm_head = nn.Linear(256, V)
        self.role_head = nn.Linear(256, N_ROLE)
        self.cmd_head = nn.Linear(256, N_CMD)

    def forward(self, h, t):
        h2 = self.rnn(self.embed(t), h)
        s = self.shared(h2)
        return h2, self.lm_head(s), self.role_head(s), self.cmd_head(s)

    def init_state(self):
        return torch.zeros(1, 256)


def parse_text(text):
    """解析 [THOUGHT]/[CMD]/[OBS] 文本为 token+role+cmdtype"""
    parts = re.split(r'(\[THOUGHT\]|\[CMD\]|\[OBS\])', text)
    role_map = {"[THOUGHT]":1,"[CMD]":2,"[OBS]":3}
    cur = 0

    toks, roles, cmds = [], [], []
    for p in parts:
        if p in role_map:
            cur = role_map[p]
        elif p.strip():
            enc = tok.encode(p)
            for tid in enc.ids:
                toks.append(tid)
                roles.append(cur)
                if cur == 2:
                    w = p.strip().split()[0] if p.strip() else ""
                    cmds.append(CMD_TYPES.get(w, 7))
                else:
                    cmds.append(-1)

    return toks, roles, cmds


def load_data(paths):
    samples = []
    for p in paths:
        if not os.path.exists(p): continue
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                toks, roles, cmds = parse_text(d["text"])
                if len(toks) >= 5:
                    samples.append({"toks":toks, "roles":roles, "cmds":cmds, "text":d["text"]})
    return samples


def train_epoch(model, samples, ep, epochs, opt, max_len):
    random.shuffle(samples)
    total_lm=0; total_r=0; total_c=0; nb=0

    for s in samples:
        tks, roles, cmd_ts = s["toks"], s["roles"], s["cmds"]
        if len(tks) > max_len:
            tks = tks[:max_len]
            roles = roles[:max_len]
            cmd_ts = cmd_ts[:max_len]
        if len(tks) < 5: continue

        h = torch.zeros(1,256)
        lm_l=0; rl=0; cl=0

        for i in range(len(tks)-1):
            h2, lm, rlgt, clgt = model(h, torch.tensor([tks[i]]))
            h = h2.detach()

            lm_l += F.cross_entropy(lm, torch.tensor([tks[i+1]]))

            if roles[i] > 0:
                rl += F.cross_entropy(rlgt, torch.tensor([roles[i]]))

            if cmd_ts[i] >= 0:
                cl += F.cross_entropy(clgt, torch.tensor([cmd_ts[i]]))

        loss = lm_l + 0.3*rl + 0.2*cl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        total_lm += lm_l.item(); total_r += rl.item()
        total_c += cl.item(); nb += 1

    return total_lm/nb, total_r/nb, total_c/nb


def gen_sample(model, prompt, temp=0.75, max_n=80):
    model.eval()
    with torch.no_grad():
        pre = tok.encode(prompt).ids
        h = torch.zeros(1,256)
        out = list(pre)

        for t in pre:
            h, _, _, _ = model(h, torch.tensor([t]))
            h = h.detach()

        for _ in range(max_n):
            h, lm, _, _ = model(h, torch.tensor([out[-1]]))
            h = h.detach()
            lp = F.softmax(lm.squeeze(0)/temp, dim=-1)
            nt = torch.multinomial(lp,1).item()
            if nt >= V: break
            out.append(nt)

        return tok.decode(out)


if __name__ == "__main__":
    # 模型初始化
    model = ThoughtModel()
    sd = torch.load("checkpoints/general-v1/model_final.pt", map_location="cpu")
    own = model.state_dict()
    for k,v in sd.items():
        if k in own and v.shape == own[k].shape:
            own[k] = v
    model.load_state_dict(own)
    print(f"🧠 模型: {sum(p.numel() for p in model.parameters()):,} params")

    # 课程学习
    data_dirs = [
        (1, ["data/thoughts/lv1.jsonl"], 40),
        (2, ["data/thoughts/lv1.jsonl", "data/thoughts/lv2.jsonl"], 70),
        (3, ["data/thoughts/lv1.jsonl", "data/thoughts/lv2.jsonl", "data/thoughts/lv3.jsonl"], 110),
        (4, ["data/thoughts/lv1.jsonl", "data/thoughts/lv2.jsonl", "data/thoughts/lv3.jsonl", "data/thoughts/lv4.jsonl"], 200),
    ]

    for lv, files, max_len in data_dirs:
        samples = load_data(files)
        if not samples:
            print(f"⚠️  L{lv}: 没有数据，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"📚 课程 L{lv}: {len(samples)} 样本, 最大长度 {max_len}")
        print(f"{'='*50}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        epochs = 5 if lv == 1 else 4

        for ep in range(epochs):
            lm_l, rl, cl = train_epoch(model, samples, ep, epochs, opt, max_len)
            print(f"  ep{ep+1}/{epochs} | LM:{lm_l:.1f} Role:{rl:.1f} Cmd:{cl:.1f}")

        # 阶段测试
        for pr in ["[THOUGHT] 看看当前目录\n[CMD] "]:
            gen = gen_sample(model, pr)
            print(f"  🧪 '{pr}'")
            print(f"       → {gen[:120]}")
            print()

    # 最终保存
    os.makedirs("checkpoints/thought-v2", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/thought-v2/model_final.pt")
    print(f"💾 checkpoints/thought-v2/model_final.pt")

    # 完整测试
    print("\n🧪 最终生成测试:")
    for pr in ["[THOUGHT] 我想看看目录里有什么\n[CMD] ",
                "[THOUGHT] 检查当前用户\n[CMD] ",
                "[THOUGHT] 好奇临时目录\n[CMD] "]:
        gen = gen_sample(model, pr)
        print(f"  '{pr}' → {gen[:150]}\n")
