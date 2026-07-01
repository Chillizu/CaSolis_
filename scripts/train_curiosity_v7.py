#!/usr/bin/env python3
"""
好奇心加权训练 v7 — 大模型 + 困惑度作为训练信号

模型: embed(192) → GRU(512) → shared(512) → 多头输出
参数: ~2.4M (2.3x 当前模型)

训练:
  1. 预训练: 现有模板数据 → 学格式
  2. 自举: Docker 中生成 → 计算困惑度 → 好奇心加权训练

好奇心加权:
  - 对每个样本的 [OBS] 部分计算困惑度
  - 困惑度 3-10 → 有趣 → 高权重
  - 困惑度 <3 → 无聊 → 低权重
  - 困惑度 >10 → 噪声 → 丢弃
"""

import os, sys, json, subprocess, random, time, re, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

N_CMD=8; N_ROLE=4

# ── 更大模型 ×2.4 ──
class CuriosityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 192)
        self.rnn = nn.GRUCell(192, 512)
        self.shared = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU())
        self.lm_head = nn.Linear(512, V)
        self.role_head = nn.Linear(512, N_ROLE)
        self.cmd_head = nn.Linear(512, N_CMD)

    def fwd(self, h, t):
        h2 = self.rnn(self.embed(t), h)
        s = self.shared(h2)
        return h2, self.lm_head(s), self.role_head(s), self.cmd_head(s)

    def perplexity_on(self, tokens: list[int]) -> float:
        """计算某段 token 的平均困惑度"""
        if len(tokens) < 2: return 0.0
        h = torch.zeros(1, 512)
        total_nll = 0.0
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                h2, lm, _, _ = self.fwd(h, torch.tensor([tokens[i]]))
                h = h2.detach()
                loss = F.cross_entropy(lm, torch.tensor([tokens[i+1]]))
                total_nll += loss.item()
        avg_nll = total_nll / (len(tokens) - 1)
        return math.exp(avg_nll)


def parse_text(text):
    """解析 [THOUGHT]/[CMD]/[OBS] 格式, 返回 tokens, roles, obs_mask"""
    parts = re.split(r'(\[THOUGHT\]|\[CMD\]|\[OBS\])', text)
    role_map = {"[THOUGHT]":1,"[CMD]":2,"[OBS]":3}
    cur = 0; toks=[]; roles=[]; obs_tokens=[]

    for p in parts:
        if p in role_map:
            cur = role_map[p]
        elif p.strip():
            enc = tok.encode(p)
            for tid in enc.ids:
                toks.append(tid); roles.append(cur)
                if cur == 3: obs_tokens.append(tid)

    return toks, roles, obs_tokens


def curiosity_weight(ppl: float) -> float:
    """困惑度 → 好奇心权重: 中等困惑度 = 高权重"""
    if ppl < 2.0: return 0.1      # 太无聊了
    if ppl < 4.0: return 0.5      # 有点无聊
    if ppl < 8.0: return 1.0      # 刚好！最有好奇心
    if ppl < 15.0: return 0.7     # 有点随机但还能学
    return 0.2                     # 太随机了, 几乎噪声


def load_data(paths):
    samples = []
    for p in paths:
        if not os.path.exists(p): continue
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                toks, roles, obs_toks = parse_text(d["text"])
                if len(toks) >= 5:
                    samples.append({
                        "toks":toks, "roles":roles, "obs_toks":obs_toks,
                        "text":d["text"][:200]
                    })
    return samples


DOCKER = "folunar-c7"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("for i in $(seq 1 5); do echo line_$i > /tmp/f$i.txt; done")
    dk("mkdir -p /tmp/sub/a")
    dk("echo 'port=8080\ndebug=true' > /tmp/sub/a/config.cfg")
    dk("echo 'hello world' > /tmp/greet.txt")
def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]


def train_epoch(model, samples, opt, max_len=150):
    random.shuffle(samples)
    total_lm=0; total_r=0; total_c=0; nb=0
    model.train()

    for s in samples:
        tks, roles = s["toks"][:max_len], s["roles"][:max_len]
        if len(tks) < 5: continue

        # 计算好奇心权重
        obs_toks = s.get("obs_toks", [])
        cw = curiosity_weight(model.perplexity_on(obs_toks)) if obs_toks else 0.5

        h = torch.zeros(1, 512); lm_l=0; rl=0; cl=0

        for i in range(len(tks)-1):
            h2, lm, rgt, cgt = model.fwd(h, torch.tensor([tks[i]]))
            h = h2.detach()
            lm_l += F.cross_entropy(lm, torch.tensor([tks[i+1]]))
            if roles[i] > 0:
                rl += F.cross_entropy(rgt, torch.tensor([roles[i]]))
            if roles[i] == 2:
                cl += F.cross_entropy(cgt, torch.tensor([0]))

        loss = cw * (lm_l + 0.3*rl + 0.2*cl)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        total_lm += lm_l.item(); total_r += rl.item()
        total_c += cl.item(); nb += 1

    return total_lm/nb, total_r/nb, total_c/nb


def bootstrap_generate(model, n_samples=5):
    """在 Docker 中生成好奇心加权训练数据"""
    samples = []
    starters = ["ls","pwd","whoami","echo hello","date","cat /tmp/greet.txt",
                "ls /tmp/","id","uname -a"]

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            h = torch.zeros(1, 512)
            full_text = ""
            cmd = random.choice(starters)

            for depth in range(3):
                out = dk(cmd)
                if not out: out = "(empty)"

                # 1. 生成思考（从隐藏状态续写）
                out_toks = tok.encode(out[:100]).ids[:15]
                for t in out_toks:
                    h, _, _, _ = model.fwd(h, torch.tensor([t]))
                    h = h.detach()

                # 续写 [THOUGHT] 部分
                gen_toks = []
                for _ in range(20):
                    prev = gen_toks[-1] if gen_toks else BOS
                    h, lm, _, _ = model.fwd(h, torch.tensor([prev]))
                    h = h.detach()
                    lp = F.softmax(lm.squeeze(0)/0.9, dim=-1)
                    nt = torch.multinomial(lp, 1).item()
                    if nt >= V: break
                    gen_toks.append(nt)
                thought = tok.decode(gen_toks) if gen_toks else "继续探索"

                # 2. 生成/选择下一个命令
                allowed = ["ls","cat","echo","pwd","whoami","id","date","head"]
                if random.random() < 0.5:
                    next_cmd = random.choice(starters)
                else:
                    cmd_toks = [BOS]
                    for _ in range(10):
                        h, lm, _, _ = model.fwd(h, torch.tensor([cmd_toks[-1] if cmd_toks else BOS]))
                        h = h.detach()
                        lp = F.softmax(lm.squeeze(0)/1.0, dim=-1)
                        nt = torch.multinomial(lp, 1).item()
                        if nt >= V: break
                        cmd_toks.append(nt)
                    gen_cmd = tok.decode(cmd_toks[1:])[:15].strip().split()[0] if len(cmd_toks)>1 else "ls"
                    next_cmd = gen_cmd if gen_cmd in allowed else "ls"

                step = f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {out}\n"
                full_text += step
                cmd = next_cmd

            samples.append({"text": full_text})

    return samples


if __name__ == "__main__":
    os.makedirs("checkpoints/curiosity-v1", exist_ok=True)

    # 初始化模型
    model = CuriosityModel()
    print(f"🧠 CuriosityModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   嵌入: 192, 隐藏: 512 ({'%.1f' % (sum(p.numel() for p in model.parameters())/1_053_766)}x current)\n")

    # 阶段 1: 预训练 (模板数据)
    print("=" * 50)
    print("📚 阶段 1: 预训练")
    print("=" * 50)

    pretrain_files = glob.glob("data/thoughts/lv*.jsonl")
    samples = load_data(pretrain_files)
    print(f"   数据: {len(samples)} 模板样本\n")

    # 从旧模型迁移权重（适配不同维度）
    try:
        old = torch.load("checkpoints/thought-v2/model_final.pt", map_location="cpu", weights_only=True)
        own = model.state_dict()
        
        # embed: old(2041,96) → new(2041,192)
        own["embed.weight"][:, :96] = old["embed.weight"]
        
        # GRU weight_ih: old(768,96) → new(1536,192)
        own["rnn.weight_ih"][:768, :96] = old["rnn.weight_ih"]
        # GRU weight_hh: old(768,256) → new(1536,512)
        own["rnn.weight_hh"][:768, :256] = old["rnn.weight_hh"]
        # GRU biases: old(768) → new(1536)
        own["rnn.bias_ih"][:768] = old["rnn.bias_ih"]
        own["rnn.bias_hh"][:768] = old["rnn.bias_hh"]
        
        # shared LayerNorm: old(256) → new(512)
        own["shared.0.weight"][:256] = old["shared.0.weight"]
        own["shared.0.bias"][:256] = old["shared.0.bias"]
        # shared Linear: old(256,256) → new(512,512)
        own["shared.1.weight"][:256, :256] = old["shared.1.weight"]
        own["shared.1.bias"][:256] = old["shared.1.bias"]
        
        # lm_head: old(2022,256) → new(2022,512)，复制前256列
        own["lm_head.weight"][:, :256] = old["lm_head.weight"]
        own["lm_head.bias"] = old["lm_head.bias"]  # 同形状(2022,)
        
        # role_head: old(4,256) → new(4,512)
        own["role_head.weight"][:, :256] = old["role_head.weight"]
        own["role_head.bias"] = old["role_head.bias"]  # 同形状(4,)
        # cmd_head: old(8,256) → new(8,512)
        own["cmd_head.weight"][:, :256] = old["cmd_head.weight"]
        own["cmd_head.bias"] = old["cmd_head.bias"]  # 同形状(8,)
        
        model.load_state_dict(own)
        print("   ✅ 迁移旧模型权重（96→192, 256→512）\n")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   ⚠️ 权重迁移失败: {e}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for ep in range(5):
        lm_l, rl, cl = train_epoch(model, samples, opt, max_len=80)
        print(f"  ep{ep+1}/8 | LM:{lm_l:.1f} Role:{rl:.1f} Cmd:{cl:.1f}")

    torch.save(model.state_dict(), "checkpoints/curiosity-v1/pretrain.pt")
    print(f"\n   💾 checkpoints/curiosity-v1/pretrain.pt\n")

    # 阶段 2: 好奇心自举
    print("=" * 50)
    print("🔄 阶段 2: 好奇心加权自举")
    print("=" * 50)
    dk_start()

    all_gen = []
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for cycle in range(8):
        # 生成
        new = bootstrap_generate(model, n_samples=3)
        all_gen.extend(new)

        # 解析并计算好奇心
        parsed = []
        for s in new:
            toks, roles, obs_toks = parse_text(s["text"])
            if len(toks) < 5: continue
            ppl = model.perplexity_on(obs_toks) if obs_toks else 0
            cw = curiosity_weight(ppl)
            parsed.append({"toks":toks, "roles":roles, "obs_toks":obs_toks,
                          "ppl":ppl, "cw":cw})

        # 训练
        random.shuffle(parsed)
        total_lm=0; total_r=0; total_c=0; nb=0
        model.train()

        for s in parsed:
            tks, roles = s["toks"][:120], s["roles"][:120]
            if len(tks) < 5: continue
            cw = s["cw"]

            h = torch.zeros(1, 512); lm_l=0; rl=0; cl=0
            for i in range(len(tks)-1):
                h2, lm, rgt, cgt = model.fwd(h, torch.tensor([tks[i]]))
                h = h2.detach()
                lm_l += F.cross_entropy(lm, torch.tensor([tks[i+1]]))
                if roles[i] > 0: rl += F.cross_entropy(rgt, torch.tensor([roles[i]]))
                if roles[i] == 2: cl += F.cross_entropy(cgt, torch.tensor([0]))

            loss = cw * (lm_l + 0.3*rl + 0.2*cl)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt2.step()
            total_lm += lm_l.item(); total_r += rl.item()
            total_c += cl.item(); nb += 1

        avg_ppl = sum(s["ppl"] for s in parsed) / max(len(parsed), 1) if parsed else 0
        avg_cw = sum(s["cw"] for s in parsed) / max(len(parsed), 1) if parsed else 0
        print(f"  周期{cycle+1:>2}/15 | LM:{total_lm/nb:.1f} R:{total_r/nb:.1f} C:{total_c/nb:.1f} | 平均困惑度:{avg_ppl:.1f} 好奇心权重:{avg_cw:.2f}")

        if (cycle+1) % 4 == 0:
            torch.save(model.state_dict(), f"checkpoints/curiosity-v1/model-{cycle+1}.pt")
            print(f"   💾 checkpoint {cycle+1}")

    # 最终保存
    torch.save(model.state_dict(), "checkpoints/curiosity-v1/model_final.pt")
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    print(f"\n✅ 完成! checkpoints/curiosity-v1/model_final.pt")
    print(f"   生成数据: {len(all_gen)} 条 (好奇心加权训练)")
    print(f"   总参数: {sum(p.numel() for p in model.parameters()):,}")
