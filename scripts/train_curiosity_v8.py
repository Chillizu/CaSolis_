#!/usr/bin/env python3
"""
好奇心加权训练 v8 — 7.6M 参数 + 困惑度训练信号

模型: embed(256) → GRU(1024) → shared(1024) → 多头
参数: ~7.6M (7.2x 原始模型)
"""

import os, sys, json, subprocess, random, time, re, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19; N_CMD=8; N_ROLE=4

class CuriosityModel(nn.Module):
    def __init__(self, embed=256, hidden=1024):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(TV, embed)
        self.rnn = nn.GRUCell(embed, hidden)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.lm_head = nn.Linear(hidden, V)
        self.role_head = nn.Linear(hidden, N_ROLE)
        self.cmd_head = nn.Linear(hidden, N_CMD)

    def fwd(self, h, t):
        return self.rnn(self.embed(t), h)

    def forward(self, h, t):
        h2 = self.rnn(self.embed(t), h)
        s = self.shared(h2)
        return h2, self.lm_head(s), self.role_head(s), self.cmd_head(s)

    def init_h(self):
        return torch.zeros(1, self.hidden)

    @torch.no_grad()
    def perplexity_on(self, tokens):
        if len(tokens) < 2: return 0.0
        h = self.init_h(); nll = 0.0
        for i in range(len(tokens)-1):
            h2, lm, _, _ = self(h, torch.tensor([tokens[i]]))
            h = h2.detach()
            nll += F.cross_entropy(lm, torch.tensor([tokens[i+1]])).item()
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

def curiosity_weight(ppl):
    if ppl<2: return 0.1
    if ppl<4: return 0.5
    if ppl<8: return 1.0
    if ppl<15: return 0.7
    return 0.2

def load_data(paths):
    samples=[]
    for p in paths:
        if not os.path.exists(p): continue
        with open(p) as f:
            for line in f:
                d=json.loads(line); toks,roles,obs=parse_text(d["text"])
                if len(toks)>=5: samples.append({"toks":toks,"roles":roles,"obs_toks":obs})
    return samples


DOCKER="casolis-v8"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("for i in $(seq 1 5); do echo line_$i > /tmp/f$i.txt; done")
    dk("mkdir -p /tmp/sub/a"); dk("echo 'port=8080' > /tmp/sub/a/config.cfg")
    dk("echo 'hello world' > /tmp/greet.txt")

def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:2000]


def migrate_weights(model):
    """从 v7 (2.78M) 迁移权重到 v8 (7.6M)"""
    try:
        old = torch.load("checkpoints/curiosity-v1/model_final.pt", map_location="cpu", weights_only=True)
        own = model.state_dict()
        own["embed.weight"][:, :192] = old["embed.weight"]
        own["rnn.weight_ih"][:1536, :192] = old["rnn.weight_ih"]
        own["rnn.weight_hh"][:1536, :512] = old["rnn.weight_hh"]
        own["rnn.bias_ih"][:1536] = old["rnn.bias_ih"]
        own["rnn.bias_hh"][:1536] = old["rnn.bias_hh"]
        own["shared.0.weight"][:512] = old["shared.0.weight"]
        own["shared.0.bias"][:512] = old["shared.0.bias"]
        own["shared.1.weight"][:512, :512] = old["shared.1.weight"]
        own["shared.1.bias"][:512] = old["shared.1.bias"]
        own["lm_head.weight"][:, :512] = old["lm_head.weight"]
        own["lm_head.bias"] = old["lm_head.bias"]
        own["role_head.weight"][:, :512] = old["role_head.weight"]
        own["role_head.bias"] = old["role_head.bias"]
        own["cmd_head.weight"][:, :512] = old["cmd_head.weight"]
        own["cmd_head.bias"] = old["cmd_head.bias"]
        model.load_state_dict(own)
        print("   ✅ 迁移 v7 权重 (192→256, 512→1024)\n")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   ⚠️ 迁移失败: {e}\n")


def train_epoch(model, samples, opt, max_len=100):
    random.shuffle(samples); tl=0; tr=0; tc=0; nb=0; model.train()
    for s in samples:
        tks, roles = s["toks"][:max_len], s["roles"][:max_len]
        if len(tks)<5: continue
        cw = curiosity_weight(model.perplexity_on(s.get("obs_toks",[]))) if s.get("obs_toks") else 0.5
        h=model.init_h(); lm_l=0; rl=0; cl=0
        for i in range(len(tks)-1):
            h2,lm,rgt,cgt=model(h,torch.tensor([tks[i]])); h=h2.detach()
            lm_l+=F.cross_entropy(lm,torch.tensor([tks[i+1]]))
            if roles[i]>0: rl+=F.cross_entropy(rgt,torch.tensor([roles[i]]))
            if roles[i]==2: cl+=F.cross_entropy(cgt,torch.tensor([0]))
        loss=cw*(lm_l+0.3*rl+0.2*cl)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        tl+=lm_l.item(); tr+=rl.item(); tc+=cl.item(); nb+=1
    return tl/nb, tr/nb, tc/nb


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick",action="store_true",help="快速测试模式")
    args=ap.parse_args()

    os.makedirs("checkpoints/curiosity-v8", exist_ok=True)
    model = CuriosityModel()
    print(f"🧠 CuriosityModel v8: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   嵌入: 256, GRU: 1024 ({'%.1f' % (sum(p.numel() for p in model.parameters())/1_053_766)}x 原始)\n")

    # ── 阶段 1: 预训练 ──
    print("="*50)
    print("📚 阶段 1: 预训练")
    print("="*50)
    samples=load_data(glob.glob("data/thoughts/lv*.jsonl"))
    if not samples:
        print("❌ 没有数据!"); sys.exit(1)
    print(f"   数据: {len(samples)} 样本\n")

    migrate_weights(model)

    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    n_ep = 3 if args.quick else 6
    for ep in range(n_ep):
        lm_l,rl,cl=train_epoch(model,samples,opt,max_len=80)
        print(f"  ep{ep+1}/{n_ep} | LM:{lm_l:.1f} Role:{rl:.1f} Cmd:{cl:.1f}")

    torch.save(model.state_dict(),"checkpoints/curiosity-v8/pretrain.pt")
    print(f"\n   💾 checkpoints/curiosity-v8/pretrain.pt\n")

    # ── 阶段 2: 好奇心自举 ──
    if not args.quick:
        print("="*50)
        print("🔄 阶段 2: 好奇心加权自举")
        print("="*50)
        dk_start(); all_gen=[]

        for cycle in range(10):
            # 快速生成
            model.eval()
            new_data=[]
            with torch.no_grad():
                for _ in range(3):
                    h=model.init_h(); ft=""
                    cmd=random.choice(["ls","pwd","whoami","date","echo hello","ls /tmp/"])
                    for d in range(2):
                        out=dk(cmd) or "(empty)"
                        # 喂输出 → 生成思考
                        for t in tok.encode(out[:80]).ids[:10]:
                            h=model.fwd(h,torch.tensor([t])); h=h.detach()
                        gt=[]
                        for _ in range(15):
                            pv=gt[-1] if gt else BOS
                            h2=model.fwd(h,torch.tensor([pv])); h=h2.detach()
                            lm=model.lm_head(model.shared(h))
                            lp=F.softmax(lm.squeeze(0)/0.9,dim=-1)
                            nt=torch.multinomial(lp,1).item()
                            if nt>=V: break
                            gt.append(nt)
                        th=tok.decode(gt)[:60] if gt else "看看"
                        # 选下一个命令
                        nc=random.choice(["ls","cat /tmp/f1.txt","pwd","echo test","whoami","id"])
                        ft+=f"[THOUGHT] {th}\n[CMD] {cmd}\n[OBS] {out}\n"
                        cmd=nc
                    new_data.append({"text":ft})

            # 训练新数据
            parsed=[]
            for s in new_data:
                toks,roles,obs=parse_text(s["text"])
                if len(toks)>=5:
                    ppl=model.perplexity_on(obs) if obs else 0
                    parsed.append({"toks":toks,"roles":roles,"ppl":ppl,"cw":curiosity_weight(ppl)})

            random.shuffle(parsed); tl=0; tr=0; tc=0; nb=0; model.train()
            for s in parsed:
                tks,roles=s["toks"][:100],s["roles"][:100]
                if len(tks)<5: continue
                h=model.init_h(); lm_l=0; rl=0; cl=0
                for i in range(len(tks)-1):
                    h2,lm,rgt,cgt=model(h,torch.tensor([tks[i]])); h=h2.detach()
                    lm_l+=F.cross_entropy(lm,torch.tensor([tks[i+1]]))
                    if roles[i]>0: rl+=F.cross_entropy(rgt,torch.tensor([roles[i]]))
                    if roles[i]==2: cl+=F.cross_entropy(cgt,torch.tensor([0]))
                loss=s["cw"]*(lm_l+0.3*rl+0.2*cl)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
                tl+=lm_l.item(); tr+=rl.item(); tc+=cl.item(); nb+=1

            ap_ppl=sum(s["ppl"] for s in parsed)/max(len(parsed),1) if parsed else 0
            print(f"  周期{cycle+1:>2}/10 | LM:{tl/nb:.1f} R:{tr/nb:.1f} C:{tc/nb:.1f} | 困惑度:{ap_ppl:.1f}")
            if (cycle+1)%5==0:
                torch.save(model.state_dict(),f"checkpoints/curiosity-v8/model-{cycle+1}.pt")
                print(f"   💾 checkpoint {cycle+1}")

        subprocess.run(["docker","kill",DOCKER],capture_output=True)

    # 最终
    torch.save(model.state_dict(),"checkpoints/curiosity-v8/model_final.pt")
    print(f"\n✅ 完成! {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   checkpoints/curiosity-v8/model_final.pt")
