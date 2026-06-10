#!/usr/bin/env python3
"""
自主意识流 v6 — 生成不积累，只训真实输出

模型自由生成(临时)→检测命令→执行→只学命令+输出
每次执行后重置流，防止自指污染
"""

import os, sys, random, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS=V+16; EOS=V+17; TV=V+19

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed=nn.Embedding(TV,96); self.rnn=nn.GRUCell(96,256)
        self.shared=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())
        self.lm_head=nn.Linear(256,V)
    def step(self,h,t):
        if t.dim()==0:t=t.unsqueeze(0)
        return self.rnn(self.token_embed(t),h),self.lm_head(self.shared(h))

D="sa"; docker_ready=False
def ensure_docker():
    global docker_ready
    if not docker_ready:
        subprocess.run(["docker","kill",D],capture_output=True)
        subprocess.run(["docker","rm",D],capture_output=True)
        subprocess.run(["docker","run","-d","--name",D,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
        docker_ready=True
def dk(cmd):
    r=subprocess.run(["docker","exec","-i",D,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:3000]

class A:
    def __init__(self,cp="checkpoints/stream-v4"):
        self.cp=cp; os.makedirs(cp,exist_ok=True)
        self.m=M(); self._load()
        self.opt=torch.optim.AdamW(self.m.parameters(),lr=3e-5)
        self.h=torch.zeros(1,256); self.n=0
        self.cmds=set(); self.stuck=0
        ensure_docker()

    def _load(self):
        import glob
        ps=sorted(glob.glob(f"{self.cp}/model-*.pt"))
        if ps:
            self.m.load_state_dict(torch.load(ps[-1],map_location="cpu",weights_only=True))
            print(f"  加载: {ps[-1]}")
        else:
            sd=torch.load("checkpoints/general-v1/model_final.pt",map_location="cpu",weights_only=True)
            self.m.load_state_dict(sd); print("  从通用模型初始化")
        print(f"  参数: {sum(p.numel() for p in self.m.parameters()):,}")

    def save(self):
        torch.save(self.m.state_dict(),f"{self.cp}/model-{self.n}.pt")

    def think(self,n=25):
        out=[]; temp=.85 if self.stuck<15 else 1.5
        for _ in range(n):
            h2,log=self.m.step(self.h,torch.tensor([BOS]) if not out else torch.tensor([out[-1]]))
            self.h=h2.detach()
            lp=F.softmax(log.squeeze(0)/temp,dim=-1)
            if self.stuck>=15 and out:
                for idx in set(out[-5:]):
                    if idx<V: lp[idx]*=.3
                lp=lp/lp.sum()
            t=torch.multinomial(lp,1).item()
            if t==EOS:
                self.h=torch.zeros(1,256)
                continue
            if t<V: out.append(t)
        return tok.decode(out) if out else "..."

    def learn(self,text):
        tks=tok.encode(text).ids[:120]
        if len(tks)<3: return 0.0
        h=torch.zeros(1,256); tl=0.0; n=0
        for i in range(len(tks)-1):
            h,lm=self.m.step(h,torch.tensor([tks[i]]))
            l=F.cross_entropy(lm,torch.tensor([tks[i+1]]))
            self.opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(self.m.parameters(),.5)
            self.opt.step(); tl+=l.item(); n+=1; h=h.detach()
        return tl/max(n,1)

    def cycle(self):
        thought=self.think(25)
        cmd=None
        m=re.search(r'\$\s*([^\n]+)',thought)
        if m: cmd=m.group(1).strip()

        if not cmd and (self.n<20 or self.stuck>=10):
            cmd=random.choice(["ls","pwd","date","echo hello","whoami","id",
                "cat /etc/hostname","echo t>/tmp/x;cat /tmp/x"])
            if self.stuck>=10: self.h=torch.zeros(1,256); self.stuck=0

        ex=False; out=""; loss=0.0
        if cmd:
            self.cmds.add(cmd); out=dk(cmd); ex=True; self.stuck=0
            loss=self.learn(f"$ {cmd}\n{out}\n")
        else:
            self.stuck+=1

        self.n+=1
        return {"thought":thought[:45],"cmd":cmd or "(none)","exec":ex,"loss":loss}


def main(cp="checkpoints/stream-v4",si=100):
    a=A(cp)
    print(f"\n{'='*50}\n🌊 v6 — 每次重置，只训真实\n{'='*50}\n")
    try:
        while True:
            r=a.cycle()
            if a.n%5==0:
                s="💾" if r["exec"] else "🤔"
                print(f"  [{a.n:>5}] {s} L:{r['loss']:.2f} | {r['thought'][:40]}")
                if r["exec"]: print(f"         ▶️ `{r['cmd'][:35]}`")
            if a.n%si==0:
                a.save(); print(f"\n  💾 {a.n}轮 | 命令:{len(a.cmds)}\n")
    except KeyboardInterrupt: print("\n⏹️")
    finally:
        a.save(); subprocess.run(["docker","kill",D],capture_output=True)
        print(f"\n✅ {a.n}轮 | 自主命令:{len(a.cmds)}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--cp",default="checkpoints/stream-v4")
    p.add_argument("--si",type=int,default=100)
    args=p.parse_args()
    main(cp=args.cp,si=args.si)
