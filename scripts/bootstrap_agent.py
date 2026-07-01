#!/usr/bin/env python3
"""
自举思考链生成器 — 模型在 Docker 中自主探索，收集自己的思考链

流程:
  1. 加载已训练的 thought-v2 模型
  2. 模型在 Docker 里自由探索：生成 [THOUGHT]→执行 [CMD]→观察 [OBS]
  3. 收集每一步为训练数据
  4. 每次收集够 N 条 → 增量训练
  5. 循环：收集→训练→收集→训练→...
"""

import os, sys, json, subprocess, random, time, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; EOS = V+17; TV = V+19

CMD_TYPES = {"ls":0,"ll":0,"la":0,"cat":1,"head":1,"tail":1,"echo":2,"printf":2,
    "touch":3,"mkdir":3,"cp":3,"mv":3,"rm":3,"grep":4,"sort":4,"wc":4,"cut":4,
    "pwd":5,"whoami":5,"id":5,"date":5,"uname":5,"hostname":5,
    "find":6,"ps":6,"df":6,"du":6,"for":7,"while":7,"if":7,"true":7}
N_CMD=8; N_ROLE=4

class ThoughtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(TV,96); self.rnn=nn.GRUCell(96,256)
        self.shared=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())
        self.lm_head=nn.Linear(256,V); self.role_head=nn.Linear(256,N_ROLE)
        self.cmd_head=nn.Linear(256,N_CMD)
    def fwd(self,h,t):
        h2=self.rnn(self.embed(t),h); s=self.shared(h2)
        return h2,self.lm_head(s),self.role_head(s),self.cmd_head(s)

DOCKER="folunar-boot"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("for i in $(seq 1 5); do echo line_$i > /tmp/f$i.txt; done")
    dk("mkdir -p /tmp/sub/a /tmp/sub/b")
    dk("echo 'port=8080\ndebug=true' > /tmp/sub/a/config.cfg")
    dk("echo 'TODO: fix login' > /tmp/todo.txt")

def dk(cmd):
    r=subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],capture_output=True,timeout=15)
    return (r.stdout or b"").decode("utf-8",errors="replace").strip()[:1500]


def load_model(path):
    m=ThoughtModel()
    sd=torch.load(path,map_location="cpu",weights_only=True)
    m.load_state_dict(sd); m.eval()
    return m

def generate_thought(m, h, context_tokens=None, max_n=30, temp=0.8):
    """从隐藏状态继续生成"""
    with torch.no_grad():
        out_tokens=[]
        for _ in range(max_n):
            prev=out_tokens[-1] if out_tokens else BOS
            h,lm,_,_=m.fwd(h,torch.tensor([prev]))
            h=h.detach()
            lp=F.softmax(lm.squeeze(0)/temp,dim=-1)
            # 略微压制重复token
            if out_tokens:
                for dup in set(out_tokens[-5:]):
                    lp[dup]*=0.85
                lp=lp/lp.sum()
            nt=torch.multinomial(lp,1).item()
            if nt>=V or nt==EOS: break
            out_tokens.append(nt)
        return h, tok.decode(out_tokens) if out_tokens else ""


def bootstrap_cycle(m, n_per_cycle=10):
    """一轮自举：生成 N 条思考链"""
    samples=[]

    # 启动命令池
    starters=["ls","pwd","whoami","id","date","uname -a",
              "cat /tmp/f1.txt","ls /tmp/","echo hello"]

    for _ in range(n_per_cycle):
        cmd=random.choice(starters)
        full_chain=""
        h=torch.zeros(1,256)

        for depth in range(3):
            out=dk(cmd)
            if not out: out="(empty)"

            # 让模型生成对当前输出的"思考"
            obs_tokens=tok.encode(out[:200]).ids[:30]
            # 喂观察给模型
            for t in obs_tokens:
                h,_,_,_=m.fwd(h,torch.tensor([t]))
                h=h.detach()

            # 生成思考
            h, thought = generate_thought(m, h, temp=0.85, max_n=25)
            if not thought: thought="继续探索"

            # 生成命令
            cmd_prompt="[CMD] "
            for c in tok.encode(cmd_prompt).ids:
                h,_,_,_=m.fwd(h,torch.tensor([c]))
                h=h.detach()

            # 决定下一个命令（混合模型生成+随机）
            if random.random()<0.4:
                next_cmd=random.choice(["ls","pwd","cat /tmp/f$((RANDOM%5+1)).txt",
                    "echo $RANDOM","ls -la /tmp/","whoami"])
            else:
                h, cmd_text = generate_thought(m, h, temp=0.9, max_n=15)
                next_cmd=cmd_text.strip().split()[0] if cmd_text.strip() else "ls"
                # 过滤不安全命令
                if not any(c in next_cmd for c in ["ls","cat","echo","pwd","whoami","id","date","uname","head","wc"]):
                    next_cmd="ls"

            step=f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {out}\n"
            full_chain+=step
            cmd=next_cmd

        samples.append({"text":full_chain,"source":"bootstrap"})

    return samples


def train_step(m, samples, opt):
    """增量训练"""
    m.train()
    total_lm=0; total_r=0; total_c=0; n=0

    for s in samples:
        text=s["text"]
        # 解析角色
        parts=re.split(r'(\[THOUGHT\]|\[CMD\]|\[OBS\])',text)
        role_map={"[THOUGHT]":1,"[CMD]":2,"[OBS]":3}
        cur=0; toks=[]; roles=[]; cmd_ts=[]

        for p in parts:
            if p in role_map: cur=role_map[p]
            elif p.strip():
                enc=tok.encode(p)
                for tid in enc.ids:
                    toks.append(tid); roles.append(cur)
                    if cur==2:
                        w=p.strip().split()[0] if p.strip() else ""
                        cmd_ts.append(CMD_TYPES.get(w,7))
                    else: cmd_ts.append(-1)

        if len(toks)<5: continue

        h=torch.zeros(1,256); lm_l=0; rl=0; cl=0
        for i in range(len(toks)-1):
            h2,lm,rgt,cgt=m.fwd(h,torch.tensor([toks[i]]))
            h=h2.detach()
            lm_l+=F.cross_entropy(lm,torch.tensor([toks[i+1]]))
            if roles[i]>0: rl+=F.cross_entropy(rgt,torch.tensor([roles[i]]))
            if cmd_ts[i]>=0: cl+=F.cross_entropy(cgt,torch.tensor([cmd_ts[i]]))
        loss=lm_l+0.3*rl+0.2*cl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),0.5)
        opt.step()
        total_lm+=lm_l.item(); total_r+=rl.item()
        total_c+=cl.item(); n+=1

    return total_lm/n, total_r/n, total_c/n


if __name__=="__main__":
    dk_start()
    os.makedirs("checkpoints/thought-v3",exist_ok=True)

    # 加载现有模型
    m=load_model("checkpoints/thought-v2/model_final.pt")
    opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
    print(f"🧠 加载模型: {sum(p.numel() for p in m.parameters()):,} params\n")

    all_samples=[]

    for cycle in range(20):
        print(f"🔄 自举周期 {cycle+1}/20")

        # 1. 收集
        new=bootstrap_cycle(m, n_per_cycle=5)
        all_samples.extend(new)
        print(f"   收集: {len(new)} 新样本 (总计 {len(all_samples)})")

        # 2. 训练
        lm_l,rl,cl=train_step(m,new,opt)
        print(f"   训练: LM:{lm_l:.1f} Role:{rl:.1f} Cmd:{cl:.1f}")

        # 3. 每5轮保存+测试
        if (cycle+1)%5==0:
            # 保存
            pt=f"checkpoints/thought-v3/model-{cycle+1}.pt"
            torch.save(m.state_dict(),pt)
            print(f"   💾 {pt}")

            # 测试生成
            with torch.no_grad():
                h=torch.zeros(1,256)
                h,_=generate_thought(m,h,max_n=60,temp=0.8)
            print(f"   🧪 测试生成 (未显式给提示):")

        # 保存所有数据
        with open("data/thoughts/bootstrap_all.jsonl","w") as f:
            for s in all_samples:
                f.write(json.dumps(s,ensure_ascii=False)+"\n")

    # 最终保存
    torch.save(m.state_dict(),"checkpoints/thought-v3/model_final.pt")
    print(f"\n✅ 完成! {len(all_samples)} 自举样本")
    print(f"   checkpoints/thought-v3/model_final.pt")
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
