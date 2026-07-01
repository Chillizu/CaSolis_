#!/usr/bin/env python3
"""
Mamba 自主代理 v3 — 无限动态命令 + 元学习 (help/man/which)

每个命令都不重样 → 模型永远学不完 → 持续学习
加入 --help/man/which 模板 → 模型学会自学新工具
"""

import os, sys, json, subprocess, random, time, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_model import MambaBlock

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; TV = V+19

class MambaAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 768)
        self.mamba = MambaBlock(768)
        self.norm = nn.LayerNorm(768)
        self.shared = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU())
        self.lm_head = nn.Linear(768, V)
        # 世界模型头: 从隐藏状态预测输出的嵌入
        self.world_head = nn.Linear(768, 768)

    def forward_seq(self, tokens):
        x = self.embed(tokens)
        y = self.mamba(x)
        y = self.norm(y)
        return self.lm_head(self.shared(y))

    def forward_with_hidden(self, tokens):
        """返回 (logits, hidden_states)"""
        x = self.embed(tokens)
        y = self.mamba(x)
        y = self.norm(y)
        logits = self.lm_head(self.shared(y))
        return logits, y

    def hidden_at(self, tokens):
        """只返回隐藏状态"""
        x = self.embed(tokens)
        y = self.mamba(x)
        return self.norm(y)

    def generate(self, ctx_ids, n=30, temp=0.85):
        out = list(ctx_ids)
        with torch.no_grad():
            for _ in range(n):
                lm = self.forward_seq(torch.tensor([out[-40:]]).long())
                lp = F.softmax(lm[0, -1] / temp, dim=-1)
                nt = torch.multinomial(lp, 1).item()
                if nt >= V: break
                out.append(nt)
        return out[len(ctx_ids):]

    def output_embedding(self, token_ids):
        """计算一组 token 的平均嵌入"""
        if not token_ids:
            return torch.zeros(768)
        emb = self.embed(torch.tensor([token_ids]).long())
        return emb.mean(dim=1).squeeze(0)  # (768,)

    def world_predict(self, h_state):
        """从隐藏状态预测输出嵌入"""
        return self.world_head(h_state)  # (768,)

    def learn(self, token_ids, output_ids=None):
        """训练语言模型 + 世界模型（如果提供输出）"""
        if len(token_ids) < 5: return 0.0, 0.0
        t = torch.tensor([token_ids[:-1]]).long()
        target = torch.tensor(token_ids[1:]).long()
        logits, hidden = self.forward_with_hidden(t)
        lm_loss = F.cross_entropy(logits.reshape(-1, V), target)
        
        world_loss = 0.0
        if output_ids and len(output_ids) > 2:
            # 用最后一个位置的隐藏状态预测输出
            h_last = hidden[0, -1]  # (768,)
            pred_emb = self.world_head(h_last)
            target_emb = self.output_embedding(output_ids)
            world_loss = F.mse_loss(pred_emb, target_emb)
        
        total_loss = lm_loss + 0.3 * world_loss
        self.opt.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.opt.step()
        return lm_loss.item(), world_loss.item() if isinstance(world_loss, torch.Tensor) else 0.0

    def curiosity_score(self, cmd_ids, out_ids):
        """好奇心分数 = 世界模型预测误差"""
        if len(out_ids) < 2:
            return 0.5
        with torch.no_grad():
            h = self.hidden_at(torch.tensor([cmd_ids]).long())
            h_last = h[0, -1]
            pred = self.world_head(h_last)
            target = self.output_embedding(out_ids)
            error = F.mse_loss(pred, target).item()
        return error


DOCKER = "mamba-night"
def dk_start():
    subprocess.run(["docker","kill",DOCKER],capture_output=True)
    subprocess.run(["docker","rm",DOCKER],capture_output=True)
    subprocess.run(["docker","run","-d","--name",DOCKER,"--rm","ubuntu:22.04","sleep","86400"],capture_output=True,timeout=30)
    dk("echo $RANDOM > /tmp/secret.txt")
    dk("echo $((RANDOM * 65536)) > /tmp/rnd_start.txt")
    dk("mkdir -p /tmp/a/x /tmp/a/y /tmp/b/z")
    for i in range(1, 11):
        dk(f"echo line_{i}_$(date +%N) > /tmp/f{i}.txt")

def dk(cmd):
    try:
        r = subprocess.run(["docker","exec","-i",DOCKER,"bash","-c",cmd],
                           capture_output=True, timeout=12)
        return (r.stdout or b"").decode("utf-8", errors="replace").strip()[:2000]
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


# ── 无限动态命令 + 元学习 ──
def make_infinite_cmd(used=set()):
    R = random
    n = R.randint(1, 9999)
    m = R.randint(1, 256)
    ch = R.choice("abcdefghijklmnopqrstuvwxyz")
    ch2 = R.choice("0123456789abcdef")
    charclass = R.choice(["[a-z]", "[0-9]", "[a-f0-9]", "[a-m]", "[n-z]"])
    depth = R.randint(1, 5)

    # ubuntu:22.04 命令池（用于 help/man/which）
    cmds = ["ls","cat","echo","head","tr","date","whoami","pwd","id",
            "uname","seq","ps","free","wc","sort","grep","find","shuf",
            "printf","cp","mv","rm","mkdir","touch","chmod","od","env",
            "cut","tee","basename","dirname","xargs","fmt","fold","nl"]
    rcmd = R.choice(cmds)

    templates = [
        # system state
        f"date +%s.%N",
        f"cat /proc/uptime",
        f"echo $(( {n} * {m} + {R.randint(0,999)} ))",
        f"echo $(( RANDOM * {n} * {m} ))",
        f"cat /proc/loadavg",
        f"free | head -3",
        f"ps aux 2>/dev/null | wc -l",
        f"cat /proc/meminfo | grep -E '^Mem'",

        # random bytes
        f"tr -dc '{charclass}' </dev/urandom | head -c{n}",
        f"head -c{n} /dev/urandom | od -An -tx1 | tr -d ' '",
        f"head -c{m} /dev/urandom | od -An -td1 | tr -d ' '",
        f"wc -c <(head -c{n} /dev/urandom) 2>/dev/null",
        f"cat /proc/stat | head -{n % 5 + 2}",
        f"cat /proc/cpuinfo | head -{n % 10 + 5}",

        # file ops
        f"echo $RANDOM > /tmp/v3_{n}.txt && cat /tmp/v3_{n}.txt",
        f"echo $(date +%N) > /tmp/v3_time_{n}.txt && cat /tmp/v3_time_{n}.txt",
        f"ls /tmp/v3_* 2>/dev/null | wc -l",
        f"cat /tmp/v3_{R.randint(1, n)}.txt 2>/dev/null || echo missing_{R.randint(0,9999)}",
        f"ls -la /tmp/ | head -{R.randint(3, 15)}",

        # explore
        f"find /tmp -name '*{ch}*' -type f 2>/dev/null | head -5",
        f"find / -maxdepth {depth} -name '*.txt' 2>/dev/null | head -{m}",
        f"grep -r '{ch}' /tmp/ 2>/dev/null | head -5",
        f"cat /tmp/f{R.randint(1, 30)}.txt",
        f"ls -R /tmp/ 2>/dev/null | head -15",

        # pipes
        f"echo $(( {n} + {m} * {R.randint(100, 999)} ))",
        f"seq {R.randint(1, 50)} {R.randint(51, 100)} | shuf | head -{R.randint(3, 10)}",
        f"printf '%x\\n' $RANDOM$RANDOM",
        f"for i in 1 2 3; do echo $((RANDOM * {n})); done | sort -n",

        # META: help / man / which — 学会自学
        f"{rcmd} --help 2>&1 | head -10",
        f"which {rcmd}",
        f"type {rcmd} 2>/dev/null || echo not_found",
        f"command -v {rcmd}",
        f"whatis {rcmd} 2>/dev/null || echo no_entry",
    ]

    # wild curiosity (10%)
    if R.random() < 0.10:
        templates += [
            f"compgen -c 2>/dev/null | shuf | head -{n % 15 + 5}",
            f"ls /usr/bin/ | shuf | head -{m % 20 + 5}",
            f"dpkg -l 2>/dev/null | head -{n % 12 + 3}",
            f"cat /etc/shells 2>/dev/null",
            f"apropos '{ch}{ch2}' 2>/dev/null | head -10",
            f"help 2>&1 | head -15",
            f"ls -laR /tmp/ 2>/dev/null | head -20",
        ]

    cmd = R.choice(templates)
    return cmd


if __name__ == "__main__":
    cp_dir = "checkpoints/mamba-v3"
    os.makedirs(cp_dir, exist_ok=True)

    model = MambaAgent()
    sd = torch.load("checkpoints/clean-v1/model_best.pt", map_location="cpu", weights_only=True)
    ms = model.state_dict()
    for k in ms:
        if k in sd and ms[k].shape == sd[k].shape:
            ms[k] = sd[k]
    model.load_state_dict(ms, strict=False)  # world_head 是新加的，不匹配
    model.opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    print(f"🧠 MambaAgent + world_head: {sum(p.numel() for p in model.parameters()):,} params (world_head: {sum(p.numel() for p in model.world_head.parameters()):,})")

    dk_start()

    total = 0
    recent_losses = []
    self_cmd_attempts = 0
    self_cmd_successes = 0
    recent_cmds = []
    recent_obs = []

    print(f"\n{'='*50}")
    print(f"🌙 Mamba clean-v1 探索 ({time.strftime('%H:%M')})")
    print(f"{'='*50}\n")

    try:
        while True:
            # ── 三层策略: 60% 无限 / 20% 系统命令 / 20% 自生 ──
            is_self = False
            is_sys = False
            r = random.random()

            if r < 0.6 or total < 20:
                # 层1: 无限动态命令（稳定学习）
                cmd = make_infinite_cmd()
                out = dk(cmd) or "(empty)"

            elif r < 0.8:
                # 层2: 系统真实命令（发现新工具）
                sys_cmd = dk("compgen -c 2>/dev/null | shuf | head -1")
                if not sys_cmd or "not found" in sys_cmd.lower():
                    sys_cmd = dk("ls /usr/bin/ | shuf | head -1")
                if sys_cmd and sys_cmd.strip() and sys_cmd[0].isalpha():
                    is_sys = True
                    discovered = sys_cmd.strip()[:30]
                    # 执行发现的命令
                    cmd = discovered
                    out = dk(cmd) or "(empty)"
                    # 计算好奇心（世界模型预测误差）
                    cmd_ids = tok.encode(cmd).ids[:20]
                    out_ids = tok.encode(out).ids[:40]
                    curiosity = model.curiosity_score(cmd_ids, out_ids) if len(out_ids) > 2 else 0.5
                    
                    # 总是展示 --help
                    help_out = dk(f"{discovered} --help 2>&1 | head -10") or ""
                    if help_out and len(help_out) > 5 and "not found" not in help_out.lower():
                        disc_text = f"[THOUGHT] discovered '{discovered}'\n[CMD] {discovered} --help\n[OBS] {help_out[:400]}\n"
                        disc_ids = tok.encode(disc_text).ids[:120]
                        model.learn(disc_ids)
                else:
                    cmd = make_infinite_cmd()
                    out = dk(cmd) or "(empty)"

            else:
                # 层3: 模型自生命令（自由探索，不训练）
                seeds = ["[THOUGHT] check\n[CMD] ", "[THOUGHT] see\n[CMD] ",
                         "[CMD] ", "[THOUGHT] look\n[CMD] "]
                ctx = random.choice(seeds)
                ctx_ids = tok.encode(ctx).ids
                gen_ids = model.generate(ctx_ids, n=14, temp=1.1)
                text = tok.decode(gen_ids)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                cmd = " ".join(lines)[:60] if lines else ""
                for pat in ["[THOUGHT]", "[CMD]", "[OBS]", "FILE:"]:
                    cmd = cmd.replace(pat, "").strip()
                if cmd and 2 <= len(cmd) < 60 and cmd not in recent_cmds[-3:]:
                    is_self = True
                    self_cmd_attempts += 1
                    out = dk(cmd) or "(empty)"
                    if out and out != "(empty)" and len(out) > 5:
                        self_cmd_successes += 1
                else:
                    cmd = make_infinite_cmd()
                    out = dk(cmd) or "(empty)"

            recent_cmds.append(cmd)
            recent_obs.append(out[:60])
            recent_cmds = recent_cmds[-20:]
            recent_obs = recent_obs[-20:]

            # generate thought
            ctx_text = f"[THOUGHT] seeing '{out[:30]}' "
            ctx_ids = tok.encode(ctx_text).ids[:20]
            thought_ids = model.generate(ctx_ids, n=15, temp=1.0)
            thought = tok.decode(thought_ids)[:60] if thought_ids else "(observe)"

            # learn: only infinite/system commands, with world model
            world_loss = 0.0
            loss = 0.0
            if not is_self and len(out) > 3:
                train_text = f"[THOUGHT] {thought}\n[CMD] {cmd}\n[OBS] {out}\n"
                train_ids = tok.encode(train_text).ids[:120]
                obs_ids = tok.encode(f"[OBS] {out}\n").ids[:40]
                lm_loss, world_loss = model.learn(train_ids, obs_ids)
                loss = lm_loss + 0.3 * world_loss
            recent_losses.append(loss)

            total += 1

            if total % 10 == 0:
                if is_sys: m = "🔍"
                elif is_self: m = "🖊️"
                else: m = "∞"
                w_str = f" W:{world_loss:.2f}" if not is_self and world_loss > 0 else ""
                print(f"  [{total:>6}] {m} {cmd[:30]:30s} | L:{loss:.2f}{w_str}")
                with open(f"{cp_dir}/log.txt", "a") as f:
                    f.write(f"[{total}] {m} {cmd[:50]} | L:{loss:.2f}{w_str}\n")

            if total % 1000 == 0:
                torch.save(model.state_dict(), f"{cp_dir}/model-{total}.pt")
                sc = f" | self-cmd: {self_cmd_successes}/{self_cmd_attempts}" if self_cmd_attempts else ""
                print(f"\n  💾 model-{total}.pt | {total} 轮{sc}\n")

    except KeyboardInterrupt:
        print(f"\n⏹️  stopped")
    finally:
        torch.save(model.state_dict(), f"{cp_dir}/model_final.pt")
        subprocess.run(["docker","kill",DOCKER],capture_output=True)
        sc = f"\n   self-cmd: {self_cmd_successes}/{self_cmd_attempts} ok" if self_cmd_attempts else ""
        print(f"\n✅ total {total} cycles")
        print(f"   final: {cp_dir}/model_final.pt{sc}")
