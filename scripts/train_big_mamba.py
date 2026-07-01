#!/usr/bin/env python3
"""
Train BIG Mamba (19.7M params) from scratch on clean data.
Saves checkpoints every 5 epochs.
"""

import os, sys, json, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_model import MambaBlock

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); TV = V + 19

class BigMamba(nn.Module):
    def __init__(self, d_model=1408):
        super().__init__()
        self.embed = nn.Embedding(TV, d_model)
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, V)

    def forward_seq(self, tokens):
        return self.lm_head(self.norm(self.mamba(self.embed(tokens))))

    def generate(self, ctx, n=40, temp=0.85):
        out = list(ctx)
        self.eval()
        with torch.no_grad():
            for _ in range(n):
                lm = self.forward_seq(torch.tensor([out[-60:]]).long())
                lp = F.softmax(lm[0, -1] / temp, dim=-1)
                nt = torch.multinomial(lp, 1).item()
                if nt >= V: break
                out.append(nt)
        self.train()
        return tok.decode(out[len(ctx):])


if __name__ == "__main__":
    os.makedirs("checkpoints/big-mamba", exist_ok=True)

    model = BigMamba()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    print(f"🧠 BigMamba: {sum(p.numel() for p in model.parameters()):,} params\n")

    # Load data
    with open("checkpoints/clean-v1/train_data.jsonl") as f:
        data = [json.loads(line) for line in f]
    print(f"📚 Training samples: {len(data)}\n")

    tokenized = [tok.encode(d["text"]).ids[:128] for d in data if len(tok.encode(d["text"]).ids) >= 10]
    print(f"📝 Tokenized: {len(tokenized)}")

    best_loss = float("inf")
    for ep in range(30):
        random.shuffle(tokenized)
        tl = 0.0; nb = 0
        model.train()
        for ids in tokenized:
            t = torch.tensor([ids[:-1]]).long()
            lm = model.forward_seq(t)
            loss = F.cross_entropy(lm.reshape(-1, V), torch.tensor(ids[1:]).long())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1

        avg = tl / nb
        scheduler.step()

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "checkpoints/big-mamba/model_best.pt")

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1:>2}/30 | loss={avg:.4f} | best={best_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
            torch.save(model.state_dict(), f"checkpoints/big-mamba/model-{ep+1}.pt")
            # gen test
            g = model.generate(tok.encode("[THOUGHT] checking output of: ls\n[CMD] ").ids, 50, 0.85)
            print(f"     gen: {g[:80]}")

    torch.save(model.state_dict(), "checkpoints/big-mamba/model_final.pt")
    print(f"\n✅ Done! Best loss: {best_loss:.4f}")
