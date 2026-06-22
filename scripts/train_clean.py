#!/usr/bin/env python3
"""
Train Mamba from scratch on clean data.
No pretrained checkpoint — start with random weights.
"""

import os, sys, json, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_model import MambaBlock

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); BOS = V+16; TV = V+19

class MambaThought(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TV, 768)
        self.mamba = MambaBlock(768)
        self.norm = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, V)

    def forward_seq(self, tokens):
        x = self.embed(tokens)
        y = self.mamba(x)
        y = self.norm(y)
        return self.lm_head(y)

    def generate(self, ctx, n=30, temp=0.85):
        out = list(ctx)
        with torch.no_grad():
            for _ in range(n):
                lm = self.forward_seq(torch.tensor([out[-60:]]).long())
                lp = F.softmax(lm[0, -1] / temp, dim=-1)
                nt = torch.multinomial(lp, 1).item()
                if nt >= V: break
                out.append(nt)
        return tok.decode(out[len(ctx):])


def tokenize(text, max_len=128):
    ids = tok.encode(text).ids[:max_len]
    return ids


if __name__ == "__main__":
    os.makedirs("checkpoints/clean-v1", exist_ok=True)

    model = MambaThought()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)  # higher LR for from-scratch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    print(f"🧠 Fresh Mamba: {sum(p.numel() for p in model.parameters()):,} params\n")

    # Load data
    with open("checkpoints/clean-v1/train_data.jsonl") as f:
        data = [json.loads(line) for line in f]
    print(f"📚 Training samples: {len(data)}\n")

    # Tokenize all data
    tokenized = []
    for d in data:
        ids = tokenize(d["text"])
        if len(ids) >= 10:
            tokenized.append(ids)
    print(f"📝 Tokenized samples: {len(tokenized)}")

    # Train
    best_loss = float("inf")
    for ep in range(20):
        random.shuffle(tokenized)
        total_loss = 0.0
        batches = 0
        model.train()

        for ids in tokenized:
            t = torch.tensor([ids[:-1]]).long()
            target = torch.tensor(ids[1:]).long()
            lm = model.forward_seq(t)
            loss = F.cross_entropy(lm.reshape(-1, V), target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/clean-v1/model_best.pt")

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1:>2}/30 | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
            # Quick generation test
            model.eval()
            with torch.no_grad():
                g = model.generate(tok.encode("[THOUGHT] checking output of: ").ids, 40, 0.85)
                print(f"     gen: {g[:80]}")
            model.train()

    torch.save(model.state_dict(), "checkpoints/clean-v1/model_final.pt")
    print(f"\n✅ Done! Best loss: {best_loss:.4f}")

    # Final generation test
    print("\n🧪 Final generation tests:")
    model.eval()
    with torch.no_grad():
        for seed in ["[THOUGHT] checking\n[CMD] ", "[THOUGHT] seeing\n[CMD] "]:
            g = model.generate(tok.encode(seed).ids, 50, 0.85)
            print(f"  {repr(seed[:20])} → {repr(g[:80])}")
