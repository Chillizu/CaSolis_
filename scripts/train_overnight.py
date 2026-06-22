#!/usr/bin/env python3
"""
Batched overnight training: 19.7M Mamba + world model on clean data
batch=8, 30 epochs, checkpoint every 5 epochs

Hardware: CPU (4 threads), ~3-4 min/epoch
Total: ~2 hours
"""

import os, sys, json, random, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_model import MambaBlock

torch.set_num_threads(4)
tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
V = tok.get_vocab_size(); TV = V + 19

class MambaWorld(nn.Module):
    """19.7M Mamba + world model head"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.embed = nn.Embedding(TV, d_model)
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, V)
        self.world_head = nn.Linear(d_model, d_model)

    def forward(self, tokens, output_ids=None):
        """训练前向: 返回 (lm_loss, world_loss)"""
        x = self.embed(tokens)
        y = self.mamba(x)
        y = self.norm(y)
        logits = self.lm_head(y)
        B, S, _ = logits.shape

        # LM loss: predict next token (mask out special tokens >= V)
        targets = tokens[:, 1:].clone()
        mask = targets < V
        lm_loss = F.cross_entropy(
            logits[:, :-1][mask].reshape(-1, V),
            targets[mask].reshape(-1)
        )

        # World loss: predict output embedding from last hidden state
        world_loss = torch.tensor(0.0)
        if output_ids is not None and output_ids.numel() > 2:
            h_last = y[torch.arange(B), (tokens != TV-1).sum(dim=1) - 1]  # last non-pad pos
            pred = self.world_head(h_last)
            target_emb = self.embed(output_ids).mean(dim=1)
            world_loss = F.mse_loss(pred, target_emb)

        return lm_loss, world_loss

    @torch.no_grad()
    def generate(self, seed_ids, n=40, temp=0.85):
        out = list(seed_ids)
        self.eval()
        for _ in range(n):
            x = self.embed(torch.tensor([out[-60:]]).long())
            y = self.norm(self.mamba(x))
            logits = self.lm_head(y)
            lp = F.softmax(logits[0, -1] / temp, dim=-1)
            nt = torch.multinomial(lp, 1).item()
            if nt >= V: break
            out.append(nt)
        self.train()
        return tok.decode(out[len(seed_ids):])


def collate(samples, max_len=128):
    """Pad sequences to same length for batching"""
    B = len(samples)
    lengths = [len(s) for s in samples]
    max_l = min(max(lengths), max_len)
    padded = torch.full((B, max_l), TV - 1, dtype=torch.long)  # pad token = BOS-1
    for i, s in enumerate(samples):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    return padded


if __name__ == "__main__":
    os.makedirs("checkpoints/big-mamba", exist_ok=True)

    model = MambaWorld()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    print(f"🧠 MambaWorld 12.6M: {sum(p.numel() for p in model.parameters()):,} params\n", flush=True)

    # Load & tokenize
    with open("checkpoints/clean-v1/train_data.jsonl") as f:
        data = [json.loads(line) for line in f]
    print(f"📚 Data: {len(data)} samples", flush=True)

    # Tokenize all
    all_ids = []
    all_out_ids = []
    for d in data:
        ids = tok.encode(d["text"]).ids[:120]
        if len(ids) >= 10:
            all_ids.append(ids)
            # Extract output portion (right after [OBS])
            text = d["text"]
            obs_idx = text.find("[OBS]")
            if obs_idx >= 0:
                obs_text = text[obs_idx+5:].strip()
                out_ids = tok.encode(obs_text).ids[:40]
            else:
                out_ids = []
            all_out_ids.append(out_ids)

    print(f"📝 Tokenized: {len(all_ids)} samples", flush=True)
    print(f"    avg len: {sum(len(s) for s in all_ids)/len(all_ids):.0f} tokens\n", flush=True)

    # Training
    BATCH = 8
    best_loss = float("inf")
    t0 = time.time()

    for ep in range(30):
        # Zip and shuffle
        combined = list(zip(all_ids, all_out_ids))
        random.shuffle(combined)
        all_ids_shuf, all_out_ids_shuf = zip(*combined)
        all_ids_shuf, all_out_ids_shuf = list(all_ids_shuf), list(all_out_ids_shuf)

        tl = 0.0; tw = 0.0; nb = 0
        model.train()

        for i in range(0, len(all_ids_shuf), BATCH):
            batch_ids = all_ids_shuf[i:i+BATCH]
            batch_out = all_out_ids_shuf[i:i+BATCH]

            tokens = collate(batch_ids)
            out_tokens = collate(batch_out, max_len=40) if any(o for o in batch_out) else None

            lm_loss, world_loss = model(tokens, out_tokens)
            loss = lm_loss + 0.3 * world_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tl += lm_loss.item()
            if isinstance(world_loss, torch.Tensor) and world_loss > 0:
                tw += world_loss.item()
            nb += 1



        avg_lm = tl / nb
        avg_w = tw / nb
        scheduler.step()

        if avg_lm < best_loss:
            best_loss = avg_lm
            torch.save(model.state_dict(), "checkpoints/big-mamba/model_best.pt")

        if (ep + 1) % 5 == 0 or ep == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (30 - ep - 1) if ep > 0 else elapsed * 30
            print(f"  ep{ep+1:>2}/30 | LM={avg_lm:.4f} W={avg_w:.4f} best={best_loss:.4f} | {int(elapsed//60)}m ETA:{int(eta//60)}m", flush=True)
            # Save + test
            torch.save(model.state_dict(), f"checkpoints/big-mamba/model-{ep+1}.pt")
            g = model.generate(tok.encode("[THOUGHT] checking\n[CMD] ").ids, 50, 0.85)
            print(f"     gen: {g[:80]}", flush=True)

    # Final
    torch.save(model.state_dict(), "checkpoints/big-mamba/model_final.pt")
    total = time.time() - t0
    print(f"\n✅ Done! {int(total//60)}m total", flush=True)
    print(f"   Best LM loss: {best_loss:.4f}", flush=True)
    print(f"   Final: checkpoints/big-mamba/model_final.pt", flush=True)
