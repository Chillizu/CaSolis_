#!/usr/bin/env python3
"""
MambaMTP training: Multi-Token Prediction + JEPA world model
Loads from existing checkpoint to warm-start the trunk
"""

import os, sys, json, random, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.mamba_mtp import MambaMTP, V, TV, PAD

torch.set_num_threads(4)
tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")


def load_and_migrate(model, ckpt_path):
    """Load old single-head checkpoint and migrate to MTP"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ms = model.state_dict()

    # Copy matching keys (embed, mamba, norm)
    loaded = 0
    for k in ms:
        if k in sd and ms[k].shape == sd[k].shape:
            ms[k] = sd[k]
            loaded += 1

    # Copy old lm_head to all 4 MTP heads
    if "lm_head.weight" in sd and sd["lm_head.weight"].shape == model.lm_heads[0].weight.shape:
        w = sd["lm_head.weight"]
        b = sd.get("lm_head.bias", None)
        for i, head in enumerate(model.lm_heads):
            head.weight.data.copy_(w)
            if b is not None:
                head.bias.data.copy_(b)

    # Copy world_head
    if "world_head.weight" in sd and sd["world_head.weight"].shape == model.world_head.weight.shape:
        model.world_head.weight.data.copy_(sd["world_head.weight"])
        if "world_head.bias" in sd:
            model.world_head.bias.data.copy_(sd["world_head.bias"])

    model.load_state_dict(ms, strict=False)
    return loaded


def collate(samples, max_len=128):
    B = len(samples)
    lengths = [len(s) for s in samples]
    max_l = min(max(lengths), max_len)
    padded = torch.full((B, max_l), PAD, dtype=torch.long)
    for i, s in enumerate(samples):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    return padded


# === Data ===
print("📂 Loading data...", flush=True)
with open("checkpoints/clean-v1/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

all_ids = []
all_out_ids = []
for d in data:
    text = d["text"]
    ids = tok.encode(text).ids[:120]
    if len(ids) >= 15:
        all_ids.append(ids)
        obs_idx = text.find("[OBS]")
        if obs_idx >= 0:
            obs_text = text[obs_idx+5:].strip()
            out_ids = tok.encode(obs_text).ids[:40]
        else:
            out_ids = []
        all_out_ids.append(out_ids)

print(f"   {len(all_ids)} samples ({sum(len(s) for s in all_ids)//len(all_ids)} avg tokens)", flush=True)

# === Model ===
print("🧠 Building MambaMTP (4 heads + world)...", flush=True)
model = MambaMTP(d_model=1024, n_pred=4)

# Try loading checkpoint
ckpt = "checkpoints/big-mamba/model_best.pt"
if os.path.exists(ckpt):
    n = load_and_migrate(model, ckpt)
    print(f"   Loaded checkpoint: {n} trunk keys + 4 heads from lm_head", flush=True)
else:
    print("   No checkpoint found, starting from scratch", flush=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"   {total_params:,} params", flush=True)

opt = torch.optim.AdamW(
    [
        {"params": model.embed.parameters(), "lr": 1e-4},
        {"params": model.mamba.parameters(), "lr": 2e-4},
        {"params": model.norm.parameters(), "lr": 2e-4},
        {"params": model.lm_heads.parameters(), "lr": 3e-4},  # heads: higher LR
        {"params": model.world_head.parameters(), "lr": 3e-4},
    ],
    weight_decay=1e-5,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

BATCH = 4
print(f"\n{'─'*45}")
print(f"  Training: {len(all_ids)} samples, batch={BATCH}")
print(f"  30 epochs, ~{120*len(all_ids)//BATCH//60//len(all_ids)*len(all_ids)} min total")
print(f"{'─'*45}\n")

best_loss = float("inf")
t_start = time.time()

for ep in range(30):
    # Shuffle
    combined = list(zip(all_ids, all_out_ids))
    random.shuffle(combined)
    ids_shuf, out_shuf = zip(*combined)

    total_lm = 0.0
    total_w = 0.0
    total_heads = [0.0] * model.n_pred
    n_batches = 0
    model.train()

    for i in range(0, len(ids_shuf), BATCH):
        batch = ids_shuf[i:i+BATCH]
        batch_out = out_shuf[i:i+BATCH]

        tokens = collate(batch)
        out_pad = None
        if any(o for o in batch_out):
            om = min(max(len(s) for s in batch_out if s), 40)
            out_pad = torch.full((len(batch_out), om), PAD, dtype=torch.long)
            for j, s in enumerate(batch_out):
                if s:
                    l = min(len(s), om)
                    out_pad[j, :l] = torch.tensor(s[:l])

        lm_loss, w_loss, head_l = model(tokens, out_pad)
        loss = lm_loss + 0.3 * w_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_lm += lm_loss.item()
        total_w += w_loss.item() if isinstance(w_loss, torch.Tensor) else 0
        for hi, hl in enumerate(head_l):
            if hi < len(total_heads):
                total_heads[hi] += hl
        n_batches += 1

    avg_lm = total_lm / n_batches
    avg_w = total_w / n_batches
    avg_heads = [h / n_batches for h in total_heads]

    scheduler.step()

    elapsed = time.time() - t_start
    eta = elapsed / (ep + 1) * (30 - ep - 1)
    head_str = " ".join(f"H{i}={h:.3f}" for i, h in enumerate(avg_heads))

    print(f"  ep{ep+1:>2}/30 | LM={avg_lm:.3f} W={avg_w:.3f} | {head_str} | {int(elapsed//60)}m ETA:{int(eta//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/mtp/model_best.pt")

    if (ep + 1) % 5 == 0 or ep == 0:
        torch.save(model.state_dict(), f"checkpoints/mtp/model-{ep+1}.pt")

        with torch.no_grad():
            seed = tok.encode("[THOUGHT] check\n[CMD] ").ids
            gen = model.generate(seed, 50, 0.85)
            txt = tok.decode(gen)
            print(f"     gen: {txt[:80]}", flush=True)

torch.save(model.state_dict(), "checkpoints/mtp/model_final.pt")
total = time.time() - t_start
print(f"\n✅ Done! {int(total//60)}m total")
print(f"   Best LM: {best_loss:.4f}")
print(f"   Checkpoints in checkpoints/mtp/")
