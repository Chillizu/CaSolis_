#!/usr/bin/env python3
"""
ModelCluster v3 训练 — 共享专家 + 负载均衡 + 可扩展
"""

import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from tokenizers import Tokenizer
from arch.model_cluster_v3 import ModelClusterV3, V

torch.set_num_threads(4)
os.makedirs("checkpoints/modelcluster-v3", exist_ok=True)

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")

print("📂 加载数据...", flush=True)
with open("checkpoints/clean-v3/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

all_ids = [tok.encode(d["text"]).ids[:120] for d in data if len(tok.encode(d["text"]).ids) >= 10]
print(f"   {len(all_ids)} 条样本", flush=True)

print("\n🧠 构建 ModelCluster v3...", flush=True)
model = ModelClusterV3(d_model=1024, n_routed=4)
model.init_from_checkpoint("checkpoints/big-mamba/model_best.pt")
total = sum(p.numel() for p in model.parameters())
print(f"   总参数: {total:,} ({total/1e6:.1f}M)", flush=True)

opt = torch.optim.AdamW([
    {"params": model.shared_expert.parameters(), "lr": 5e-5},
    {"params": model.routed_experts.parameters(), "lr": 1e-4},
    {"params": model.router.parameters(), "lr": 5e-4},
    {"params": model.lm_head.parameters(), "lr": 5e-5},
    {"params": model.world_model.parameters(), "lr": 3e-4},
    {"params": model.confidence_head.parameters(), "lr": 3e-4},
    {"params": model.embed.parameters(), "lr": 5e-5},
], weight_decay=1e-6)

def collate(batch):
    B = len(batch)
    max_l = min(max(len(s) for s in batch), 128)
    padded = torch.full((B, max_l), V + 18, dtype=torch.long)
    for i, s in enumerate(batch):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    return padded

BATCH = 4
EPOCHS = 5
print(f"\n{'─'*45}")
print(f"  训练: {len(all_ids)} samples, batch={BATCH}, {EPOCHS} epochs")
print(f"{'─'*45}\n")

best_loss = float("inf")
t_start = time.time()

for ep in range(EPOCHS):
    random.shuffle(all_ids)
    total_lm = 0.0
    total_w = 0.0
    total_l = 0.0
    usage_ep = [0.0] * model.n_routed
    n_batches = 0
    model.train()

    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i:i+BATCH]
        tokens = collate(batch)

        explore = max(0.3 - ep * 0.05, 0.05)
        loss, metrics, usage = model(tokens, explore_rate=explore)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_lm += metrics["lm"]
        total_w += metrics["world"]
        total_l += metrics["load"]
        for k in range(model.n_routed):
            usage_ep[k] += usage[k]
        n_batches += 1

    avg_lm = total_lm / n_batches
    avg_w = total_w / n_batches
    avg_l = total_l / n_batches
    usage_str = " ".join(f"E{k}:{usage_ep[k]/n_batches:.0%}" for k in range(model.n_routed))
    elapsed = time.time() - t_start

    print(f"  ep{ep+1}/{EPOCHS} | LM={avg_lm:.3f} W={avg_w:.3f} L={avg_l:.3f} | {usage_str} | {int(elapsed//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/modelcluster-v3/model_best.pt")

    # 生成测试
    seed = tok.encode("[THOUGHT] check\n[CMD] ").ids
    with torch.no_grad():
        gen = model.generate(seed, 40, 0.85, explore=True)
        txt = tok.decode(gen)
        print(f"    gen: {txt[:60]}", flush=True)

torch.save(model.state_dict(), "checkpoints/modelcluster-v3/model_final.pt")
total_t = time.time() - t_start
print(f"\n✅ 完成! {int(total_t//60)}m")
print(f"   最佳 LM: {best_loss:.4f}")
print(f"   使用率: {usage_str}")
