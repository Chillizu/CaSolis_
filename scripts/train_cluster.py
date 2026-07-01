#!/usr/bin/env python3
"""
ModelCluster 训练脚本

阶段 1: 联合训练（所有专家 + 路由器一起训）
数据: 1500 条干净数据
目标: 路由器学会「哪个专家适合哪个任务」
"""

import os, sys, json, random, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.model_cluster import ModelCluster, V

torch.set_num_threads(4)
os.makedirs("checkpoints/modelcluster", exist_ok=True)

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")

print("📂 加载数据...", flush=True)
with open("checkpoints/clean-v1/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

all_ids = [tok.encode(d["text"]).ids[:120] for d in data if len(tok.encode(d["text"]).ids) >= 10]
print(f"   {len(all_ids)} 条样本", flush=True)

# 模型
print("\n🧠 构建 ModelCluster...", flush=True)
model = ModelCluster(d_model=1024, n_experts=4)
model.init_from_checkpoint("checkpoints/big-mamba/model_best.pt")
total = sum(p.numel() for p in model.parameters())
print(f"   总参数: {total:,} ({total/1e6:.1f}M)", flush=True)

# 优化器（各组件不同学习率）
opt = torch.optim.AdamW([
    {"params": model.embed.parameters(), "lr": 1e-4},
    {"params": model.experts.parameters(), "lr": 2e-4},
    {"params": model.router.parameters(), "lr": 5e-4},  # 路由器学习率更高
    {"params": model.lm_head.parameters(), "lr": 1e-4},
    {"params": model.world_head.parameters(), "lr": 2e-4},
    {"params": model.confidence_head.parameters(), "lr": 3e-4},
], weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
BATCH = 4

print(f"\n{'─'*45}")
print(f"  训练: {len(all_ids)} samples, batch={BATCH}, 10 epochs")
print(f"{'─'*45}\n")

best_loss = float("inf")
t_start = time.time()

for ep in range(10):
    random.shuffle(all_ids)
    total_loss_ep = 0.0
    total_lm = 0.0
    total_world = 0.0
    total_conf = 0.0
    total_load = 0.0
    usage_ep = [0.0] * 4
    n_batches = 0

    model.train()

    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i:i+BATCH]
        # Pad
        max_l = min(max(len(s) for s in batch), 128)
        padded = torch.full((len(batch), max_l), V + 18, dtype=torch.long)
        for j, s in enumerate(batch):
            l = min(len(s), max_l)
            padded[j, :l] = torch.tensor(s[:l])

        # 探索率随训练衰减
        explore_rate = max(0.3 - ep * 0.03, 0.05)

        loss, metrics, usage = model(padded, explore_rate=explore_rate)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss_ep += loss.item()
        total_lm += metrics["lm"]
        total_world += metrics["world"]
        total_conf += metrics["conf"]
        total_load += metrics["load_balance"]
        for k in range(4):
            usage_ep[k] += usage[k]
        n_batches += 1

    avg_loss = total_loss_ep / n_batches
    avg_lm = total_lm / n_batches
    avg_world = total_world / n_batches
    avg_conf = total_conf / n_batches
    avg_load = total_load / n_batches
    usage_str = " ".join(f"E{k}:{usage_ep[k]/n_batches:.1%}" for k in range(4))

    scheduler.step()

    elapsed = time.time() - t_start
    eta = (elapsed / (ep + 1)) * (10 - ep - 1)

    print(f"  ep{ep+1:>2}/10 | LM={avg_lm:.3f} W={avg_world:.3f} | "
          f"{usage_str} | {int(elapsed//60)}m ETA:{int(eta//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/modelcluster/model_best.pt")
        print(f"    💾 新最佳模型!", flush=True)

    if (ep + 1) % 2 == 0 or ep == 0:
        torch.save(model.state_dict(), f"checkpoints/modelcluster/model-{ep+1}.pt")

        # 生成样本
        with torch.no_grad():
            seed = tok.encode("[THOUGHT] check\n[CMD] ").ids
            gen = model.generate(seed, 50, 0.85, explore=True)
            txt = tok.decode(gen)
            print(f"     gen: {txt[:70]}", flush=True)

torch.save(model.state_dict(), "checkpoints/modelcluster/model_final.pt")
total_t = time.time() - t_start
print(f"\n✅ 训练完成! {int(total_t//60)}m")
print(f"   最佳 LM loss: {best_loss:.4f}")
