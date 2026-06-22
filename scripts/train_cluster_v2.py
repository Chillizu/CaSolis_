#!/usr/bin/env python3
"""
ModelCluster 在 clean-v3 上从头训练
高质量数据（Docker 验证）+ 多样化数据
"""

import os, sys, json, random, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.model_cluster import ModelCluster, V, TV

torch.set_num_threads(4)
os.makedirs("checkpoints/modelcluster-v2", exist_ok=True)

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
PAD = TV - 1

print("📂 加载数据...", flush=True)
with open("checkpoints/clean-v3/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

all_ids = [tok.encode(d["text"]).ids[:120] for d in data if len(tok.encode(d["text"]).ids) >= 10]
print(f"   {len(all_ids)} 条样本", flush=True)

# 模型
print("\n🧠 构建 ModelCluster v2...", flush=True)
model = ModelCluster(d_model=1024, n_experts=4)
model.init_from_checkpoint("checkpoints/big-mamba/model_best.pt")
print(f"   总参数: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# 优化器
opt = torch.optim.AdamW([
    {"params": model.embed.parameters(), "lr": 8e-5},
    {"params": model.experts.parameters(), "lr": 1e-4},
    {"params": model.router.parameters(), "lr": 5e-4},
    {"params": model.lm_head.parameters(), "lr": 5e-5},
    {"params": model.world_head.parameters(), "lr": 1e-4},
    {"params": model.confidence_head.parameters(), "lr": 3e-4},
], weight_decay=1e-6)

def collate(batch):
    B = len(batch)
    max_l = min(max(len(s) for s in batch), 128)
    padded = torch.full((B, max_l), PAD, dtype=torch.long)
    for i, s in enumerate(batch):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    return padded

BATCH = 4
print(f"\n{'─'*45}")
print(f"  训练: {len(all_ids)} samples, batch={BATCH}, 5 epochs")
print(f"{'─'*45}\n")

best_loss = float("inf")
t_start = time.time()

for ep in range(5):
    random.shuffle(all_ids)
    total_loss_ep = 0.0
    total_lm = 0.0
    usage_ep = [0.0] * 4
    n_batches = 0

    model.train()

    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i:i+BATCH]
        tokens = collate(batch)
        B, S = tokens.shape

        # Embed
        x = model.embed(tokens)

        # 各专家前向
        expert_out = []
        for expert in model.experts:
            expert_out.append(expert(x))
        expert_out = torch.stack(expert_out, dim=1)

        # 逐 step
        total_loss = torch.tensor(0.0)
        expert_counts = torch.zeros(4)
        lm_avg = 0.0
        n_steps = 0

        for t in range(S - 1):
            h_t = expert_out[:, :, t]
            target = tokens[:, t + 1]

            explore = random.random() < max(0.3 - ep * 0.05, 0.05)
            route = model.router(x[:, t],
                                 None if t == 0 else route.detach(),
                                 explore=explore)

            # 计算各专家的 loss
            losses_k = []
            for k in range(model.n_experts):
                logits_k = model.lm_head(h_t[:, k])
                valid = target < V
                if valid.sum() > 0:
                    losses_k.append(F.cross_entropy(logits_k[valid], target[valid]))
                else:
                    losses_k.append(torch.tensor(0.0))

            # 路由加权
            loss_t = sum(route[:, k].detach().mean() * losses_k[k]
                        for k in range(model.n_experts))

            # 世界 loss (E2)
            if model.n_experts >= 3:
                e2_h = h_t[:, 2]
                pred_emb = model.world_head(e2_h)
                next_target = target.clamp(0, V-1)
                next_emb = model.embed(next_target)
                loss_t = loss_t + 0.3 * F.mse_loss(pred_emb, next_emb.detach())

            # 路由计数
            choices = route.argmax(dim=1)
            for b in range(B):
                expert_counts[choices[b]] += 1

            total_loss = total_loss + loss_t
            lm_avg += losses_k[0].item()
            n_steps += 1

        # Load balance
        target_dist = torch.full((4,), 1.0/4)
        actual_dist = expert_counts / (expert_counts.sum() + 1e-8)
        load_loss = 0.05 * torch.sum(target_dist * torch.log(actual_dist + 1e-8))

        loss = total_loss / n_steps - load_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss_ep += loss.item()
        total_lm += lm_avg / n_steps
        for k in range(4):
            usage_ep[k] += expert_counts[k].item() / (n_steps * B)
        n_batches += 1

    avg_loss = total_loss_ep / n_batches
    avg_lm = total_lm / n_batches
    usage_str = " ".join(f"E{k}:{usage_ep[k]/n_batches:.0%}" for k in range(4))
    elapsed = time.time() - t_start

    print(f"  ep{ep+1}/5 | LM={avg_lm:.3f} | {usage_str} | {int(elapsed//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/modelcluster-v2/model_best.pt")

    # 生成测试
    seed = tok.encode("[THOUGHT] check\n[CMD] ").ids
    with torch.no_grad():
        gen = model.generate(seed, 40, 0.85, explore=True)
        txt = tok.decode(gen)
        print(f"    gen: {txt[:60]}", flush=True)

torch.save(model.state_dict(), "checkpoints/modelcluster-v2/model_final.pt")
total_t = time.time() - t_start
print(f"\n✅ 训练完成! {int(total_t//60)}m")
print(f"   最佳 LM: {best_loss:.4f}")
print(f"   最终使用率: {usage_str}")
