#!/usr/bin/env python3
"""
ModelCluster — 专家分工微调
让每个专家只做自己擅长的事，路由器学会选人
"""

import os, sys, json, random, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer
from arch.model_cluster import ModelCluster, V

torch.set_num_threads(4)
os.makedirs("checkpoints/modelcluster", exist_ok=True)

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")
device = torch.device("cpu")

# 加载数据
print("📂 加载数据...", flush=True)
with open("checkpoints/clean-v1/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

all_ids = []
all_out_ids = []
all_diversity = []
for d in data:
    text = d["text"]
    ids = tok.encode(text).ids[:120]
    if len(ids) < 10:
        continue
    all_ids.append(ids)
    # Extract output portion for world model
    obs_idx = text.find("[OBS]")
    if obs_idx >= 0:
        obs_text = text[obs_idx+5:].strip()
        out_ids = tok.encode(obs_text).ids[:40]
    else:
        out_ids = []
    all_out_ids.append(out_ids)
    # Diversity score: unique tokens / total tokens
    uniq = len(set(ids))
    all_diversity.append(uniq / max(len(ids), 1))

print(f"   {len(all_ids)} 条样本 (avg diversity: {sum(all_diversity)/len(all_diversity):.2f})", flush=True)

# 模型
print("\n🧠 加载 ModelCluster...", flush=True)
model = ModelCluster(d_model=1024, n_experts=4)
ckpt = "checkpoints/modelcluster/model_best.pt"
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    print("   ✅ 已加载 checkpoint", flush=True)
else:
    model.init_from_checkpoint("checkpoints/big-mamba/model_best.pt")
    print("   ⚠️ 从 base checkpoint 初始化", flush=True)

total = sum(p.numel() for p in model.parameters())
print(f"   总参数: {total:,} ({total/1e6:.1f}M)", flush=True)

# 优化器
opt = torch.optim.AdamW([
    {"params": model.experts.parameters(), "lr": 1e-4},  # 专家学习率较低（微调）
    {"params": model.router.parameters(), "lr": 3e-4},   # 路由器学快点
    {"params": model.lm_head.parameters(), "lr": 5e-5},
    {"params": model.world_head.parameters(), "lr": 1e-4},
    {"params": model.confidence_head.parameters(), "lr": 1e-4},
    {"params": model.embed.parameters(), "lr": 5e-5},
], weight_decay=1e-6)

BATCH = 4
PAD = V + 18

def collate(samples, out_samples, max_len=128):
    """Collate with output targets"""
    B = len(samples)
    max_l = min(max(len(s) for s in samples), max_len)
    padded = torch.full((B, max_l), PAD, dtype=torch.long)
    for i, s in enumerate(samples):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    
    # Output IDs for world model
    out_padded = None
    if out_samples and any(o for o in out_samples):
        om = min(max(len(s) for s in out_samples if s), 30)
        if om > 0:
            out_padded = torch.full((B, om), PAD, dtype=torch.long)
            for i, s in enumerate(out_samples):
                if s:
                    l = min(len(s), om)
                    out_padded[i, :l] = torch.tensor(s[:l])
    return padded, out_padded


print(f"\n{'─'*45}")
print("  专家分工微调 — 5 epochs")
print(f"{'─'*45}\n")

best_loss = float("inf")
t_start = time.time()

for ep in range(5):
    # 每个 epoch shuffle
    combined = list(zip(all_ids, all_out_ids, all_diversity))
    random.shuffle(combined)
    ids_shuf, out_shuf, div_shuf = zip(*combined)

    total_loss_ep = 0.0
    total_lm = 0.0
    total_specialized = [0.0] * 4
    usage_ep = [0.0] * 4
    n_batches = 0

    model.train()

    for i in range(0, len(ids_shuf), BATCH):
        batch = ids_shuf[i:i+BATCH]
        batch_out = out_shuf[i:i+BATCH]
        batch_div = torch.tensor(div_shuf[i:i+BATCH])

        tokens, out_tokens = collate(batch, batch_out)
        B, S = tokens.shape
        d = model.d_model

        # Embed
        x = model.embed(tokens)

        # 各专家前向
        expert_out = []
        for expert in model.experts:
            expert_out.append(expert(x))
        expert_out = torch.stack(expert_out, dim=1)  # (B, K, S, d)

        # 逐 step 训练
        last_expert_onehot = torch.zeros(B, 4)
        total_loss = torch.tensor(0.0, requires_grad=True)
        expert_counts = torch.zeros(4)

        for t in range(S - 1):
            h_t = expert_out[:, :, t]  # (B, K, d)
            target = tokens[:, t + 1]

            # 路由
            explore = random.random() < max(0.2 - ep * 0.03, 0.05)
            route = model.router(x[:, t], last_expert_onehot if t == 0 else route.detach(), explore=explore)

            # 专家分工 loss 计算
            # E0 (命令): 标准 LM loss
            logits_e0 = model.lm_head(h_t[:, 0])
            valid0 = target < V
            loss_e0 = F.cross_entropy(logits_e0[valid0], target[valid0]) if valid0.sum() > 0 else torch.tensor(0.0)

            # E1 (创造): LM loss + 多样性奖励（低多样性样本加惩罚，高多样性样本减惩罚）
            logits_e1 = model.lm_head(h_t[:, 1])
            valid1 = target < V
            loss_e1_lm = F.cross_entropy(logits_e1[valid1], target[valid1]) if valid1.sum() > 0 else torch.tensor(0.0)
            diversity_bonus = batch_div.mean().item() * 0.3  # 鼓励多样性
            loss_e1 = loss_e1_lm - diversity_bonus

            # E2 (世界): World loss，不训 LM
            h_last_e2 = h_t[:, 2]
            pred_emb = model.world_head(h_last_e2)
            if out_tokens is not None and out_tokens.numel() > 2:
                out_emb = model.embed(out_tokens).mean(dim=1)
                loss_e2 = F.mse_loss(pred_emb, out_emb.detach())
            else:
                # Fallback: 预测下一个 token 的嵌入
                next_target = target.clamp(0, V-1)
                next_emb = model.embed(next_target)
                loss_e2 = 0.5 * F.mse_loss(pred_emb, next_emb.detach())

            # E3 (评估): 自信度 + 路由准确度
            conf = model.confidence_head(h_t[:, 3]).squeeze(-1)
            # 自信度目标 = 路由权重 × (1 - 该专家的 LM loss)
            # 简单版：如果这个专家的 LM loss 低，自信度就应该高
            logits_e3 = model.lm_head(h_t[:, 3])
            valid3 = target < V
            if valid3.sum() > 0:
                e3_lm = F.cross_entropy(logits_e3[valid3], target[valid3], reduction='none')
                conf_target = 1.0 / (1.0 + e3_lm.detach())
                loss_e3 = F.mse_loss(conf[:len(conf_target)], conf_target)
            else:
                loss_e3 = torch.tensor(0.0)

            # 汇总: 路由加权
            loss_step = (route[:, 0].detach().mean() * loss_e0 +
                         route[:, 1].detach().mean() * loss_e1 +
                         route[:, 2].detach().mean() * loss_e2 +
                         route[:, 3].detach().mean() * loss_e3)

            # Load balancing
            expert_choice = route.argmax(dim=1)
            for b in range(B):
                expert_counts[expert_choice[b]] += 1

            total_loss = total_loss + loss_step

        # Load balance penalty
        target_dist = torch.full((4,), 1.0/4)
        actual_dist = expert_counts / (expert_counts.sum() + 1e-8)
        load_loss = -0.05 * torch.sum(target_dist * torch.log(actual_dist + 1e-8))

        loss = total_loss / (S-1) + load_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 统计
        total_loss_ep += loss.item()
        total_lm += (loss_e0.item() + loss_e1_lm.item()) / 2
        total_specialized[0] += loss_e0.item()
        total_specialized[1] += loss_e1.item()
        total_specialized[2] += loss_e2.item()
        total_specialized[3] += loss_e3.item() if isinstance(loss_e3, torch.Tensor) else 0
        for k in range(4):
            usage_ep[k] += (route[:, k].detach().mean().item() * 100)
        n_batches += 1

    avg_loss = total_loss_ep / n_batches
    avg_lm = total_lm / n_batches
    usage_str = " ".join(f"E{k}:{usage_ep[k]/n_batches:.0f}%" for k in range(4))
    spec_str = " ".join(f"E{k}:{v/n_batches:.3f}" for k, v in enumerate(total_specialized))

    elapsed = time.time() - t_start
    print(f"  ep{ep+1}/5 | LM={avg_lm:.3f} | {usage_str} | {int(elapsed//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/modelcluster/model_specialized.pt")
        print(f"    💾 保存最佳!", flush=True)

    # 生成测试
    seed = tok.encode("[THOUGHT] check\n[CMD] ").ids
    with torch.no_grad():
        for explore in [True, False]:
            gen = model.generate(seed, 50, 0.85, explore=explore)
            txt = tok.decode(gen)
            mode = "探索" if explore else "利用"
            print(f"    {mode}: {txt[:60]}", flush=True)

total_t = time.time() - t_start
print(f"\n✅ 分工微调完成! {int(total_t//60)}m")
print(f"   最佳 loss: {best_loss:.4f}")
print(f"   最终使用率: {usage_str}")
