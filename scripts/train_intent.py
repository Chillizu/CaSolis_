#!/usr/bin/env python3
"""
把 clean-v3 数据转为意图格式，然后训练 ModelCluster v3

数据格式:
  [THOUGHT] ... [INTENT] intent_name key=val ... [OBS] ...
"""

import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from arch.model_cluster_v3 import ModelClusterV3, V
from arch.intent_translator import parse_intent, intent_to_command, INTENTS, format_intent_text

torch.set_num_threads(4)
os.makedirs("checkpoints/modelcluster-v3", exist_ok=True)

tok = Tokenizer.from_file("checkpoints/general-v1/tokenizer.json")

# ─── 数据转换 ───────────────────────────
print("📂 加载并转换数据为意图格式...", flush=True)
with open("checkpoints/clean-v3/train_data.jsonl") as f:
    data = [json.loads(line) for line in f]

converted = 0
failed = 0
intent_data = []

for d in data:
    text = d["text"]
    # 提取 CMD 和 OBS
    cmd_idx = text.find("[CMD]")
    obs_idx = text.find("[OBS]")
    if cmd_idx < 0:
        failed += 1
        continue

    if obs_idx >= 0:
        cmd_text = text[cmd_idx+5:obs_idx].strip()
        obs_text = text[obs_idx:].strip()
    else:
        cmd_text = text[cmd_idx+5:].strip()
        obs_text = "[OBS] (no output)"

    # 解析为意图
    test_text = f"[THOUGHT] check\n[CMD] {cmd_text}\n{obs_text}"
    result = parse_intent(test_text)

    if result and result[0] in INTENTS:
        intent_name, params = result
        # 生成意图格式文本
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        new_text = text.replace(
            f"[CMD] {cmd_text}",
            f"[INTENT] {intent_name} {param_str}"
        )
        intent_data.append({"text": new_text, "cmd": cmd_text, "intent": intent_name})
        converted += 1
    else:
        # 保留原格式但标注为 UNKNOWN
        failed += 1

print(f"   转换: {converted} 条, 失败: {failed} 条", flush=True)

# 统计意图分布
intent_counts = {}
for s in intent_data:
    intent_counts[s["intent"]] = intent_counts.get(s["intent"], 0) + 1
print(f"   意图分布: {dict(sorted(intent_counts.items(), key=lambda x: -x[1]))}", flush=True)

# Tokenize
all_ids = []
for s in intent_data:
    ids = tok.encode(s["text"]).ids[:120]
    if len(ids) >= 10:
        all_ids.append(ids)

print(f"\n📝 Tokenized: {len(all_ids)} 条", flush=True)

# ─── 模型 ──────────────────────────────
print("\n🧠 ModelCluster v3 + 意图训练...", flush=True)
model = ModelClusterV3(d_model=1024, n_routed=4)
model.init_from_checkpoint("checkpoints/big-mamba/model_best.pt")
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = torch.optim.AdamW([
    {"params": model.shared_expert.parameters(), "lr": 5e-5},
    {"params": model.routed_experts.parameters(), "lr": 1e-4},
    {"params": model.router.parameters(), "lr": 5e-4},
    {"params": model.lm_head.parameters(), "lr": 5e-5},
    {"params": model.world_model.parameters(), "lr": 3e-4},
    {"params": model.confidence_head.parameters(), "lr": 3e-4},
    {"params": model.embed.parameters(), "lr": 5e-5},
], weight_decay=1e-6)

BATCH = 4
EPOCHS = 5

print(f"\n{'─'*45}")
print(f"  训练: {len(all_ids)} 意图样本, {EPOCHS} epochs")
print(f"{'─'*45}\n")

def collate(batch):
    B = len(batch)
    max_l = min(max(len(s) for s in batch), 128)
    padded = torch.full((B, max_l), V + 18, dtype=torch.long)
    for i, s in enumerate(batch):
        l = min(len(s), max_l)
        padded[i, :l] = torch.tensor(s[:l])
    return padded

best_loss = float("inf")
t_start = time.time()

for ep in range(EPOCHS):
    random.shuffle(all_ids)
    total_lm = 0.0
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
        total_l += metrics["load"]
        for k in range(model.n_routed):
            usage_ep[k] += usage[k]
        n_batches += 1

    avg_lm = total_lm / n_batches
    avg_l = total_l / n_batches
    usage_str = " ".join(f"E{k}:{usage_ep[k]/n_batches:.0%}" for k in range(model.n_routed))
    elapsed = time.time() - t_start

    print(f"  ep{ep+1}/{EPOCHS} | LM={avg_lm:.3f} L={avg_l:.3f} | {usage_str} | {int(elapsed//60)}m", flush=True)

    if avg_lm < best_loss:
        best_loss = avg_lm
        torch.save(model.state_dict(), "checkpoints/modelcluster-v3/model_intent_best.pt")

    # 生成测试
    seed = tok.encode("[THOUGHT] check\n[INTENT] ").ids
    with torch.no_grad():
        gen = model.generate(seed, 40, 0.85, explore=True)
        txt = tok.decode(gen)
        print(f"    gen: {txt[:60]}", flush=True)

torch.save(model.state_dict(), "checkpoints/modelcluster-v3/model_intent_final.pt")
total_t = time.time() - t_start
print(f"\n✅ 完成! {int(total_t//60)}m")
