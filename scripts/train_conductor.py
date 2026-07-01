"""
训练指挥家 (Conductor) — 直接分类法

ConductorHead: MiniLM → 64隐藏 → 16想法向量 + 11分类
训练: 交叉熵 (同现有 11 类分类器)
想法向量不参与 CE 损失, 但共享隐藏层 → 自然学到有语义的表征

训练后: 
  - 想法向量用于保姆翻译
  - 分类 logits 用于 A/B 对比
"""

import sys, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from agent.conductor import Conductor, ConductorHead, N_DIMS, INTENTS

# ── 配置 ──────────────────────────────────────────────────────
DATA_PATH = "data/intent_train_v3.jsonl"
CHECKPOINT_PATH = "checkpoints/conductor/head.pt"
N_EPOCHS = 50
BATCH_SIZE = 128
LR = 3e-4

# ── 加载数据 ──────────────────────────────────────────────────
print(f"加载数据: {DATA_PATH}")
with open(DATA_PATH) as f:
    rows = [json.loads(line) for line in f]
rows = [r for r in rows if r["intent"] in INTENTS]
print(f"有效样本: {len(rows)}")

from collections import Counter
dist = Counter(r["intent"] for r in rows)
for n, c in dist.most_common():
    print(f"  {n:15s} {c}")

# ── 初始化 Conductor ─────────────────────────────────────────
print("\n初始化 Conductor...")
student = Conductor(device="cpu")
optimizer = torch.optim.AdamW(student.head.parameters(), lr=LR, weight_decay=1e-4)

encoder = student.encoder

# ── 预编码 ────────────────────────────────────────────────────
print("预编码所有样本...")
states = [r["state_text"] for r in rows]
embs = encoder.encode(states, convert_to_tensor=True).clone()
target_idxs = torch.tensor([INTENTS.index(r["intent"]) for r in rows])

# ── 训练 ──────────────────────────────────────────────────────
print(f"\n训练 {N_EPOCHS} 轮 (batch={BATCH_SIZE}, lr={LR})...")
start = time.time()
best_loss = float("inf")

for epoch in range(N_EPOCHS):
    perm = torch.randperm(len(rows))
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    student.head.train()
    for i in range(0, len(rows), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE].tolist()
        batch_emb = embs[idx].to(student.device)
        targets = target_idxs[idx].to(student.device)

        thought, pred_logits = student.forward_emb(batch_emb)

        # 分类损失 (主要)
        ce = F.cross_entropy(pred_logits, targets)

        # 对比损失 (SupCon): 同类余弦相似度接近 1, 异类接近 -1
        # 归一化想法向量到单位球面
        thought_norm = F.normalize(thought, dim=-1)  # (B, 16)
        cos_sim = thought_norm @ thought_norm.T  # (B, B) 余弦相似度矩阵
        
        targets_exp = targets.unsqueeze(1)
        same_mask = (targets_exp == targets_exp.T).float()  # (B, B)
        
        # 对每个锚点: 同类相似度尽可能高 (>0.8), 异类尽可能低 (<-0.2)
        pos_loss = (same_mask * F.relu(0.8 - cos_sim)).sum() / (same_mask.sum() + 1)
        neg_loss = ((1 - same_mask) * F.relu(cos_sim + 0.2)).sum() / ((1 - same_mask).sum() + 1)
        
        contrastive = pos_loss + neg_loss

        total = ce + 0.05 * contrastive

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_acc += (pred_logits.argmax(dim=-1) == targets).float().mean().item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    # 验证 (200 样本)
    student.head.eval()
    with torch.no_grad():
        val_thought, val_logits = student.forward_emb(embs[:200])
        val_acc = (val_logits.argmax(dim=-1) == target_idxs[:200].to(student.device)).float().mean().item()
        dim_std = val_thought.std(dim=0).mean().item()

    if avg_loss < best_loss:
        best_loss = avg_loss
        student.save(CHECKPOINT_PATH)

    elapsed = time.time() - start
    dt_per_epoch = elapsed / (epoch + 1)
    remaining = dt_per_epoch * (N_EPOCHS - epoch - 1)
    print(f"  [{epoch+1:2d}/{N_EPOCHS}] loss={avg_loss:.4f}  acc={avg_acc:.0%}  val_acc={val_acc:.0%}  dim_std={dim_std:.3f}  ({elapsed:.0f}s, ~{remaining:.0f}s)")

# ── 最终验证 ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"最终验证...")
student.load(CHECKPOINT_PATH)
student.head.eval()

test_cases = [
    ("当前目录: /etc 已知文件: passwd 上步: 读取 /etc/passwd 的内容 历史: 无", "READ"),
    ("当前目录: / 已知文件:  上步: 获取当前时间 历史: 无", "INFO"),
    ("当前目录: / 已知文件:  上步: 查看 CPU 信息 历史: 无", "INFO"),
    ("当前目录: / 已知文件: etc/passwd 上步: 统计 /etc/passwd 有多少行 历史: 无", "COUNT"),
    ("当前目录: / 已知文件: etc 上步: 列出 /etc 目录下的内容 历史: 无", "LIST"),
    ("当前目录: / 已知文件: etc/passwd 上步: 在 /etc/passwd 中搜索 root 历史: 无", "SEARCH"),
    ("当前目录: / 已知文件: python3 上步: 检查 python3 是否安装 历史: 无", "INSPECT"),
    ("当前目录: / 已知文件: ls 上步: 查看 ls 的帮助文档 历史: 无", "HELP"),
    ("当前目录: /etc 已知文件: passwd 上步: 显示 /etc/passwd 的配置内容 历史: 无", "READ_ETC"),
    ("当前目录: / 已知文件:  上步: 查看连接的 USB 设备列表 历史: 无", "USB_DEVICES"),
    ("当前目录: /etc 已知文件:  上步: 检查 /etc 的磁盘使用量 历史: 无", "DISK_USAGE"),
]

with torch.no_grad():
    test_embs = encoder.encode([t[0] for t in test_cases], convert_to_tensor=True).clone()
    test_thought, test_logits = student.forward_emb(test_embs)
    test_preds = test_logits.argmax(dim=-1)

ok = 0
for i, (state, expected) in enumerate(test_cases):
    pred = INTENTS[test_preds[i].item()]
    thought_np = test_thought[i].numpy().round(2)
    correct = pred == expected
    if correct: ok += 1
    print(f"  {'✅' if correct else '❌'} pred={pred:12s} exp={expected:12s}  thought=[", end="")
    print(", ".join(f"{v:.1f}" for v in thought_np[:6]), end="")
    print(" ...]")

print(f"\n  测试准确率: {ok}/{len(test_cases)} = {ok/len(test_cases)*100:.0f}%")
print(f"  想法向量维度标准差: {test_thought.std(dim=0).mean().item():.3f}")

# ── 想法向量 t-SNE 预览 ──────────────────────────────────────
print(f"\n想法向量各维度均值 (前200样本):")
with torch.no_grad():
    sample_thought, _ = student.forward_emb(embs[:200])
mean_per_dim = sample_thought.mean(dim=0)
for d in range(N_DIMS):
    print(f"  dim {d:2d}: {mean_per_dim[d].item():+.3f}")
