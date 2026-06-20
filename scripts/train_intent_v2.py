"""
训练意图分类器

架构: SentenceTransformer(all-MiniLM-L6-v2) + Linear(384, 8)
数据: data/intent_train.jsonl (3732 条)
输出: checkpoints/intent_classifier/
"""

import json
import os
import sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ── 配置 ─────────────────────────────────────────────────

DATA_PATH = "data/intent_train_v3.jsonl"
OUTPUT_DIR = "checkpoints/intent_classifier"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 5e-4
EVAL_SPLIT = 0.15
N_CLASSES = 11
INTENT_NAMES = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE"]


# ── 数据加载 ─────────────────────────────────────────────

def load_data(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["state_text"])
            labels.append(data["intent_id"])
    return texts, labels


# ── 分类头 ───────────────────────────────────────────────

class IntentHead(nn.Module):
    """MLP(384 → 128 → 8) + LayerNorm + Dropout(0.2)"""
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── 训练 ─────────────────────────────────────────────────

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {DEVICE}")

    # 加载数据
    texts, labels = load_data(DATA_PATH)
    print(f"数据: {len(texts)} 条")
    print(f"标签分布: {dict(Counter(labels).most_common())}")

    # 加载 Sentence Transformer (冻结)
    print("加载 all-MiniLM-L6-v2...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encoder.to(DEVICE)
    encoder.eval()

    # 预计算所有嵌入
    print("编码状态文本...")
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"嵌入形状: {embeddings.shape}")

    # 训练/测试分割
    n = len(embeddings)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * (1 - EVAL_SPLIT))
    train_idx, eval_idx = indices[:split], indices[split:]

    train_emb = torch.FloatTensor(embeddings[train_idx])
    train_labels = torch.LongTensor(np.array(labels)[train_idx])
    eval_emb = torch.FloatTensor(embeddings[eval_idx])
    eval_labels = torch.LongTensor(np.array(labels)[eval_idx])

    print(f"训练: {len(train_idx)}, 评估: {len(eval_idx)}")

    # 分类头
    head = IntentHead().to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 训练循环
    best_acc = 0.0
    train_emb, eval_emb = train_emb.to(DEVICE), eval_emb.to(DEVICE)
    train_labels, eval_labels = train_labels.to(DEVICE), eval_labels.to(DEVICE)

    for epoch in range(EPOCHS):
        head.train()

        # Mini-batch SGD
        perm = torch.randperm(len(train_emb), device=DEVICE)
        total_loss = 0
        n_batches = 0

        for start in range(0, len(train_emb), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch_emb = train_emb[idx]
            batch_labels = train_labels[idx]

            logits = head(batch_emb)
            loss = F.cross_entropy(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # 评估
        head.eval()
        with torch.no_grad():
            logits = head(eval_emb)
            preds = logits.argmax(dim=1)
            acc = (preds == eval_labels).float().mean().item()

        # 保存 best
        if acc > best_acc:
            best_acc = acc
            torch.save(head.state_dict(), os.path.join(OUTPUT_DIR, "best_head.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={acc:.3f}")

    # 加载 best, 最终评估
    head.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_head.pt")))
    head.eval()
    with torch.no_grad():
        logits = head(eval_emb)
        preds = logits.argmax(dim=1)
        acc = (preds == eval_labels).float().mean().item()

        # 混淆矩阵
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(eval_labels.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1

    print(f"\n{'=' * 50}")
    print(f"  最终验证准确率: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  最佳模型保存: {OUTPUT_DIR}/best_head.pt")
    print(f"{'=' * 50}")

    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    print(f"  {'':8s}", end="")
    for name in INTENT_NAMES:
        print(f"{name:8s}", end="")
    print()
    for i, name in enumerate(INTENT_NAMES):
        print(f"  {name:8s}", end="")
        for j in range(N_CLASSES):
            print(f"{cm[i,j]:4d}   ", end="")
        # 召回率
        total = cm[i].sum()
        recall = cm[i,i] / total if total > 0 else 0
        print(f"  recall={recall:.2f}")

    # 每类准确率
    print(f"\n  每类准确率:")
    for i, name in enumerate(INTENT_NAMES):
        total = cm[i].sum()
        correct = cm[i,i]
        if total > 0:
            print(f"    {name:10s}  {correct}/{total} = {correct/total*100:.1f}%")

    return head


if __name__ == "__main__":
    train()
