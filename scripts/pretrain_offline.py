#!/usr/bin/env python3
"""离线预训练 — 词级模型从零训练到能预测命令输出"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# ── 配 tokenizer ─────────────────────────────────────────
TOKENIZER_PATH = "data/offline_tokenizer.json"
tok = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tok.get_vocab_size()  # 1500
N_ACTION = 16
ACTION_START = VOCAB_SIZE           # 1500
ACTION_END = ACTION_START + N_ACTION  # 1516
SPECIAL_START = ACTION_END           # 1516
BOS_TOKEN = SPECIAL_START            # 1516
EOS_TOKEN = SPECIAL_START + 1        # 1517
PAD_TOKEN = SPECIAL_START + 2        # 1518
TOTAL_VOCAB = SPECIAL_START + 3      # 1519

def encode(text):
    return tok.encode(text).ids

def decode(tokens):
    word_toks = [t for t in tokens if t < VOCAB_SIZE]
    return tok.decode(word_toks)


class WordModel(nn.Module):
    """缩小的词级模型 (hidden=256)"""

    def __init__(self, hidden_dim=256, embed_dim=96):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(TOTAL_VOCAB, embed_dim)

        self.rnn = nn.GRUCell(embed_dim, hidden_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.lm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

        # 行动和特殊 token 不需要输出——只预测词 token

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.orthogonal_(p, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(p)

    def init_state(self, batch=1):
        return torch.zeros(batch, self.hidden_dim)

    def step(self, h, token):
        if token.dim() == 0:
            token = token.unsqueeze(0)
        emb = self.token_embed(token)
        h_new = self.rnn(emb, h)
        s = self.shared(h_new)
        lm_logits = self.lm_head(s)
        return h_new, lm_logits


class OfflineDataset(Dataset):
    """从 jsonl 构建训练序列"""

    def __init__(self, jsonl_path, max_len=256):
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.max_len = max_len
        self._build_sequences()

    def _build_sequences(self):
        self.sequences = []
        for d in self.data:
            cmd = d.get("cmd", "")
            output = d.get("output", "")
            text = f"$ {cmd}\n{output}\n"
            tokens = encode(text)
            # 加 BOS/EOS
            seq = [BOS_TOKEN] + tokens[:self.max_len-2] + [EOS_TOKEN]
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 输入 = 0..n-1, 目标 = 1..n
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        # 只对词 token 计算损失
        mask = y < VOCAB_SIZE
        return x, y, mask


def train_offline(
    model_path="checkpoints/word-offline-v1",
    epochs=200,
    lr=3e-4,
    batch_size=8,
    hidden_dim=256,
):
    os.makedirs(model_path, exist_ok=True)

    model = WordModel(hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * 50, eta_min=1e-5)

    ds = OfflineDataset("data/offline_raw.jsonl")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    print(f"🧠 离线预训练 — {model_path}")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {VOCAB_SIZE} 词 + {N_ACTION} 行动")
    print(f"   数据: {len(ds)} 条序列")
    print(f"   轮次: {epochs} | lr: {lr} | batch: {batch_size}")
    print()

    best_loss = float("inf")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, mask in loader:
            h = model.init_state(batch=x.size(0))

            loss = 0.0
            n_tokens = 0

            # 逐 token 前馈
            for i in range(x.size(1)):
                h, lm_logits = model.step(h, x[:, i])

                # 只对有 mask 的位置算损失
                valid = mask[:, i]
                if valid.any():
                    l = F.cross_entropy(
                        lm_logits[valid],
                        y[valid, i],
                        reduction="mean",
                    )
                    loss += l
                    n_tokens += 1

                h = h.detach()

            if n_tokens > 0:
                loss = loss / n_tokens
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                sched.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  epoch {epoch+1:>4d}/{epochs} | loss: {avg_loss:.4f} | lr: {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"{model_path}/model-e{epoch+1}.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{model_path}/model_best.pt")

    torch.save(model.state_dict(), f"{model_path}/model_final.pt")
    print(f"\n✅ 完成! 最终损失: {total_loss/max(n_batches,1):.4f}")
    print(f"   最佳损失: {best_loss:.4f}")

    # ── 测试 ──
    print("\n🧪 生成测试:")
    model.eval()
    h = model.init_state()
    token = BOS_TOKEN
    out_tokens = []

    with torch.no_grad():
        for _ in range(100):
            tok_t = torch.tensor([token], dtype=torch.long)
            h, lm_logits = model.step(h, tok_t)
            h = h.detach()

            lp = F.softmax(lm_logits.squeeze(0) / 0.8, dim=-1)
            token = torch.multinomial(lp, 1).item()

            if token == EOS_TOKEN:
                break
            if token < VOCAB_SIZE:
                out_tokens.append(token)

    print(f"  生成: {decode(out_tokens)[:200]}")

    return model


def _collate(batch):
    """动态 padding"""
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    batch_x = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=PAD_TOKEN) for x in xs
    ])
    batch_y = torch.stack([
        F.pad(y, (0, max_len - y.size(0)), value=PAD_TOKEN) for y in ys
    ])
    batch_mask = torch.stack([
        F.pad(m, (0, max_len - m.size(0)), value=False) for m in masks
    ])
    return batch_x, batch_y, batch_mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=str, default="checkpoints/word-offline-v1")
    args = parser.parse_args()

    train_offline(
        model_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        hidden_dim=args.hidden,
    )
