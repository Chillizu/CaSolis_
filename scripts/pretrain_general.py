#!/usr/bin/env python3
"""训 BPE tokenizer + 大模型 离线预训练"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# ── 加载 Tokenizer ──
tok_path = "checkpoints/general-v1/tokenizer.json"
if not os.path.exists(tok_path):
    from tokenizers import models, trainers, pre_tokenizers, decoders
    print("训练 BPE tokenizer...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2000, special_tokens=["[BOS]", "[EOS]", "[PAD]", "[ACT]", "[THK]"],
        min_frequency=2, show_progress=True,
    )
    tok.train(["data/offline_merged_corpus.txt"])
    os.makedirs("checkpoints/general-v1", exist_ok=True)
    tok.save(tok_path)
    for s in ["ls -la", "total 128", "root", "Filesystem", "Linux", "MemTotal:", "hello world"]:
        print(f"    {s:30s} → {len(tok.encode(s).ids)} tokens")
else:
    tok = Tokenizer.from_file(tok_path)

VOCAB = tok.get_vocab_size()
print(f"  词表: {VOCAB} tokens")

# ── 2. 模型参数 ──
HIDDEN = 256
EMBED = 96
N_ACTION = 16
ACTION_START = VOCAB
BOS_TOKEN = VOCAB + N_ACTION
EOS_TOKEN = VOCAB + N_ACTION + 1
PAD_TOKEN = VOCAB + N_ACTION + 2
TOTAL_VOCAB = VOCAB + N_ACTION + 3

def encode(text):
    return tok.encode(text).ids

def decode(tokens):
    word_toks = [t for t in tokens if t < VOCAB]
    return tok.decode(word_toks)


class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(TOTAL_VOCAB, EMBED)
        self.rnn = nn.GRUCell(EMBED, HIDDEN)
        self.shared = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.lm_head = nn.Linear(HIDDEN, VOCAB)

    def init_state(self, batch=1):
        return torch.zeros(batch, HIDDEN)

    def step(self, h, token):
        if token.dim() == 0: token = token.unsqueeze(0)
        h_new = self.rnn(self.token_embed(token), h)
        return h_new, self.lm_head(self.shared(h_new))


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, max_len=128):
        with open(jsonl_path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        self.max_len = max_len
        self.seqs = []
        for d in self.data:
            text = f"$ {d['cmd']}\n{d['output']}\n"
            toks = encode(text)[:max_len-2]
            self.seqs.append([BOS_TOKEN] + toks + [EOS_TOKEN])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        x = torch.tensor(s[:-1])
        y = torch.tensor(s[1:])
        mask = y < VOCAB
        return x, y, mask


def collate(batch):
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    bx = torch.stack([F.pad(x, (0, max_len-len(x)), value=PAD_TOKEN) for x in xs])
    by = torch.stack([F.pad(y, (0, max_len-len(y)), value=PAD_TOKEN) for y in ys])
    bm = torch.stack([F.pad(m, (0, max_len-len(m)), value=False) for m in masks])
    return bx, by, bm


# ── 3. 训练 ──
model = BigModel()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5)
ds = SFTDataset("data/offline_merged.jsonl")
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)

print(f"离线预训练 — 通用模型")
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"   词表: {VOCAB} | 隐藏: {HIDDEN}")
print(f"   数据: {len(ds)} 条")
print()

EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batch = 0
    for x, y, mask in loader:
        h = model.init_state(batch=x.size(0))
        loss = 0.0
        n_tok = 0
        for i in range(x.size(1)):
            h, logits = model.step(h, x[:, i])
            v = mask[:, i]
            if v.any():
                loss += F.cross_entropy(logits[v], y[v, i], reduction="mean")
                n_tok += 1
            h = h.detach()
        if n_tok > 0:
            (loss / n_tok).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            sched.step()
            total_loss += (loss / n_tok).item()
            n_batch += 1

    if (epoch+1) % 25 == 0:
        avg = total_loss / max(n_batch, 1)
        print(f"  epoch {epoch+1}/{EPOCHS} | loss: {avg:.4f}")

    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), f"checkpoints/general-v1/model-e{epoch+1}.pt")

torch.save(model.state_dict(), "checkpoints/general-v1/model_final.pt")
print(f"\n✅ 离线预训练完成！")

# ── 测试 ──
model.eval()
h = model.init_state()
t = BOS_TOKEN
out = []
with torch.no_grad():
    for _ in range(100):
        h, logits = model.step(h, torch.tensor([t]))
        h = h.detach()
        lp = F.softmax(logits.squeeze(0) / 0.7, dim=-1)
        t = torch.multinomial(lp, 1).item()
        if t == EOS_TOKEN: break
        if t < VOCAB: out.append(t)
print(f"生成测试: {decode(out)[:200]}")
