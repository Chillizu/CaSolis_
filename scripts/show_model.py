#!/usr/bin/env python3
"""展示离线预训练模型的能力"""

import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_offline import WordModel, encode, decode, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE

def load(path="checkpoints/word-offline-v1/model_best.pt"):
    model = WordModel(hidden_dim=256)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def generate(model, prompt="", max_len=150, temp=0.7):
    h = model.init_state()
    tokens = encode(prompt) if prompt else [BOS_TOKEN]

    with torch.no_grad():
        for tok in tokens:
            h, _ = model.step(h, torch.tensor([tok]))
            h = h.detach()

        for _ in range(max_len):
            h, logits = model.step(h, torch.tensor([tokens[-1]]))
            h = h.detach()

            lp = F.softmax(logits.squeeze(0) / temp, dim=-1)
            next_tok = torch.multinomial(lp, 1).item()

            if next_tok == EOS_TOKEN:
                break
            if next_tok < VOCAB_SIZE:
                tokens.append(next_tok)

    return decode(tokens[len(encode(prompt)):] if prompt else tokens)

print("🧪 离线预训练模型能力展示\n")

model = load()

# 1. 自动续写命令输出
print("=" * 60)
print("1. 给定命令前缀，续写命令输出")
print("=" * 60)
for prefix in ["$ ls -la\n", "$ df -h\n", "$ cat /etc/hostname\n", "$ uname -a\n"]:
    out = generate(model, prompt=prefix, max_len=50)
    print(f"  输入: {prefix.strip()}")
    print(f"  输出: {out[:200]}")
    print()

# 2. 自由生成
print("=" * 60)
print("2. 自由生成（只用 BOS）")
print("=" * 60)
for _ in range(3):
    out = generate(model, max_len=80, temp=0.9)
    print(f"  {out[:200]}")
    print()

# 3. 连贯性测试
print("=" * 60)
print("3. 多步生成（看能不能保持一致）")
print("=" * 60)
out = generate(model, max_len=200)
print(f"  {out[:400]}")
