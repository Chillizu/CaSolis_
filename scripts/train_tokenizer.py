"""训练 BPE tokenizer"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=2000,
    special_tokens=["[BOS]", "[EOS]", "[PAD]", "[ACT]", "[THK]"],
    min_frequency=2,
    show_progress=True,
)

tokenizer.train(["data/tokenizer_corpus.txt"], trainer)

test_sents = [
    "ls -la", "total 128", "root", "/home/agent_user",
    "cat /etc/hostname", "hello world", "drwxr-xr-x",
    "Filesystem", "Size", "Used", "Available", "Use%",
    "Linux", "x86_64", "GMT", "UTC",
]

print(f"词表大小: {tokenizer.get_vocab_size()}")
print()
for s in test_sents:
    enc = tokenizer.encode(s)
    tokens_str = " ".join(enc.tokens[:8])
    print(f"  {s:30s} → {enc.ids[:6]}  ({len(enc.ids)} tokens)")
    print(f"    {tokens_str}")

tokenizer.save("data/tokenizer-2k.json")
print(f"\n已保存!")
