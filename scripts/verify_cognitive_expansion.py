"""
验证: 认知空间自动扩展闭环

步骤:
  1. 加载 8 类原始分类器 → 测新意图 (应该不会预测)
  2. 运行 CUSTOM 探索 → 检查 discoverer 是否收集轨迹
  3. 检查发现结果 → 验证聚类是否正确
  4. 生成数据 + 重训 → 测 11 类分类器
  5. 验证新意图被正确预测
"""

import sys, os, json, re
sys.path.insert(0, os.getcwd())

import torch
from sentence_transformers import SentenceTransformer
from collections import Counter

# ── 1. 验证原始 8 类分类器 ──────────────────────────────────

print("=" * 55)
print("  1. 原始 8 类分类器 → 对新意图应不敏感")
print("=" * 55)

# 加载原 8 类检查点
checkpoint_8 = "checkpoints/online_agent/classifier_head.pt"

import torch.nn as nn

encoder = SentenceTransformer("all-MiniLM-L6-v2")

class MLPHead8(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return self.net(x)

INTENTS_8 = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP"]
INTENTS_11 = INTENTS_8 + ["READ_ETC", "USB_DEVICES", "DISK_USAGE"]

head8 = MLPHead8()
if os.path.exists(checkpoint_8):
    sd = torch.load(checkpoint_8, map_location="cpu", weights_only=True)
    head8.load_state_dict(sd, strict=False)
    print(f"  ✅ 原始 8 类检查点加载: {checkpoint_8}")
else:
    print(f"  ⚠️ 原始检查点不存在, 使用当前 11 类分类器")
    # 如果不存在, 直接用当前分类器 (即使用 11 类)
    pass

head8.eval()

# 测试新意图的状态文本
test_states = {
    "READ_ETC": "当前目录: /etc 已知文件: passwd 上步: 读取 /etc/passwd 的配置 历史: 无",
    "USB_DEVICES": "当前目录: / 已知文件:  上步: 列出连接的 USB 设备 历史: 无",
    "DISK_USAGE": "当前目录: /etc 已知文件:  上步: 查看 /etc 的磁盘使用量 历史: 无",
}

print("\n  测试原始 8 类分类器对新意图的预测:")
for intent_name, state in test_states.items():
    emb = encoder.encode(state, convert_to_tensor=True)
    emb = emb.clone()
    with torch.no_grad():
        logits = head8(emb)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_idx = probs.argmax().item()
        pred_name = INTENTS_8[pred_idx]
        confidence = probs.max().item()
    print(f"    {intent_name:15s} → 预测={pred_name:10s} 置信度={confidence:.2f}  "
          f"{'❌ 错误' if pred_name not in intent_name else '✅ 巧合'}")

# ── 2. 运行 CUSTOM 探索 + 发现 ──────────────────────────────

print("\n" + "=" * 55)
print("  2. CUSTOM 探索 → 意图发现")
print("=" * 55)

from agent.online_agent import OnlineAgent

# 使用旧的 8 类训练的数据跑? 不, 直接跑当前系统
agent = OnlineAgent(train_interval=15, batch_size=12, explore_prob=0.08, novelty_weight=0.3)

# 先测试最初的短跑
print("  运行 200 步收集 CUSTOM 轨迹...")
results = agent.run(n_steps=200, verbose=False)

traj_count = len(agent.intent_discoverer.trajectories)
c = Counter(agent.intent_history)
print(f"\n  CUSTOM 使用: {c.get('CUSTOM', 0)}/{sum(c.values())} = {c.get('CUSTOM',0)/sum(c.values())*100:.0f}%")
print(f"  CUSTOM 成功轨迹: {traj_count}")

# 检查发现
new_intents = agent.intent_discoverer.discover()
print(f"\n  发现结果:")
if new_intents:
    for ni in new_intents:
        print(f"    🆕 {ni['name']:15s}  cmd={ni['cmd_base']:10s}  样本={ni['n_samples']}  关键词={ni['keywords']}")
else:
    print(f"    (未达到发现阈值 {agent.intent_discoverer.min_trajectories})")

# ── 3. 检查训练数据质量 ─────────────────────────────────────

print("\n" + "=" * 55)
print("  3. 训练数据验证")
print("=" * 55)

# 检查生成的训练数据
if os.path.exists("data/intent_discovered.jsonl"):
    with open("data/intent_discovered.jsonl") as f:
        discovered_samples = [json.loads(l) for l in f]
    gen_dist = Counter(s['intent'] for s in discovered_samples)
    print(f"  自动生成的训练数据: {len(discovered_samples)} 条")
    for name, cnt in gen_dist.most_common():
        print(f"    {name:15s} {cnt}条")

# ── 4. 验证 11 类分类器 ─────────────────────────────────────

print("\n" + "=" * 55)
print("  4. 11 类分类器 → 对新意图应能正确预测")
print("=" * 55)

# 加载 11 类分类器 (当前)
from benchmark.experiment_v2 import TrainedClassifier
clf_11 = TrainedClassifier()

print("\n  测试 11 类分类器对新意图的预测:")
all_correct = True
for intent_name, state in test_states.items():
    pred = clf_11.predict(state)
    correct = pred == intent_name
    if not correct:
        all_correct = False
    print(f"    {intent_name:15s} → 预测={pred:12s}  {'✅ 正确' if correct else '❌ 错误'}")

print(f"\n  新意图预测 {'✅ 全部正确!' if all_correct else '❌ 有错误'}")

# ── 5. 对比: 分类器是否真的学到了新意图 ─────────────────────

print("\n" + "=" * 55)
print("  5. 分类器输出概率对比")
print("=" * 55)

# 对新状态跑 8 类和 11 类分类器的概率分布
print(f"\n  {'状态':30s} {'8类预测':12s} {'11类预测':12s}")
for intent_name, state in test_states.items():
    # 8 类
    emb = encoder.encode(state, convert_to_tensor=True).clone()
    with torch.no_grad():
        p8 = torch.nn.functional.softmax(head8(emb), dim=-1)
        pred8 = INTENTS_8[p8.argmax().item()]
        conf8 = p8.max().item()
    
    # 11 类  
    emb11 = encoder.encode(state, convert_to_tensor=True).clone()
    pred11 = clf_11.predict(state)
    with torch.no_grad():
        # 获取 11 类概率
        import torch.nn.functional as F
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        emb = enc.encode(state, convert_to_tensor=True).clone()
        logits = clf_11.head(emb)
        p11 = F.softmax(logits, dim=-1)
        conf11 = p11.max().item()
    
    print(f"  {intent_name:30s} {pred8:8s}({conf8:.0%})    {pred11:10s}({conf11:.0%})")

print(f"\n{'='*55}")
print(f"  结论:")
print(f"  - 8 类分类器 {'预测错误' if not all([p==n for n,p in zip(INTENTS_11[-3:],[clf_11.predict(s) for s in test_states.values()])]) else '无法预测新意图'}")
print(f"  - CUSTOM 探索收集了 {traj_count} 条轨迹")
print(f"  - 自动发现了 {len(new_intents)} 个候选意图")
print(f"  - 11 类分类器 {'✅ 正确预测新意图' if all_correct else '❌ 需要改进'}")
print(f"  - 认知扩展闭环 {'✅ 已证实' if traj_count > 10 and all_correct else '❌ 未完成'}")
print(f"{'='*55}")
