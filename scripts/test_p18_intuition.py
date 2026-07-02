"""
P18 Phase 3-LITE 验证: IntuitionBuffer
1. 存储 + 查询正确性
2. 熟悉度单调增长
3. OnlineAgent 集成
4. 方向建议有区分度
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f casolis-sandbox 2>/dev/null")

from agent.intuition_buffer import IntuitionBuffer
import torch

print("=" * 60)
print("P18 Phase 3-LITE: IntuitionBuffer")
print("=" * 60)

# 1. 基础功能
print("\n--- 存储 + 查询 ---")
buf = IntuitionBuffer(capacity=1024)
assert buf.size == 0

# 存储随机经验
for i in range(20):
    t = torch.randn(16)
    buf.store(t, intent=i % 3, reward=float(i) / 20.0)

assert buf.size == 20
print(f"  [OK] Stored 20 entries")

# 查询
query = torch.randn(16)
result = buf.query(query, top_k=5)
assert "familiarity" in result
assert "analogy" in result
assert "direction" in result
familiarity = result["familiarity"]
print(f"  Familiarity: {familiarity:.4f} (range 0-1)")
assert 0.0 <= familiarity <= 1.0, f"Familiarity out of range: {familiarity}"
print(f"  Analogy intent: {result['analogy']['intent']}")
print(f"  Direction: {result['direction']}")
assert len(result["direction"]) == 3
print(f"  [OK] Query returns all 3 outputs")

# 2. 熟悉度增长
print("\n--- 熟悉度增长 ---")
buf2 = IntuitionBuffer(capacity=32)
probe = torch.randn(16)
fam_scores = []
for i in range(10):
    buf2.store(probe.clone() + torch.randn(16) * 0.05,  # 相似向量
               intent=0, reward=0.5)
    r = buf2.query(probe, top_k=5)
    fam_scores.append(r["familiarity"])
print(f"  Familiarity progression: {[f'{s:.3f}' for s in fam_scores]}")
assert fam_scores[-1] > fam_scores[0], \
    f"Familiarity should increase: {fam_scores[0]:.3f} -> {fam_scores[-1]:.3f}"
print(f"  [OK] Familiarity increases with repetition")

# 3. 方向建议
print("\n--- 方向建议 ---")
buf3 = IntuitionBuffer(capacity=64)
# 存 20 条 intent=0 (OBSERVE) 高奖励 + 2 条 intent=1 (CREATE) 低奖励
intent0_vec = torch.randn(16)
for i in range(20):
    buf3.store(intent0_vec + torch.randn(16) * 0.1, intent=0, reward=1.0)
for i in range(2):
    buf3.store(torch.randn(16), intent=1, reward=0.1)

r = buf3.query(intent0_vec, top_k=5)
print(f"  Direction: {r['direction']}")
assert r["direction"][0] > r["direction"][1], \
    f"OBSERVE should dominate: {r['direction']}"
print(f"  [OK] Direction favors dominant intent")

# 4. 容量测试
print("\n--- 容量 ---")
buf4 = IntuitionBuffer(capacity=16)
for i in range(32):
    buf4.store(torch.randn(16), intent=i % 3, reward=0.5)
assert buf4.size == 16  # 不应超过 capacity
print(f"  [OK] Capacity cap: {buf4.size}/16")

# 5. OnlineAgent 集成
print("\n--- OnlineAgent 集成 ---")
from agent.online_agent import OnlineAgent
os.system("docker rm -f casolis-sandbox 2>/dev/null")
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
os.system("docker rm -f casolis-sandbox 2>/dev/null")

assert hasattr(agent, 'intuition_buffer'), "intuition_buffer missing"
assert agent.intuition_buffer.capacity == 1024
print(f"  IntuitionBuffer capacity={agent.intuition_buffer.capacity}")

for i in range(15):
    agent.step()
print(f"  15 steps, buffer size: {agent.intuition_buffer.size}")
assert agent.intuition_buffer.size >= 1, "No entries recorded"
print(f"  [OK] OnlineAgent integration")

os.system("docker rm -f casolis-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("Phase 3-LITE PASSED")
print(f"{'='*60}")
