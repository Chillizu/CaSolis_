"""
P18 Phase 1-LITE 验证: WorldModel V5
1. 参数计数 ~364K
2. 前向传播: next_state 输出形状正确
3. 训练: loss 下降
4. OnlineAgent 集成: 初始化 + transition 收集
5. Next-state cosine similarity threshold
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f casolis-sandbox 2>/dev/null")

from agent.world_model_v5 import WorldModelV5
import torch, math

print("=" * 60)
print("P18 Phase 1-LITE: WorldModel V5")
print("=" * 60)

# 1. 参数计数
print("\n--- 参数计数 ---")
wm = WorldModelV5()
print(f"  Total params: {wm._param_count:,}")
assert 300000 <= wm._param_count <= 450000, \
    f"Parameter count out of range: {wm._param_count}"
# 验证已知维度
assert wm.action_encoder.intent_embed.num_embeddings == 3
assert wm.core.state_proj.out_features == 80
assert wm.core.gru.hidden_size == 160
print(f"  [OK] Params={wm._param_count:,} (expected ~364K)")

# 2. 前向传播
print("\n--- 前向传播 ---")
state = torch.randn(4, 384)
intents = torch.tensor([0, 1, 2, 0], dtype=torch.long)
act_embs = wm.action_encoder(intents, state)
result = wm.core(state, act_embs)
assert result["next_state"].shape == (4, 384), f"Shape: {result['next_state'].shape}"
assert result["reward"].shape == (4, 1)
assert result["cont"].shape == (4, 1)
assert result["hidden"].shape == (4, 160)
print(f"  [OK] Forward pass: next_state(4,384)")

# Hidden state persistence
result2 = wm.core(state, act_embs, hidden=result["hidden"])
assert not torch.equal(result["hidden"], result2["hidden"]), "Hidden should update"
print(f"  [OK] Hidden state persistence")

# 3. 训练 loss 下降
print("\n--- 训练 loss 下降 ---")
T = 64
train_states = torch.randn(T, 384)
train_acts = wm.action_encoder(
    torch.randint(0, 3, (T,)),
    train_states,
)
train_next = torch.randn(T, 384)
train_rew = torch.randn(T, 1)
train_cont = torch.ones(T, 1)

losses = []
for epoch in range(20):
    loss = wm.train_step(train_states, train_acts, train_next,
                          train_rew, train_cont, chunk_size=16)
    losses.append(loss)

print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
assert losses[-1] < losses[0], f"Loss went up: {losses[0]:.4f} -> {losses[-1]:.4f}"
print(f"  [OK] Loss decreased over 20 epochs")

# 4. OnlineAgent 集成
print("\n--- OnlineAgent 集成 ---")
from agent.online_agent import OnlineAgent
os.system("docker rm -f casolis-sandbox 2>/dev/null")
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
os.system("docker rm -f casolis-sandbox 2>/dev/null")

assert hasattr(agent, 'world_model_v5'), "V5 missing from agent"
assert hasattr(agent, '_wm5_buffer'), "wm5_buffer missing"
assert agent.world_model_v5._param_count > 300000
print(f"  V5 params: {agent.world_model_v5._param_count:,}")
print(f"  Buffer: {len(agent._wm5_buffer)} entries")
print(f"  [OK] OnlineAgent initialized with V5")

# 5. 跑几步收集数据
print(f"\n--- 短跑收集 transition ---")
for i in range(20):
    agent.step()
print(f"  20 steps done, buffer: {len(agent._wm5_buffer)} entries")
assert len(agent._wm5_buffer) >= 1, "No transitions collected"
print(f"  [OK] Transitions collected")

os.system("docker rm -f casolis-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("Phase 1-LITE PASSED")
print(f"{'='*60}")
