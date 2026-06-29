"""
P18 联合验证: WorldModel V5 + IntuitionBuffer
- 50 步烟测
- 两个模块同时工作
- 验证核心指标
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f folunar-sandbox 2>/dev/null")

from agent.online_agent import OnlineAgent
import torch
print("=" * 60)
print("P18 联合验证: V5 + IntuitionBuffer")
print("=" * 60)

os.system("docker rm -f folunar-sandbox 2>/dev/null")
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")

assert hasattr(agent, 'world_model_v5')
assert hasattr(agent, 'intuition_buffer')
assert agent.world_model_v5._param_count > 300000
assert agent.intuition_buffer.capacity == 1024

print(f"  V5 params: {agent.world_model_v5._param_count:,}")
print(f"  IntuitionBuffer: cap={agent.intuition_buffer.capacity}")
print()

# Run 50 steps
v5_losses = []
ib_sizes = []
for i in range(50):
    agent.step()
    
    # Track metrics
    if agent.world_model_v5.train_losses:
        v5_losses.append(agent.world_model_v5.train_losses[-1])
    ib_sizes.append(agent.intuition_buffer.size)

print(f"50 steps complete.")
print(f"  Success rate: {agent.success_count}/{agent.step_count} "
      f"({agent.success_count/max(agent.step_count,1)*100:.0f}%)")
print(f"  IntuitionBuffer entries: {agent.intuition_buffer.size}")
print(f"  V5 training steps: {len(v5_losses)}")

# Check metrics
assert agent.intuition_buffer.size >= 10, \
    f"IntuitionBuffer too small: {agent.intuition_buffer.size}"
print(f"\n[OK] IntuitionBuffer >= 10 entries")

# V5: at least attempted training
if v5_losses:
    print(f"  V5 losses: {len(v5_losses)} training steps")
    print(f"  Last loss: {v5_losses[-1]:.4f}")
    print(f"[OK] V5 training ran")
else:
    print("  [SKIP] V5 training not yet triggered (needs buffer >= 20)")

# Query IntuitionBuffer
if agent.intuition_buffer.size >= 3:
    thought = agent.persistent_thought
    result = agent.intuition_buffer.query(thought)
    print(f"\nIntuition query:")
    print(f"  Familiarity: {result['familiarity']:.3f}")
    print(f"  Direction: {result['direction']}")
    assert 0 <= result['familiarity'] <= 1.0001, \
        f"Familiarity out of range: {result['familiarity']}"
    print(f"[OK] Intuition query returns valid results")

# Check V5 forward pass
state_emb = agent.classifier.get_embedding(
    agent.state_encoder.get_state_text())
intent_idx = 0
act_emb = agent.world_model_v5.encode_action(intent_idx, state_emb.unsqueeze(0))
v5_result = agent.world_model_v5.core(state_emb.unsqueeze(0), act_emb)
pred_next = v5_result["next_state"]
print(f"\nV5 forward pass:")
print(f"  Input state: {state_emb.shape}")
print(f"  Predicted next: {pred_next.shape}")
assert pred_next.shape == (1, 384)
print(f"[OK] V5 forward pass works after 50 steps")

os.system("docker rm -f folunar-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("P18 联合验证 PASSED")
print(f"{'='*60}")
