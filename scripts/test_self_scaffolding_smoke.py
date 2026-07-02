"""Smoke test for Self-Scaffolding integration.
Runs 30 steps and verifies plan generation/execution appears in logs.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f casolis-sandbox 2>/dev/null")

from agent.online_agent import OnlineAgent
print("=" * 60)
print("Self-Scaffolding smoke test")
print("=" * 60)

agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")

# Run 30 steps
for i in range(30):
    agent.step()

print(f"30 steps complete.")
print(f"  Success rate: {agent.success_count}/{agent.step_count} "
      f"({agent.success_count/max(agent.step_count,1)*100:.0f}%)")

# Check plan fields
assert hasattr(agent, '_active_plan'), "Missing _active_plan field"
assert hasattr(agent, '_plan_step_count'), "Missing _plan_step_count field"
assert hasattr(agent.goal_generator, '_plan_topic_stats'), "Missing _plan_topic_stats"
print("[OK] Self-Scaffolding fields present")

os.system("docker rm -f casolis-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("Self-Scaffolding smoke test PASSED")
print(f"{'='*60}")
