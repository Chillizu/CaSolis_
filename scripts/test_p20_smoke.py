"""
P20 联合验证: SalienceSignal + HabitSystem + StateEncoder 注意力门控
- 50 步烟测
- 验证 P20 新模块集成到主循环
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f casolis-sandbox 2>/dev/null")

from agent.online_agent import OnlineAgent
import torch
print("=" * 60)
print("P20 联合验证: Salience + Habit + StateEncoder")
print("=" * 60)

os.system("docker rm -f casolis-sandbox 2>/dev/null")
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")

assert hasattr(agent, 'salience'), "Missing SalienceSignal"
assert hasattr(agent, 'habit_system'), "Missing HabitSystem"
print(f"  SalienceSignal: ready")
print(f"  HabitSystem: ready")
print()

# Run 50 steps
for i in range(50):
    agent.step()

print(f"50 steps complete.")
print(f"  Success rate: {agent.success_count}/{agent.step_count} "
      f"({agent.success_count/max(agent.step_count,1)*100:.0f}%)")

# Check salience was updated
sal_stats = agent.salience.get_stats()
assert sal_stats['window_len'] > 0, "SalienceSignal window empty"
print(f"  Salience last={sal_stats['last']:.3f} mean={sal_stats['mean']:.3f} max={sal_stats['max']:.3f}")
print(f"[OK] SalienceSignal updated during run")

# Check habits were registered
habit_stats = agent.habit_system.get_stats()
assert habit_stats['n_habits'] > 0, "No habits registered"
print(f"  Habits registered: {habit_stats['n_habits']}")
print(f"[OK] HabitSystem registered habits")

# Check StateEncoder with salience hints still works
state_text = agent.state_encoder.get_state_text(salience_hints={"cpu": 0.9})
assert isinstance(state_text, str) and len(state_text) > 0
print(f"[OK] StateEncoder salience_hints path works")

os.system("docker rm -f casolis-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("P20 联合验证 PASSED")
print(f"{'='*60}")
