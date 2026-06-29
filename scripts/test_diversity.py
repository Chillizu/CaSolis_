#!/usr/bin/env python3
"""Test probe/command diversity after homogeneity fixes"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'
from agent.online_agent import OnlineAgent
from collections import Counter

agent = OnlineAgent(conductor_gate=0.6)
agent.run(n_steps=30)

# Goal sources
srcs = Counter(h.get("source","") for h in agent.goal_generator.goal_history)
cmd_goals = [h for h in agent.goal_generator.goal_history if "try_cmd" in str(h.get("source",""))]
print("命令尝试:", len(cmd_goals))
for h in cmd_goals[:5]:
    d = h.get("description","")[:60]
    print(f"  step {h.get('step','')}: {d}")

# Probe diversity
probes = [h for h in agent.goal_generator.goal_history
          if "probe:" in str(h.get("source","")) or "experiment:" in str(h.get("source",""))]
dst_cmds = Counter(h.get("description","") for h in probes)
print(f"探针/实验 ({len(probes)}):")
for c, n in dst_cmds.most_common(8):
    print(f"  {c[:60]} x{n}")

print("意图:", dict(Counter(agent.intent_history).most_common()))
print("成功:", agent.success_count, "/", agent.step_count)
agent.pstore.close()
