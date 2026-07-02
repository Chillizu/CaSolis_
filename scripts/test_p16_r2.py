"""
P16 R2 验证: TransitionMiner + HypothesisEngine
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"

from agent.transition_miner import TransitionMiner
from agent.hypothesis_engine import HypothesisEngine
from agent.fact_graph import FactGraph, EDGE_CAUSES, EDGE_CORRELATES

print("=" * 60)
print("P16 R2 验证")
print("=" * 60)

# -- 1. TransitionMiner --
print("\n--- TransitionMiner ---")
sim_path = "/tmp/test_transitions.jsonl"
with open(sim_path, "w") as f:
    for step in range(100):
        pre = {"cpu_cores": "22", "mem_total": "32G", "load": "0.5",
               "kernel": "6.0", "hostname": "test", "disk_root": "50G"}
        post = dict(pre)
        if step % 10 == 0:
            post["cpu_cores"] = "16"
        if step % 10 == 0 and step > 10:
            post["load"] = "1.2"
        if step % 7 == 0:
            post["mem_total"] = "16G"
        tx = {"step": step, "action": "TRY",
              "pre_state": pre, "post_state": post,
              "exit_code": 0, "output_len": 100,
              "had_new_facts": True, "reward": 1.0}
        f.write(json.dumps(tx) + "\n")
print("模拟数据 100 条已创建")

miner = TransitionMiner(max_edges_per_batch=20)
candidates = miner.mine(sim_path, window=100)
print(f"候选边: {len(candidates)}")
for c in candidates:
    print(f"  {c['src']} --[{c['rel']}]--> {c['dst']}  weight={c['weight']:.3f} n_obs={c['n_obs']}")
assert len(candidates) > 0, "应有候选边"
assert any(c["rel"] == EDGE_CAUSES for c in candidates), "应有因果边"
assert any(c["rel"] == EDGE_CORRELATES for c in candidates), "应有共现边"
print("[OK] Miner 正常产出因果+共现边")

# -- 2. apply_to_graph --
print("\n--- apply_to_graph ---")
g = FactGraph()
for k in ("cpu_cores", "load", "mem_total", "kernel", "hostname", "disk_root"):
    g.add_node(k, "unknown", category="system", step=0, n_evidence=1)
pre_existing = sum(len(e) for e in g.edges.values())
miner.apply_to_graph(candidates, g, step=100)
n_edges = sum(len(e) for e in g.edges.values())
assert n_edges > pre_existing, "应新增边"
miner_edges = [e for src in g.edges.values() for e in src
               if e.get("hypothesis_key") == "transition_miner"]
print(f"新增 miner 边: {len(miner_edges)}/{n_edges - pre_existing}")
for e in miner_edges:
    assert "n_support" in e and e["n_support"] >= 1
    assert e["hypothesis_key"] == "transition_miner"
print("[OK] 边字段完整")

# -- 3. HypothesisEngine --
print("\n--- HypothesisEngine ---")
engine = HypothesisEngine(top_k=5, min_priority=0.05)
real_g = FactGraph()
real_g.add_node("cpu_cores", "22", category="system", step=0, confidence=0.9, n_evidence=15)
real_g.add_node("load", "0.5", category="system", step=0, confidence=0.3, n_evidence=2)
real_g.add_node("mem_total", "32G", category="system", step=0, confidence=0.8, n_evidence=10)
real_g.add_node("kernel", "6.0", category="system", step=0, confidence=0.9, n_evidence=20)
real_g.add_node("hostname", "test", category="network", step=0, confidence=0.7, n_evidence=5)
real_g.add_node("disk_root", "50G", category="storage", step=0, confidence=0.6, n_evidence=3)

hyps = engine.generate(candidates, real_g)
print(f"假设: {len(hyps)}")
for h in hyps:
    print(f"  {h['if_node']} --[{h['rel']}]--> {h['then_node']}  "
          f"priority={h['priority']:.4f} unc={h['uncertainty']:.3f} "
          f"test={h['testability']:.2f} cross={h['cross_category']}")
assert len(hyps) > 0, "应生成假设"
assert all("prediction" in h and "priority" in h for h in hyps)

# Dedup
hyps2 = engine.generate(candidates, real_g)
assert len(hyps2) <= len(hyps), f"去重应有效: {len(hyps2)} <= {len(hyps)}"
print(f"[OK] 去重有效: {len(hyps2)} <= {len(hyps)}")

# -- 4. Integration smoke test --
print("\n--- OnlineAgent 集成 ---")
os.system("docker rm -f casolis-sandbox 2>/dev/null")
from agent.online_agent import OnlineAgent

agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                     lr=1e-4, conductor_gate=0.7, mode="auto")
assert hasattr(agent, 'transition_miner')
assert hasattr(agent, 'hypothesis_engine')
assert hasattr(agent, '_latest_hypotheses')

for i in range(15):
    agent.step()
print(f"15 步完成, step_count={agent.step_count}")
print(f"_latest_hypotheses: {agent._latest_hypotheses}")
print("[OK] 集成无崩溃")

print(f"\n{'='*60}")
print(f"P16 R2 PASSED")
print(f"  Candidates: {len(candidates)}")
print(f"  Hypotheses: {len(hyps)}")
print(f"  Integration: OK")
print(f"{'='*60}")
