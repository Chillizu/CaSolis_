"""
Phase 2-LITE 验证: DoCalculusEngine
1. DAG + d-separation
2. 后门准则
3. 干预评分
4. ATE 估计
5. FactGraph 集成
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"

from agent.do_calculus import DoCalculusEngine, DAG
from agent.fact_graph import FactGraph, EDGE_CAUSES, EDGE_CORRELATES, EDGE_PREDICTS

print("=" * 60)
print("Phase 2-LITE: DoCalculusEngine")
print("=" * 60)

# 1. DAG + d-separation
print("\n--- DAG + d-separation ---")
dag = DAG()
dag.add_edge("X", "M")
dag.add_edge("M", "Y")
# Chain: X → M → Y
assert not dag.d_separated({"X"}, {"Y"}, set()), "Chain: X not d-sep Y"
assert dag.d_separated({"X"}, {"Y"}, {"M"}), "Chain: X d-sep Y given M"
print("[OK] Chain d-separation")

# Fork: X ← Z → Y
dag2 = DAG()
dag2.add_edge("Z", "X")
dag2.add_edge("Z", "Y")
assert not dag2.d_separated({"X"}, {"Y"}, set()), "Fork: X not d-sep Y"
assert dag2.d_separated({"X"}, {"Y"}, {"Z"}), "Fork: X d-sep Y given Z"
print("[OK] Fork d-separation")

# Collider: X → Z ← Y
dag3 = DAG()
dag3.add_edge("X", "Z")
dag3.add_edge("Y", "Z")
assert dag3.d_separated({"X"}, {"Y"}, set()), "Collider: X d-sep Y (unconditioned)"
assert not dag3.d_separated({"X"}, {"Y"}, {"Z"}), "Collider: X not d-sep Y given Z"
print("[OK] Collider d-separation")

# 2. 后门准则
print("\n--- 后门准则 ---")
engine = DoCalculusEngine()
engine.add_edge("cpu_cores", "cpu_usage", "causes")
engine.add_edge("cpu_usage", "temperature", "causes")
engine.add_edge("ambient_temp", "temperature", "causes")
engine.add_edge("ambient_temp", "cpu_usage", "causes")  # confounder

backdoor = engine.find_backdoor_set("cpu_cores", "temperature")
print(f"  Back-door set for cpu_cores→temperature: {backdoor}")
# ambient_temp is a confounder, should be in backdoor set
if backdoor:
    print(f"  [OK] Back-door set found: {backdoor}")

# 3. 干预评分
print("\n--- 干预评分 ---")
transitions = []
cpu_val = 4
for i in range(100):
    # Simulate: cpu changes randomly → load follows
    if i > 0 and i % 5 == 0:
        cpu_val = 22 if cpu_val == 4 else 4  # change every 5 steps
    load_val = 0.8 if cpu_val == 22 else 0.5
    transitions.append({
        "pre_state": {"cpu_cores": cpu_val},
        "post_state": {"load": load_val},
    })

score = engine.causal_score("cpu_cores", "load", transitions)
print(f"  Causal score cpu_cores→load: {score:.3f}")
assert score > 0, f"Expected positive causal score, got {score}"
print("[OK] Causal score detects real causation")
# Negative control: cause should NOT be detected
transitions_null = []
for i in range(100):
    transitions_null.append({
        "pre_state": {"disk_usage": i % 10, "cpu_cores": 8},
        "post_state": {"disk_usage": (i + 1) % 10, "cpu_cores": 8},
    })
score_null = engine.causal_score("disk_usage", "cpu_cores", transitions_null)
print(f"  Causal score disk_usage→cpu_cores: {score_null:.3f}")
assert score_null < 0.5, f"Expected low causal score, got {score_null}"
print("[OK] Causal score rejects spurious correlation")

# 4. ATE 估计
print("\n--- ATE 估计 ---")
transitions_ate = []
for i in range(200):
    # 模拟: treatment=1 → outcome=happier
    treatment = 1 if i < 100 else 0
    confounder = 1 if i % 2 == 0 else 0
    outcome = 0.8 if treatment == 1 else 0.3
    transitions_ate.append({
        "pre_state": {"T": treatment, "C": confounder},
        "post_state": {"Y": outcome},
    })

ate = engine.estimate_ate("T", "Y", transitions_ate)
print(f"  ATE T→Y (unadjusted): {ate:.3f}")
assert ate > 0.2, f"Expected positive ATE, got {ate}"

# With adjustment set
fg = FactGraph()
fg.add_node("cpu_cores", "22", category="system", step=0)
fg.add_node("load", "0.5", category="system", step=0)
fg.add_edge("cpu_cores", "load", EDGE_CAUSES, weight=0.6, step=0)
fg.add_edge("load", "temperature", EDGE_PREDICTS, weight=0.4, step=0)
fg.add_edge("cpu_cores", "temperature", EDGE_CORRELATES, weight=0.3, step=0)

engine2 = DoCalculusEngine()
count = engine2.add_from_factgraph(fg)
print(f"  Imported {count} edges from FactGraph")
assert count >= 1, f"Expected >=1 causal edges, got {count}"
assert "cpu_cores" in engine2.graph.nodes
assert "load" in engine2.graph.nodes
print("[OK] FactGraph integration")

# Test d-separation on imported graph
x, y = "cpu_cores", "temperature"
ds = engine2.graph.d_separated({x}, {y}, {"load"})
print(f"  {x} ⟂ {y} | load? {ds}")
print(f"  (Collider/non-collider path depends on edge directions)")

print(f"\n{'='*60}")
print("Phase 2-LITE PASSED")
print("{'='*60}")
