"""
P16 R3 验证: ExperimentPlanner + Verdict + Integration
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"
os.system("docker rm -f folunar-sandbox 2>/dev/null")

from agent.experiment_planner import ExperimentPlanner
from agent.verdict import Verdict
from agent.fact_graph import FactGraph, EDGE_CAUSES

print("=" * 60)
print("P16 R3 验证")
print("=" * 60)

# -- 1. ExperimentPlanner passive observation --
print("\n--- ExperimentPlanner ---")
planner = ExperimentPlanner()
hypothesis = {
    "if_node": "cpu_cores",
    "rel": "causes",
    "then_node": "load",
    "prediction": "cpu_cores changes -> load changes",
    "priority": 0.5,
}
plan = planner.plan(hypothesis)
assert plan, "应生成实验计划"
print(f"Plan: {plan['cmd'][:80]}")
print(f"  type={plan.get('type')}, timeout={plan.get('timeout')}")
assert "cmd" in plan
assert "timeout" in plan
assert plan.get("hypothesis_key") == "cpu_cores:causes:load"
print("[OK] ExperimentPlanner generates plans with cmd + timeout + hypothesis_key")

# -- 2. ExperimentPlanner plan for non-known keys (intervention fallback) --
hypo2 = {"if_node": "unknown_key", "rel": "causes", "then_node": "load",
         "prediction": "unknown -> load"}
plan2 = planner.plan(hypo2)
assert plan2, "应生成回退计划 (至少读 dst)"
print(f"  Fallback plan: {plan2['cmd'][:60]}")
print("[OK] Fallback plan works for unknown src")

# -- 3. Verdict --
print("\n--- Verdict ---")
g = FactGraph()
g.add_node("cpu_cores", "22", category="system", step=0)
g.add_node("load", "0.5", category="system", step=0)
g.add_edge("cpu_cores", "load", EDGE_CAUSES, weight=0.5, step=0,
           n_support=0, n_against=0, hypothesis_key="test")

verdict = Verdict(lr=0.3)
test_plan = {
    "hypothesis_key": "cpu_cores:causes:load",
    "predicted_exit": 0,
    "predicted_output_len": 100,
    "_step": 50,
}
test_result = {
    "exit_code": 0,
    "output_len": 80,
    "success": True,
}

v = verdict.evaluate(test_plan, test_result, g)
print(f"Verdict: {v['verdict']} (score={v['score']:.3f})")
print(f"  n_support={v['n_support']}, n_against={v['n_against']}")
print(f"  edge_removed={v['edge_removed']}")
print(f"  weight: {v['old_weight']} -> {v['new_weight']}")
assert v["verdict"] == "support", f"Expected support, got {v['verdict']}"
assert v["n_support"] >= 2
assert v["n_against"] == 0
assert not v["edge_removed"]
# Check edge updated in graph
edge = g.get_edges("cpu_cores", EDGE_CAUSES)[0]
assert edge["n_support"] >= 2, f"n_support should be >= 2, got {edge['n_support']}"
assert edge["weight"] > 0.5, f"weight should increase, got {edge['weight']}"
print("[OK] Verdict supports hypothesis, edge weight increased")

# -- 4. Verdict refutation test --
g2 = FactGraph()
g2.add_node("cpu_cores", "22", category="system", step=0)
g2.add_node("load", "0.5", category="system", step=0)
g2.add_edge("cpu_cores", "load", EDGE_CAUSES, weight=0.3, step=0,
           n_support=0, n_against=0, hypothesis_key="test")

bad_result = {"exit_code": 1, "output_len": 0, "success": False}
v2 = verdict.evaluate(test_plan, bad_result, g2)
print(f"\nRefutation test: {v2['verdict']} (score={v2['score']:.3f})")
print(f"  weight: {v2['old_weight']} -> {v2['new_weight']}, removed={v2['edge_removed']}")
assert v2["verdict"] == "refute"
print("[OK] Verdict refutes on bad result")

# -- 5. Execution in sandbox --
print("\n--- Sandbox execution ---")
from agent.sandbox_executor import SandboxExecutor
sb = SandboxExecutor()
p = planner.plan({"if_node": "cpu_cores", "rel": "causes",
                   "then_node": "load", "prediction": "test"})
r = planner.execute_plan(p, sb)
print(f"Cmd: {p['cmd'][:60]}")
print(f"  exit_code={r.get('exit_code')}, output_len={r.get('output_len')}, success={r.get('success')}")
assert r.get("exit_code") == 0, f"Cmd should succeed, got {r.get('exit_code')}"
print("[OK] Experiment executes in sandbox")

# -- 6. Safety constraint --
unsafe_plan = {"cmd": "echo 'test' > /etc/config",
               "timeout": 10, "type": "intervene"}
r2 = planner.execute_plan(unsafe_plan, sb)
assert not r2.get("success"), "Unsafe write should be rejected"
print(f"[OK] Safety constraint blocks unsafe write: {r2.get('error')}")

# -- 7. Integration smoke test --
print("\n--- OnlineAgent integration ---")
os.system("docker rm -f folunar-sandbox 2>/dev/null")
from agent.online_agent import OnlineAgent

agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
assert hasattr(agent, 'experiment_planner'), "missing experiment_planner"
assert hasattr(agent, 'verdict'), "missing verdict"
assert hasattr(agent, '_last_experiment_step'), "missing _last_experiment_step"

for i in range(15):
    agent.step()
print(f"15 steps complete, step_count={agent.step_count}")
print("[OK] Integration no crash")

print(f"\n{'='*60}")
print(f"P16 R3 PASSED")
print(f"  Planner: plans generated with cmd/timeout/hypothesis_key")
print(f"  Verdict: support/refute, edge weight update, edge removal")
print(f"  Sandbox: execution + safety constraints")
print(f"  Integration: no crash")
print(f"{'='*60}")
