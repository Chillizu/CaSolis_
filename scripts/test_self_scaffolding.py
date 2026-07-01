"""Unit tests for Self-Scaffolding plan execution."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.goal_generator import GoalGenerator
from agent.planner import Plan
from agent.workbench import Workbench


def test_decide_plan_returns_three_step_plan():
    wb = Workbench()
    # Seed some facts so tag_counts is not empty
    wb.facts = {
        "cpu_model": {"value": "Intel", "category": "cpu"},
        "cpu_core": {"value": 8, "category": "cpu"},
    }
    gg = GoalGenerator()
    gg._plan_probability = 1.0  # 测试中强制生成计划
    plan = gg.decide_plan(wb, step=1)
    assert plan is not None, "decide_plan should return a plan when facts exist"
    assert isinstance(plan, Plan), "decide_plan should return a Plan instance"
    assert len(plan.steps) == 3, f"expected 3 steps, got {len(plan.steps)}"
    # deps: first step none, others depend on prior steps
    assert plan.steps[0]["deps"] == []
    assert plan.steps[1]["deps"] == [0]
    assert plan.steps[2]["deps"] == [0, 1]
    print("[PASS] test_decide_plan_returns_three_step_plan")


def test_plan_done_after_all_steps():
    wb = Workbench()
    wb.facts = {
        "mem_total": {"value": 16000000, "category": "memory"},
    }
    gg = GoalGenerator()
    gg._plan_probability = 1.0
    plan = gg.decide_plan(wb, step=2)
    assert plan is not None
    assert not plan.done
    for s in plan.steps:
        plan.mark_step_done(s["id"], file_path=f"/tmp/step_{s['id']}.txt")
    assert plan.done, "plan should be done after all steps marked"
    print("[PASS] test_plan_done_after_all_steps")


def test_record_plan_outcome_updates_stats():
    wb = Workbench()
    wb.facts = {
        "cpu_model": {"value": "AMD", "category": "cpu"},
    }
    gg = GoalGenerator()
    gg._plan_probability = 1.0
    plan = gg.decide_plan(wb, step=3)
    assert plan is not None
    # mark first two steps ok, last step fail -> success rate 2/3
    plan.mark_step_done(plan.steps[0]["id"], "/tmp/0.txt")
    plan.mark_step_done(plan.steps[1]["id"], "/tmp/1.txt")
    plan.steps[2]["success"] = False
    plan.mark_step_done(plan.steps[2]["id"], "")
    gg.record_plan_outcome(plan, 2 / 3)
    topic = plan.topic or "unknown"
    assert topic in gg._plan_topic_stats, "topic stats should be recorded"
    stats = gg._plan_topic_stats[topic]
    assert stats["ok"] == 2, f"expected ok=2, got {stats['ok']}"
    assert stats["fail"] == 1, f"expected fail=1, got {stats['fail']}"
    print("[PASS] test_record_plan_outcome_updates_stats")


def test_topic_success_rate_prior():
    gg = GoalGenerator()
    rate = gg._topic_success_rate("nonexistent_topic")
    assert rate == 0.5, f"expected neutral prior 0.5, got {rate}"
    print("[PASS] test_topic_success_rate_prior")


if __name__ == "__main__":
    test_decide_plan_returns_three_step_plan()
    test_plan_done_after_all_steps()
    test_record_plan_outcome_updates_stats()
    test_topic_success_rate_prior()
    print("\nAll Self-Scaffolding unit tests PASSED.")
