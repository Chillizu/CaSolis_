"""
P20 behavior comparison: 200 steps with vs. without salience/habit.
Runs two separate subprocesses with identical starting state, then compares logs.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = ROOT / "run_logs"


def reset_state():
    """Reset sandbox and persistent data to baseline."""
    subprocess.run(["docker", "rm", "-f", "casolis-sandbox"], capture_output=True)
    subprocess.run(["git", "checkout", "--", "data/persistent/"], cwd=ROOT, check=True)
    # Remove any new run logs so they don't interfere with file discovery
    for p in LOG_DIR.glob("run_*.jsonl"):
        p.unlink()


def run_agent(label: str, disable_p20: bool) -> Path:
    """Run 200 steps and return the produced log file."""
    env = os.environ.copy()
    if disable_p20:
        env["P20_DISABLE"] = "1"

    script = f"""
import sys, os, json
sys.path.insert(0, "{ROOT}")
os.environ["HF_HUB_OFFLINE"] = "1"
if os.environ.get("P20_DISABLE") == "1":
    from agent import salience, habit
    salience.SalienceSignal.update = lambda *a, **k: None
    salience.SalienceSignal.recent_mean = lambda *a, **k: 0.5
    salience.SalienceSignal.recent_max = lambda *a, **k: 0.5
    salience.SalienceSignal.get_stats = lambda *a, **k: {{}}
    habit.HabitSystem.suggest = lambda *a, **k: None
    habit.HabitSystem.register = lambda *a, **k: None
    habit.HabitSystem.get_stats = lambda *a, **k: {{}}
from agent.online_agent import OnlineAgent
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
for _ in range(200):
    agent.step()
print("FINAL success_rate=" + str(agent.success_count / max(agent.step_count, 1)) + " steps=" + str(agent.step_count))
"""
    subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=ROOT,
        env=env,
        timeout=600,
        check=True,
    )
    # Find the log file produced by this run
    logs = sorted(LOG_DIR.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    assert logs, f"No run log produced for {label}"
    log = logs[-1]
    dest = log.with_name(f"{label}_{log.name}")
    log.rename(dest)
    return dest


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def analyze_log(path: Path) -> dict:
    """Parse a JSONL run log and compute behavior metrics."""
    intents = []
    success = 0
    total = 0
    facts_start = None
    facts_end = 0
    habit_triggers = 0
    sal_updates = 0
    last_intent = None
    max_streak = 0
    cur_streak = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "step":
                continue
            total += 1
            intent = e.get("intent", "UNKNOWN")
            intents.append(intent)
            if e.get("source") == "habit":
                habit_triggers += 1
            if "salience" in str(e.get("print", "")).lower() or e.get("habit_stats"):
                sal_updates += 1  # rough proxy
            if e.get("exit_code") == 0:
                success += 1
            if facts_start is None:
                facts_start = e.get("facts_before", 0)
            facts_end = e.get("facts_before", 0) + e.get("facts_delta", 0)

            if intent == last_intent:
                cur_streak += 1
            else:
                cur_streak = 1
            max_streak = max(max_streak, cur_streak)
            last_intent = intent

    counter = Counter(intents)
    return {
        "total_steps": total,
        "success_rate": success / total if total else 0,
        "intent_entropy": entropy(counter),
        "unique_intents": len(counter),
        "top_3_intents": counter.most_common(3),
        "max_consecutive_same_intent": max_streak,
        "facts_growth": facts_end - (facts_start or 0),
        "habit_triggers": habit_triggers,
        "salience_updates": sal_updates,
    }


def main():
    print("=" * 60)
    print("P20 behavior comparison: 200 steps x 2 runs")
    print("=" * 60)

    reset_state()
    print("\n[Run 1/2] P20 ENABLED")
    log_enabled = run_agent("p20_enabled", disable_p20=False)

    reset_state()
    print("\n[Run 2/2] P20 DISABLED")
    log_disabled = run_agent("p20_disabled", disable_p20=True)

    print("\n[Analysis]")
    m_enabled = analyze_log(log_enabled)
    m_disabled = analyze_log(log_disabled)

    print(f"\n--- P20 ENABLED ---")
    for k, v in m_enabled.items():
        print(f"  {k}: {v}")

    print(f"\n--- P20 DISABLED ---")
    for k, v in m_disabled.items():
        print(f"  {k}: {v}")

    print(f"\n--- Delta (P20 - DISABLED) ---")
    for k in m_enabled:
        if isinstance(m_enabled[k], (int, float)):
            print(f"  {k}: {m_enabled[k] - m_disabled[k]:+.4f}")

    print("\n[Cleanup]")
    reset_state()
    print("Done.")


if __name__ == "__main__":
    main()
