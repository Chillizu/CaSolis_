"""Compare Self-Scaffolding enabled vs disabled.
Runs two 50-step trials and prints behavior metrics.
"""
import json, math, os, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = ROOT / "run_logs"


def reset_state():
    subprocess.run(["docker", "rm", "-f", "casolis-sandbox"], capture_output=True)
    # remove persistent data and logs
    for p in (ROOT / "data" / "persistent").glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            import shutil
            shutil.rmtree(p)
    for p in LOG_DIR.glob("run_*.jsonl"):
        p.unlink()


def run_trial(label: str, disable_scaffold: bool) -> Path:
    env = os.environ.copy()
    if disable_scaffold:
        env["SELF_SCAFFOLD_DISABLE"] = "1"
    else:
        env.pop("SELF_SCAFFOLD_DISABLE", None)

    script = """
import sys, os
sys.path.insert(0, '{root}')
os.environ['HF_HUB_OFFLINE'] = '1'
from agent.online_agent import OnlineAgent
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode='auto')
for _ in range(50):
    agent.step()
print('__RESULT__ success={{}}/{{}}'.format(agent.success_count, agent.step_count))
""".format(root=ROOT.as_posix())
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180,
        )
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for {label}:")
        print(e.stdout[-3000:] if e.stdout else "")
        raise
    finally:
        os.unlink(script_path)

    logs = sorted(LOG_DIR.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    assert logs, f"No run log for {label}"
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
    intents = []
    sources = []
    plan_ids = set()
    plan_success = []
    success_count = 0
    total = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") != "step":
                continue
            total += 1
            intents.append(r.get("intent", ""))
            sources.append(r.get("source", ""))
            if r.get("success"):
                success_count += 1
            pid = r.get("plan_id", "")
            if pid:
                plan_ids.add(pid)
                psr = r.get("plan_success_rate", -1.0)
                if psr >= 0:
                    plan_success.append(psr)
    intent_counter = Counter(intents)
    return {
        "total_steps": total,
        "success_rate": success_count / max(total, 1),
        "intent_entropy": entropy(intent_counter),
        "intent_dist": dict(intent_counter),
        "plan_ids": list(plan_ids),
        "plan_count": len(plan_ids),
        "avg_plan_success_rate": sum(plan_success) / max(len(plan_success), 1),
    }


def main():
    print("=" * 60)
    print("Self-Scaffolding comparison: 50 steps with vs without")
    print("=" * 60)

    reset_state()
    print("Run 1: Self-Scaffolding ENABLED")
    enabled_log = run_trial("enabled", disable_scaffold=False)
    enabled_metrics = analyze_log(enabled_log)

    reset_state()
    print("Run 2: Self-Scaffolding DISABLED")
    disabled_log = run_trial("disabled", disable_scaffold=True)
    disabled_metrics = analyze_log(disabled_log)

    print("\n--- Metrics ---")
    print(f"Enabled:  {enabled_metrics}")
    print(f"Disabled: {disabled_metrics}")
    print("\n--- Delta ---")
    print(f"plan_count: +{enabled_metrics['plan_count'] - disabled_metrics['plan_count']}")
    print(f"success_rate: {enabled_metrics['success_rate']:.2%} vs {disabled_metrics['success_rate']:.2%}")
    print(f"intent_entropy: {enabled_metrics['intent_entropy']:.3f} vs {disabled_metrics['intent_entropy']:.3f}")

    assert enabled_metrics["plan_count"] > disabled_metrics["plan_count"], "Expected more plans when enabled"
    print("\n[OK] Self-Scaffolding comparison complete")


if __name__ == "__main__":
    main()
