"""
Creative Marathon v4 — full pipeline logging
Every output traces: GoalGenerator decision → Prompt → DeepSeek response → File
"""
import sys, os, json, time
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent

OUT = "data/marathon_v4"
os.makedirs(f"{OUT}/files", exist_ok=True)

agent = OnlineAgent(
    buffer_size=100, train_interval=99, batch_size=16,
    lr=1e-4, conductor_gate=0.7, mode="auto",
    api_backend="deepseek", model="deepseek-v4-flash",
)

log_file = open(f"{OUT}/run.jsonl", "w", encoding="utf-8")

def log_decision(step):
    """Log GoalGenerator's decision before step consumes it"""
    intention = agent.goal_generator.decide_creative_intention(agent.workbench, step)
    entry = {
        "event": "decision",
        "step": step,
        "style": intention["style"],
        "intention": intention["intention"],
        "category": intention["category"],
        "tag_history": list(getattr(agent.goal_generator, '_tag_history', [])[-5:]),
    }
    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_file.flush()
    return entry

TOTAL_STEPS = 5000
known_sandbox = set()

for step in range(1, TOTAL_STEPS + 1):
    # Log GoalGenerator decision before step (matches internal trigger at step%5==0)
    if step % 5 == 0:
        log_decision(step)

    agent.step()

    # Track new files in sandbox every 25 steps
    if step % 25 == 0:
        r = agent.sandbox.execute(
            'find /tmp -maxdepth 1 -name "llm_*" -type f 2>/dev/null', timeout=3)
        if r and r.stdout:
            current = set(f.strip() for f in r.stdout.strip().split("\n") if f.strip())
            new = current - known_sandbox
            for path in sorted(new):
                content = agent.sandbox.execute(f'cat "{path}"', timeout=3)
                if content and content.stdout:
                    raw = content.stdout
                    fname = path.replace("/tmp/", f"step{step}_")
                    with open(f"{OUT}/files/{fname}", "w") as f:
                        f.write(raw)

                    # Match to the nearest decision (every 5 steps)
                    decision_step = (step // 5) * 5
                    entry = {
                        "event": "output",
                        "step": step,
                        "file": fname,
                        "size": len(raw),
                        "path": path,
                        "decision_at_step": decision_step,
                        "raw_preview": raw[:500],
                        "full_raw": raw,
                    }
                    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(f"[{step:5d}] {fname} ({len(raw)}B)")
            known_sandbox = current

log_file.write(json.dumps({"event": "done", "steps": TOTAL_STEPS}, ensure_ascii=False) + "\n")
log_file.close()
agent.sandbox.execute("docker rm -f folunar-sandbox", timeout=5)
print(f"\nDone. Output in {OUT}/")
