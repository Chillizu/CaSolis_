"""
Creative Marathon v2 — full output logging, everything preserved
"""
import sys, os, json, time, shutil
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent

# Output directory
OUT = "data/marathon_v2"
os.makedirs(f"{OUT}/files", exist_ok=True)

agent = OnlineAgent(
    buffer_size=100, train_interval=99, batch_size=16,
    lr=1e-4, conductor_gate=0.7, mode="auto",
    api_backend="deepseek", model="deepseek-v4-flash",
)

log = open(f"{OUT}/run.jsonl", "w", encoding="utf-8")

def write_log(entry):
    log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.flush()

write_log({"event": "start", "time": time.time()})

TOTAL_STEPS = 5000
CHECK_INTERVAL = 200
last_ck = 0
llm_entries = 0
goal_decisions = []

for step in range(1, TOTAL_STEPS + 1):
    success, reward = agent.step()

    # Check async result every step (the agent's check_async_result runs every 5)
    if step % 5 == 0:
        result = agent.creative_writer.check_async_result()
        if result and result.get("source") == "llm":
            content = result.get("content", "")
            style = result.get("style", "?")
            path = result.get("path", "?")
            quality = result.get("quality", 0)
            desc = result.get("desc", "")
            llm_entries += 1

            # Save file locally
            fname = f"llm_{step}_{style}_{llm_entries}.md"
            if content.startswith("```"):
                fname = f"llm_{step}_{style}_{llm_entries}.py"
            with open(f"{OUT}/files/{fname}", "w") as f:
                f.write(content)

            entry = {
                "step": step, "style": style, "quality": quality,
                "size": len(content), "file": fname, "desc": desc,
            }
            write_log(entry)
            print(f"[{step:5d}] {fname} ({len(content)}B, q={quality:.2f}) {desc[:60]}")

    if step - last_ck >= CHECK_INTERVAL:
        elapsed = (time.time() - (time.time() - (step / 4))) / 60  # rough
        r = agent.sandbox.execute(f'find /tmp -name "llm_*" -type f 2>/dev/null | wc -l', timeout=3)
        sandbox_count = r.stdout.strip() if r and r.stdout else "?"
        write_log({
            "checkpoint": step, "llm_total": llm_entries,
            "sandbox_files": sandbox_count,
        })
        last_ck = step

elapsed = (time.time() - (time.time() - (TOTAL_STEPS / 4))) / 60

write_log({"event": "done", "steps": TOTAL_STEPS, "llm_files": llm_entries})

# Copy files from sandbox
print("\n=== Copying from sandbox ===")
r = agent.sandbox.execute('find /tmp -name "llm_*" -type f 2>/dev/null', timeout=5)
if r and r.stdout:
    for path in r.stdout.strip().split("\n"):
        if not path.strip():
            continue
        c = agent.sandbox.execute(f'cat "{path}"', timeout=3)
        if c and c.stdout:
            safe = path.replace("/tmp/", f"{OUT}/files/sandbox_")
            with open(safe, "w") as f:
                f.write(c.stdout)
            print(f"  saved {path}")

agent.sandbox.execute("docker rm -f folunar-sandbox", timeout=5)
log.close()
print(f"\nDone. {llm_entries} LLM files in {OUT}/")
