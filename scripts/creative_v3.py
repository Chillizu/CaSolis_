"""
Creative Marathon v3 — track outputs from sandbox, not competing with agent
"""
import sys, os, json, time
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent

OUT = "data/marathon_v3"
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

TOTAL_STEPS = 10000
CHECK_INTERVAL = 200
known_sandbox = set()
last_ck = 0
llm_count = 0

for step in range(1, TOTAL_STEPS + 1):
    agent.step()

    # Track sandbox files (don't compete with async_result)
    if step % 25 == 0:
        r = agent.sandbox.execute(
            'find /tmp -maxdepth 1 -name "llm_*" -type f 2>/dev/null', timeout=3)
        if r and r.stdout:
            current = set(f.strip() for f in r.stdout.strip().split("\n") if f.strip())
            new = current - known_sandbox
            for path in sorted(new):
                content = agent.sandbox.execute(f'cat "{path}"', timeout=3)
                if content and content.stdout:
                    llm_count += 1
                    fname = path.replace("/tmp/", f"step{step}_")
                    with open(f"{OUT}/files/{fname}", "w") as f:
                        f.write(content.stdout)
                    sz = len(content.stdout)
                    ct = "script" if path.endswith(".py") else "content"
                    write_log({"step": step, "file": fname, "size": sz, "type": ct})
                    print(f"[{step:5d}] {fname} ({sz}B)")
            known_sandbox = current

    if step - last_ck >= CHECK_INTERVAL:
        write_log({"checkpoint": step, "llm": llm_count, "sandbox": len(known_sandbox)})
        last_ck = step

write_log({"event": "done", "steps": TOTAL_STEPS, "llm_files": llm_count})

# Final sandbox copy
r = agent.sandbox.execute('find /tmp -maxdepth 1 -name "llm_*" -type f 2>/dev/null', timeout=5)
if r and r.stdout:
    for path in r.stdout.strip().split("\n"):
        if not path.strip(): continue
        c = agent.sandbox.execute(f'cat "{path}"', timeout=3)
        if c and c.stdout:
            fname = f"sandbox_{path.replace('/tmp/','')}"
            with open(f"{OUT}/files/{fname}", "w") as f:
                f.write(c.stdout)

agent.sandbox.execute("docker rm -f folunar-sandbox", timeout=5)
log.close()
print(f"\nDone. {llm_count} LLM files in {OUT}/")
