"""
Long run — track creative output from the two-layer architecture
"""
import sys, os, json, time, torch
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent

LOG_DIR = "data/marathon_logs"
os.makedirs(LOG_DIR, exist_ok=True)
ts = int(time.time())
log_path = os.path.join(LOG_DIR, f"creative_long_{ts}.jsonl")
log_file = open(log_path, "w", encoding="utf-8")

def log(obj):
    log_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log_file.flush()

print("=" * 50)
print("Creative Long Run")
print(f"模型: deepseek-v4-flash (GoalGenerator 决定意图)")
print(f"日志: {log_path}")
print("=" * 50)

agent = OnlineAgent(
    buffer_size=100,
    train_interval=99,
    batch_size=16,
    lr=1e-4,
    conductor_gate=0.7,
    mode="auto",
    api_backend="deepseek",
    model="deepseek-v4-flash",
)

log({"type": "start", "model": "deepseek-v4-flash", "architecture": "two_layer"})

INTERVAL = 200
start = time.time()
step = 0
last_ck = 0
llm_files_seen = set()

try:
    while time.time() - start < 3600:  # 1h max
        success, reward = agent.step()
        step += 1

        # Check for new LLM files
        try:
            r = agent.sandbox.execute('find /tmp -name "llm_*" -type f 2>/dev/null', timeout=3)
            if r and r.stdout:
                current = set(f.strip() for f in r.stdout.strip().split("\n") if f.strip())
                new = current - llm_files_seen
                if new:
                    for f in new:
                        sz = agent.sandbox.execute(f'wc -c "{f}"', timeout=3)
                        sz_s = sz.stdout.strip().split()[0] if sz and sz.stdout else '?'
                        print(f"  [NEW] {f} ({sz_s}B)")
                    llm_files_seen = current
        except:
            pass

        if step - last_ck >= INTERVAL:
            elapsed = (time.time() - start) / 3600
            sr = agent.success_count / max(step, 1)
            log({
                "type": "checkpoint", "step": step, "elapsed_h": round(elapsed, 2),
                "success_rate": round(sr, 3), "reward": round(agent.total_reward, 1),
                "llm_files": len(llm_files_seen),
            })
            last_ck = step

except KeyboardInterrupt:
    print("\nInterrupted")

elapsed = (time.time() - start) / 60
print(f"\n{'='*50}")
print(f"Run: {step} steps in {elapsed:.0f}min")
print(f"LLM files created: {len(llm_files_seen)}")
for f in sorted(llm_files_seen):
    print(f"  {f}")
print(f"{'='*50}")

log({"type": "final", "step": step, "llm_files": len(llm_files_seen),
     "llm_list": sorted(llm_files_seen)})
log_file.close()
