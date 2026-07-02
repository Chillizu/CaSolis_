"""
DeepSeek 长程马拉松 — 使用 deepseek-v4-flash API
完全日志记录 + 模型快照 + 产出追踪
"""
import sys, os, json, time, torch

sys.path.insert(0, "/home/chillizu/Projects/CaSolis_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent

LOG_DIR = "data/marathon_logs"
os.makedirs(LOG_DIR, exist_ok=True)
ts = int(time.time())
log_path = os.path.join(LOG_DIR, f"deepseek_{ts}.jsonl")
snap_path = os.path.join(LOG_DIR, f"deepseek_snapshot_{ts}.json")
log_file = open(log_path, "w", encoding="utf-8")

def log(obj):
    log_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log_file.flush()

# ── 初始化 ──
print("=" * 50)
print("DeepSeek Marathon 启动")
print(f"模型: deepseek-v4-flash")
print(f"日志: {log_path}")
print(f"快照: {snap_path}")
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

# ── 模型架构快照 ──
snapshot = {
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version,
    "torch": torch.__version__,
    "device": "cpu",
    "model": "deepseek-v4-flash",
    "api_backend": "deepseek",
}
log({"type": "marathon_start", "model_snapshot": snapshot})

# 参数统计
total_params = agent.world_model_v5._param_count
if hasattr(agent, 'world_model_v4') and hasattr(agent.world_model_v4, 'parameters'):
    total_params += sum(p.numel() for p in agent.world_model_v4.parameters())

arch = {
    "world_model_v5": {"params": 364135, "gru_hidden": 160, "cond_dim": 80},
    "intuition_buffer": {"capacity": 1024, "top_k": 5},
    "classifier": {"type": "IntentClassifier"},
    "world_model_v4": {"params": 445550},
    "creative_writer": {"model": "deepseek-v4-flash", "backend": "deepseek"},
}
log({"type": "architecture", "arch": arch})
log({"type": "total_params", "total": total_params})

# ── 运行 ──
DEADLINE = 8 * 3600  # 8小时
INTERVAL = 500       # 每500步检查点
start_wall = time.time()
step = 0
last_checkpoint = 0

print(f"\nMarathon started: {snapshot['start_time']}")
print(f"Deadline: +8h\n")

try:
    while time.time() - start_wall < DEADLINE:
        success, reward = agent.step()
        step += 1
        if step - last_checkpoint >= INTERVAL:
            elapsed = (time.time() - start_wall) / 3600
            success_rate = agent.success_count / max(step, 1)
            cw = agent.creative_writer
            cw_stats = cw.get_stats() if cw else {}

            checkpoint = {
                "type": "checkpoint",
                "step": step,
                "elapsed_h": round(elapsed, 2),
                "success": f"{agent.success_count}/{step}",
                "success_rate": round(success_rate, 3),
                "mode": agent.current_mode,
                "v5_train_steps": getattr(agent.world_model_v5, '_train_steps', 0) if hasattr(agent, 'world_model_v5') else 0,
                "ib_size": getattr(getattr(agent, 'intuition_buffer', None), 'capacity', 0) if hasattr(agent, 'intuition_buffer') else 0,
                "total_reward": round(getattr(agent, 'total_reward', 0), 1),
                "llm_calls": cw_stats.get("total_calls", 0),
                "llm_success": cw_stats.get("llm_success", 0),
                "llm_fallback": cw_stats.get("fallback", 0),
            }
            log(checkpoint)
            print(f"  [{step:5d}] {elapsed:.1f}h  "
                  f"成功率{success_rate*100:.0f}%  "
                  f"reward={checkpoint['total_reward']:.0f}  "
                  f"LLM成功={cw_stats.get('llm_success', 0)}")

            # 检查沙箱中的 LLM 文件
            try:
                r = agent.sandbox.execute('find /tmp -name "llm_*" -type f 2>/dev/null', timeout=5)
                if r and r.stdout:
                    files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
                    if files:
                        total_size = 0
                        for f in files[:20]:
                            sz = agent.sandbox.execute(f'wc -c "{f}"', timeout=3)
                            if sz and sz.stdout:
                                total_size += int(sz.stdout.strip().split()[0])
                        print(f"    LLM文件: {len(files)}个 ({total_size}B)")
            except Exception:
                pass

            last_checkpoint = step

except KeyboardInterrupt:
    print("\nMarathon interrupted by user")

# ── 最终摘要 ──
elapsed = (time.time() - start_wall) / 3600
success_rate = agent.success_count / max(step, 1)
summary = {
    "type": "final",
    "step": step,
    "elapsed_h": round(elapsed, 2),
    "success": f"{agent.success_count}/{step}",
    "success_rate": round(success_rate, 3),
    "total_reward": round(getattr(agent, 'total_reward', 0), 1),
}
log(summary)
log_file.close()

# 保存模型快照
snapshot_data = {
    "step": step,
    "success_rate": success_rate,
    "total_reward": getattr(agent, 'total_reward', 0),
    "timestamp": time.time(),
}
import json as _json
with open(snap_path, "w") as f:
    _json.dump(snapshot_data, f, indent=2)

print(f"\n{'='*50}")
print(f"Marathon finished: {step} steps in {elapsed:.1f}h")
print(f"Success rate: {success_rate*100:.1f}%")
print(f"Log: {log_path}")
print(f"{'='*50}")
