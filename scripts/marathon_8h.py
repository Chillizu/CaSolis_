"""
8 小时马拉松 — 完整日志记录所有模型细节与产出
"""
import sys, os, json, time, torch
sys.path.insert(0, "/home/chillizu/Projects/CaSolis_")
os.environ["HF_HUB_OFFLINE"] = "1"

LOG_DIR = "data/marathon_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"marathon_{int(time.time())}.jsonl")
MODEL_DUMP = os.path.join(LOG_DIR, f"model_snapshot_{int(time.time())}.json")

def log(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── 1. 记录所有模型细节 ──
from agent.online_agent import OnlineAgent
from agent.world_model_v5 import WorldModelV5
from agent.do_calculus import DoCalculusEngine
from agent.intuition_buffer import IntuitionBuffer
from agent.self_model import SelfModel

model_snapshot = {
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_threads": torch.get_num_threads(),
    "device": "cpu",
}

log({"type": "marathon_start", "model_snapshot": model_snapshot})
with open(MODEL_DUMP, "w") as f:
    json.dump(model_snapshot, f, indent=2)

# ── 2. 启动 Agent ──
agent = OnlineAgent(buffer_size=2000, train_interval=15, batch_size=64,
                    lr=1e-4, conductor_gate=0.65, mode="auto")

# 记录所有模型架构
arch = {
    "world_model_v5": {
        "params": agent.world_model_v5._param_count,
        "gru_hidden": agent.world_model_v5.core.gru.hidden_size,
        "cond_dim": agent.world_model_v5.core.state_proj.out_features,
    },
    "intuition_buffer": {
        "capacity": agent.intuition_buffer.capacity,
        "top_k": agent.intuition_buffer.top_k,
    },
    "classifier": {
        "type": type(agent.classifier).__name__,
    },
    "world_model_v4": {
        "params": sum(p.numel() for p in agent.world_model_v4.parameters()) if hasattr(agent, 'world_model_v4') else 0,
    },
    "creative_writer": {
        "model": agent.creative_writer.model if agent.creative_writer else "none",
        "backend": agent.creative_writer.api_backend if agent.creative_writer else "none",
    },
}

log({"type": "architecture", "arch": arch})

# 记录所有参数
total_params = agent.world_model_v5._param_count
if hasattr(agent, 'world_model_v4') and hasattr(agent.world_model_v4, 'parameters'):
    total_params += sum(p.numel() for p in agent.world_model_v4.parameters())

log({"type": "total_params", "total": total_params})

# ── 3. 主循环 ──
DEADLINE = time.time() + 8 * 3600  # 8 小时
INTERVAL = 500  # 每 500 步记录一次

step = 0
created_files = []
hypotheses_found = []
self_reflects = []
causal_edges = []
last_save = 0

print(f"Marathon started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log: {LOG_FILE}")
print(f"Model snapshot: {MODEL_DUMP}")
print(f"Deadline: +8h\n")

while time.time() < DEADLINE:
    try:
        succ, rwd = agent.step()
        step += 1

        if step % INTERVAL == 0:
            elapsed = time.time() - (DEADLINE - 8*3600)
            v5_loss = agent.world_model_v5.train_losses[-1] if agent.world_model_v5.train_losses else None
            mode = agent.current_mode
            ib_size = agent.intuition_buffer.size

            # 收集创建的文件
            r = agent.sandbox.execute("find /tmp -name 'llm_*' -o -name 'script_*' -o -name 'discover_*' 2>/dev/null | head -10")
            if r and r.stdout:
                created_files = r.stdout.strip().split('\n')

            # 收集因果边
            if hasattr(agent.workbench, 'graph') and agent.workbench.graph:
                for src, edges in agent.workbench.graph.edges.items():
                    for e in edges:
                        if e.get("causal_score", 0) > 0.3:
                            causal_edges.append(f"{src}→{e['to']}:{e.get('causal_score',0):.2f}")
                causal_edges = causal_edges[-20:]

            entry = {
                "type": "checkpoint",
                "step": step,
                "elapsed_h": round(elapsed / 3600, 2),
                "success": f"{agent.success_count}/{agent.step_count}",
                "success_rate": round(agent.success_count / max(agent.step_count, 1), 3),
                "mode": mode,
                "v5_train_steps": len(agent.world_model_v5.train_losses),
                "v5_last_loss": v5_loss,
                "ib_size": ib_size,
                "total_reward": round(agent.total_reward if hasattr(agent, 'total_reward') else 0, 1),
                "created_files": created_files[-5:],
                "causal_edges": causal_edges[-5:],
                "self_reflect_count": len(self_reflects),
            }
            log(entry)

            # 每 2000 步保存完整状态
            if step - last_save >= 2000:
                if hasattr(agent, 'world_model_v5'):
                    agent.world_model_v5.save(f"data/persistent/world_model_v5.pt")
                if hasattr(agent, 'self_model'):
                    agent.self_model.save()
                last_save = step

            print(f"[{step:6d}] {elapsed/3600:.1f}h succ={entry['success_rate']*100:.0f}% "
                  f"mode={mode} v5={entry['v5_train_steps']} ib={ib_size} "
                  f"files={len(created_files)}")

    except Exception as e:
        log({"type": "error", "step": step, "error": str(e)})
        print(f"[ERROR] {e}")
        time.sleep(1)

# ── 4. 最终总结 ──
final_elapsed = time.time() - (DEADLINE - 8*3600)
summary = {
    "type": "marathon_end",
    "total_steps": step,
    "total_time_h": round(final_elapsed / 3600, 2),
    "final_success": f"{agent.success_count}/{agent.step_count}",
    "final_success_rate": round(agent.success_count / max(agent.step_count, 1), 3),
    "final_total_reward": round(agent.total_reward if hasattr(agent, 'total_reward') else 0, 1),
    "final_mode": agent.current_mode,
    "v5_total_train": len(agent.world_model_v5.train_losses),
    "ib_final_size": agent.intuition_buffer.size,
    "total_params": total_params,
    "created_files": created_files,
    "self_reflects": self_reflects,
    "causal_edges": causal_edges,
}

log(summary)

# 列出所有创建的文件
print(f"\n{'='*60}")
print(f"MARATHON COMPLETE: {step} steps in {final_elapsed/3600:.1f}h")
print(f"Success: {agent.success_count}/{agent.step_count} ({agent.success_count/max(agent.step_count,1)*100:.1f}%)")
print(f"Total params: {total_params:,}")
print(f"Log: {LOG_FILE}")
print(f"{'='*60}")

# 显示所有创建的文件
r = agent.sandbox.execute("find /tmp -name 'llm_*' -o -name 'script_*' -o -name 'discover_*' -o -name 'self_intent_*' -o -name 'creation_*' 2>/dev/null | sort")
if r and r.stdout:
    files = [f for f in r.stdout.strip().split('\n') if f]
    print(f"\nCreated {len(files)} files:")
    for f in files:
        c = agent.sandbox.execute(f"wc -c {f} 2>/dev/null")
        size = c.stdout.strip().split()[0] if c and c.stdout else "?"
        print(f"  {f} ({size}B)")

os.system("docker rm -f casolis-sandbox 2>/dev/null")
print("\nDone.")
