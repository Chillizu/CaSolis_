"""
Multi-step plan execution demo:
  GoalGenerator.decide_plan() → generates structured plan
  → execute each step with DeepSeek
  → save files per step
  → log complete trace
"""
import sys, os, json, time
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent
from agent.planner import Plan

OUT = "data/plan_demo"
os.makedirs(f"{OUT}/files", exist_ok=True)

agent = OnlineAgent(
    buffer_size=100, train_interval=99, batch_size=16,
    lr=1e-4, conductor_gate=0.7, mode="auto",
    api_backend="deepseek", model="deepseek-v4-flash",
)

log_file = open(f"{OUT}/run.jsonl", "w", encoding="utf-8")
cw = agent.creative_writer

# Set higher plan probability for demo
agent.goal_generator._plan_probability = 1.0

print("=== 阶段1: GoalGenerator 生成计划 ===\n")
plan = agent.goal_generator.decide_plan(agent.workbench, 0)
if not plan:
    print("No plan generated, forcing one...")
    from agent.planner import generate_plan
    plan = generate_plan("kernel-deep-probe", "/proc/sys kernel parameters", "plan_demo")
    agent.goal_generator._active_plan = plan

print(f"\n计划: {plan.plan_id}")
print(f"主题: {plan.topic}")
print(f"步骤 ({len(plan.steps)}步):")
for s in plan.steps:
    deps = s['deps'] if s['deps'] else "无"
    status = "等待" if not s['done'] else "完成"
    print(f"  [{s['id']}] {s['style']:10s} 依赖={deps}  {status}")
    print(f"       {s['desc']}")

print("\n=== 阶段2: 逐步执行 ===")
cycle = 0
while not plan.done:
    current = plan.current_step
    if not current:
        break
    cycle += 1
    print(f"\n--- 步骤 {current['id']} ({cycle}/{plan.remaining()}) ---")
    print(f"意图: {current['desc'][:80]}")

    # Build prompt
    prompt = cw.build_prompt(agent.workbench, current['style'], current['desc'])
    print(f"Prompt ({len(prompt)} chars)")

    # DeepSeek generate
    start_t = time.time()
    raw = cw.generate(prompt, timeout=30.0)
    elapsed = time.time() - start_t

    if not raw:
        print(f"  [失败] DeepSeek 无返回, 跳过此步")
        plan.mark_step_done(current['id'])
        continue

    # Clean fences
    clean = raw
    if clean.startswith("```"):
        first_nl = clean.find("\n")
        if first_nl > 0:
            clean = clean[first_nl+1:]
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    ext = ".py" if (".py" in raw[:50] or current['style'] in ("code",)) else ".md"
    fname = f"step{current['id']}_{current['style']}_{plan.plan_id}{ext}"
    filepath = f"{OUT}/files/{fname}"
    with open(filepath, "w") as f:
        f.write(clean)

    print(f"  DeepSeek: {len(raw)}B -> {len(clean)}B clean ({elapsed:.1f}s)")
    print(f"  保存: {fname}")
    print(f"  预览: {clean[:150]}")

    # Record
    plan.mark_step_done(current['id'], filepath)
    log_entry = {
        "event": "plan_step",
        "plan_id": plan.plan_id,
        "step_id": current['id'],
        "style": current['style'],
        "desc": current['desc'],
        "generation": {
            "elapsed_s": round(elapsed, 2),
            "raw_length": len(raw),
            "clean_length": len(clean),
            "raw_full": raw,
        },
        "file": fname,
    }
    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    log_file.flush()

    # Push result to workbench for next step
    agent.workbench._add_fact(
        f"plan_{plan.plan_id}_step{current['id']}",
        clean[:80], "PLAN",
        current['desc'][:60],
        0, category="content",
    )

print(f"\n=== 完成! ===")
print(f"计划 {plan.plan_id}: {len(plan.steps)}步, 全部完成")
print(f"日志: {OUT}/run.jsonl")
print(f"文件: {OUT}/files/")

agent.sandbox.execute("docker rm -f folunar-sandbox", timeout=5)
log_file.close()
