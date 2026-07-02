"""
Inline Creative Marathon — 每步: 决策→prompt→生成→文件, 完整追踪
"""
import sys, os, json, time
sys.path.insert(0, "/home/chillizu/Projects/CaSolis_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent
from agent.creative_writer import CreativeWriter

OUT = "data/marathon_inline"
os.makedirs(f"{OUT}/files", exist_ok=True)

agent = OnlineAgent(
    buffer_size=100, train_interval=99, batch_size=16,
    lr=1e-4, conductor_gate=0.7, mode="auto",
    api_backend="deepseek", model="deepseek-v4-flash",
)
cw = agent.creative_writer

log_file = open(f"{OUT}/run.jsonl", "w", encoding="utf-8")
total_generated = 0

def log(entry):
    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_file.flush()

log({"event": "start", "time": time.time()})

TOTAL_STEPS = 2000

for step in range(1, TOTAL_STEPS + 1):
    # === 正常的 agent 步进 (收集事实/跑命令/进化工作栏) ===
    agent.step()

    # === 每 10 步: 内联创作循环 ===
    if step % 10 == 0:
        # ── Phase 1: GoalGenerator 决策 ──
        intention = agent.goal_generator.decide_creative_intention(agent.workbench, step)
        style = intention["style"]
        intention_text = intention["intention"]

        # ── Phase 2: 构建 Prompt ──
        prompt = cw.build_prompt(agent.workbench, style, intention_text)

        # ── Phase 3: DeepSeek 生成 ──
        start_t = time.time()
        raw = cw.generate(prompt, timeout=25.0)
        elapsed = time.time() - start_t

        if not raw:
            log({
                "event": "skip",
                "step": step,
                "style": style,
                "intention": intention_text,
                "reason": "no_response",
            })
            continue

        # 清理 Markdown 代码围栏
        clean = raw
        if clean.startswith("```"):
            first_nl = clean.find("\n")
            if first_nl > 0:
                clean = clean[first_nl+1:]
            if clean.endswith("```"):
                clean = clean[:-3].strip()

        # ── Phase 4: 保存文件 ──
        ext = ".py" if (raw.startswith("```") or style in ("code", "script")) else ".md"
        fname = f"step{step}_{style}_{total_generated+1}{ext}"
        with open(f"{OUT}/files/{fname}", "w") as f:
            f.write(clean)
        total_generated += 1

        # ── 记录完整链路 ──
        entry = {
            "event": "creation",
            "id": total_generated,
            "step": step,
            "style": style,
            "intention": intention_text,
            "tag_history": list(getattr(agent.goal_generator, '_tag_history', [])[-5:]),
            "prompt": {
                "template": style,
                "length": len(prompt),
                "text": prompt,
            },
            "generation": {
                "elapsed_s": round(elapsed, 2),
                "raw_length": len(raw),
                "clean_length": len(clean),
                "raw_full": raw,
            },
            "file": {
                "name": fname,
                "size": len(clean),
            }
        }
        log(entry)
        print(f"[{step:5d}] #{total_generated} {style:10s} {len(clean):,}B ({elapsed:.1f}s) {intention_text[:60]}")

        # ── Phase 5: 推入工作栏(供下一步使用) ──
        agent.workbench._add_fact(
            f"llm_inline_{step}", clean[:80], "LLM",
            f"inline:{intention_text[:50]}",
            step, category="content",
        )

        # 每 20 次记录中间状态
        if total_generated % 20 == 0:
            log({"event": "checkpoint", "step": step, "generated": total_generated})

log({"event": "done", "steps": TOTAL_STEPS, "total_generated": total_generated})
log_file.close()

print(f"\n=== 完成 ===")
print(f"步数: {TOTAL_STEPS}")
print(f"LLM 创作: {total_generated} 次")
print(f"总输出: {sum(os.path.getsize(f'{OUT}/files/{f}') for f in os.listdir(f'{OUT}/files/')):,}B")
print(f"日志: {OUT}/run.jsonl")
print(f"文件: {OUT}/files/")
agent.sandbox.execute("docker rm -f casolis-sandbox", timeout=5)
