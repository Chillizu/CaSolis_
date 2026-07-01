"""
Full pipeline trace: GoalGenerator decision → Prompt → DeepSeek output → File
"""
import sys, os, json, time
sys.path.insert(0, "/home/chillizu/Projects/Folunar_")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

from agent.online_agent import OnlineAgent
from agent.workbench import Workbench
from agent.goal_generator import GoalGenerator
from agent.creative_writer import CreativeWriter

agent = OnlineAgent(
    buffer_size=100, train_interval=99, batch_size=16,
    lr=1e-4, conductor_gate=0.7, mode="auto",
    api_backend="deepseek", model="deepseek-v4-flash",
)

log = open("data/detailed_pipeline.jsonl", "w", encoding="utf-8")
step = 0

for cycle in range(5):
    step += 5
    log_entry = {"cycle": cycle+1, "trigger_step": step}

    # === Phase 1: GoalGenerator decides ===
    intention = agent.goal_generator.decide_creative_intention(agent.workbench, step)
    log_entry["goal_generator"] = {
        "style": intention["style"],
        "category": intention["category"],
        "intention_text": intention["intention"],
        "tag_history": agent.goal_generator._tag_history[-3:] if hasattr(agent.goal_generator, '_tag_history') else [],
    }
    print(f"\n=== Cycle {cycle+1} ===")
    print(f"[GoalGenerator] style={intention['style']}")
    print(f"[GoalGenerator] intention={intention['intention']}")

    # === Phase 2: Build prompt ===
    prompt = agent.creative_writer.build_prompt(
        agent.workbench, intention["style"], intention["intention"])
    log_entry["prompt"] = {"style": intention["style"], "length": len(prompt), "text": prompt}
    print(f"[Prompt] ({len(prompt)} chars)")
    print(prompt[:200] + "...\n")

    # === Phase 3: DeepSeek generates ===
    text = agent.creative_writer.generate(prompt, timeout=15.0)
    if text:
        # Determine file extension
        ext = ".py" if text.startswith("```") or intention["style"] in ("code",) else ".md"
        # Clean markdown fences
        clean = text
        if clean.startswith("```"):
            first_nl = clean.find("\n")
            if first_nl > 0:
                clean = clean[first_nl+1:]
            if clean.endswith("```"):
                clean = clean[:-3].strip()
        
        log_entry["deepseek"] = {
            "raw_length": len(text),
            "clean_length": len(clean),
            "raw_preview": text[:300],
            "full_raw": text,
        }
        print(f"[DeepSeek] {len(text)} chars raw -> {len(clean)} chars clean")
        print(f"[Output] First 200 chars:")
        print(clean[:200])
        
        # === Phase 4: Save to sandbox (simulate) ===
        path = f"/tmp/demo_llm_{intention['style']}_{step}.md"
        if ext == ".py":
            path = f"/tmp/demo_llm_code_{step}.py"
        encoded = clean.encode("utf-8").hex()
        agent.sandbox.execute(f"echo '{encoded}' | xxd -r -p > {path}", timeout=5)
        
        # Copy to our local files
        with open(f"data/detailed_output_{cycle+1}{ext}", "w") as f:
            f.write(clean)
        
        log_entry["file"] = {"path": path, "size": len(clean), "type": "code" if ext == ".py" else "content"}
        print(f"[Saved] {path} ({len(clean)}B)")
    else:
        log_entry["deepseek"] = {"raw_length": 0, "error": "No response"}
        print("[DeepSeek] No response")

    # Push cycle to workbench (so next cycle sees new data)
    if text:
        agent.workbench._add_fact(
            f"llm_demo_{step}", clean[:80], "LLM",
            f"demo:{intention['intention'][:50]}",
            step, category="content")

    log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    log.flush()

log.close()
print(f"\nFull log saved to data/detailed_pipeline.jsonl")
agent.sandbox.execute("docker rm -f folunar-sandbox 2>/dev/null", timeout=5)
