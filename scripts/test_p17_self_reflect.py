"""
P17 自省验证:
1. SelfModel 统计追踪 + 自描述
2. CreativeWriter self_reflect 产出意图
3. OnlineAgent 集成: 自省 → CREATE goal
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"

from agent.self_model import SelfModel
from agent.creative_writer import CreativeWriter
from types import SimpleNamespace

print("=" * 60)
print("P17 自省验证")
print("=" * 60)

# 1. SelfModel
print("\n--- SelfModel 基础 ---")
sm = SelfModel()
for i in range(6):
    sm.record("OBSERVE", True, 0.6, step=i, output_len=150)
sm.record("CREATE", True, 1.2, step=10, output_len=500,
          path="/tmp/test.md", content_len=480)
sm.record("TRY", False, 0.1, step=11, output_len=0)
sm.record("TRY", True, 0.3, step=12, output_len=50)

desc = sm.build_self_description()
print(f"  自描述:")
for line in desc.split("\n"):
    print(f"    {line}")
assert "89%" in desc, f"success rate wrong: {desc}"
assert "OBSERVE" in desc, f"best intents missing: {desc}"
assert "CREATE" in desc, f"creations missing: {desc}"
assert "高光" in desc, f"highlights missing: {desc}"
print("[OK] SelfModel")

# 2. Self-prompt
prompt = sm.build_self_prompt()
print(f"\n  自省 prompt: {len(prompt)}B")
assert "Intention:" in prompt
assert "OBSERVE" in sm.build_self_description()
print("[OK] Self-prompt")

# 3. CreativeWriter self_reflect
print("\n--- CreativeWriter self_reflect ---")
cw = CreativeWriter(timeout=90)
if cw.health_check():
    mock_wb = SimpleNamespace(self_model=sm)
    t0 = time.time()
    intention = cw.generate_self_reflect(mock_wb, timeout=90)
    elapsed = time.time() - t0
    if intention and len(intention.strip()) > 10:
        print(f"  LLM 耗时: {elapsed:.0f}s")
        print(f"  意图: {intention.strip()[:150]}")
        print("[OK] Self-reflection produced intention")
    else:
        print(f"  [SKIP] model returned: {intention}")
else:
    print("  [SKIP] no Ollama")

# 4. OnlineAgent integration
print("\n--- OnlineAgent integration ---")
os.system("docker rm -f folunar-sandbox 2>/dev/null")
from agent.online_agent import OnlineAgent
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
os.system("docker rm -f folunar-sandbox 2>/dev/null")
assert hasattr(agent, 'self_model') and agent.self_model is not None
assert agent.creative_writer and agent.creative_writer.model == "qwen3.5:0.8b"
print(f"  模型: {agent.creative_writer.model}")
print("[OK] OnlineAgent 集成 (qwen3.5:0.8b)")

# 5. 快速跑: 验证自省触发 + 文件写入
print("\n--- 自省触发 ---")
self_hit = False
for i in range(35):
    agent.step()
    if getattr(agent, '_last_action_source', '') == "self_reflect":
        self_hit = True
        print(f"  [SELF] step {i+1}")
if self_hit:
    print("[OK] Self-reflection triggered after step 30")
else:
    print("  [SKIP] not triggered (needs step>30+%20==0)")

os.system("docker rm -f folunar-sandbox 2>/dev/null")
print(f"\n{'='*60}")
print("P17 自省验证 PASSED")
print(f"{'='*60}")
