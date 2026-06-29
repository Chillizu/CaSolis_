"""快速验证: 自省方向链路 — 解析→偏置→执行"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_OFFLINE"] = "1"

from agent.online_agent import OnlineAgent
os.system("docker rm -f folunar-sandbox 2>/dev/null")
agent = OnlineAgent(buffer_size=100, train_interval=99, batch_size=16,
                    lr=1e-4, conductor_gate=0.7, mode="auto")
os.system("docker rm -f folunar-sandbox 2>/dev/null")

# 跑几步让 system 有基础事实
for i in range(15):
    agent.step()

# 模拟自省: 直接设置方向
test_intentions = [
    "I want to explore the filesystem and write a summary script",
    "I want to check network connections and see what's running",
    "I want to write a Python script that analyzes CPU usage",
]

for intention in test_intentions:
    print(f"\n[{'] 自省: '.join([intention[:40]])}]")
    agent._parse_and_apply_direction(intention)
    print(f"  方向={agent._self_direction}, 剩余={agent._self_remaining}步")
    assert agent._self_remaining == 5
    assert agent._self_direction

    # 模拟方向步
    for step_n in range(5):
        intent, params = agent._direction_step()
        assert intent, f"step {step_n} 无意图"
        print(f"  步{step_n+1}: {intent} {str(params)[:50]}")
    print(f"  完成后剩余={agent._self_remaining}步 (应为0)")
    assert agent._self_remaining == 0

# 验证 cmd_selector bias 被设置了 (最后一个 intention 的 bias)
if hasattr(agent, 'cmd_selector'):
    bias = agent.cmd_selector.cluster_bias
    print(f"\ncmd_selector cluster_bias: {bias}")
    assert bias, "cluster_bias should be set"

print(f"\nOK: 方向链路完整")
os.system("docker rm -f folunar-sandbox 2>/dev/null")
