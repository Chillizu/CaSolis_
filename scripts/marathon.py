"""
马拉松运行脚本 — 9 小时无人值守
"""

import sys, os, warnings, time, json, signal
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'

from agent.online_agent import OnlineAgent

RUN_HOURS = 9
CHECKPOINT_INTERVAL = 500
LOG_INTERVAL = 1000

t_start = time.time()
step_count = 0

agent = OnlineAgent(conductor_gate=0.6)

print(f"\n{'='*60}")
print(f"马拉松开始 — {RUN_HOURS}小时")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

try:
    while time.time() - t_start < RUN_HOURS * 3600:
        success, reward = agent.step()
        step_count += 1

        # 每 1000 步报告
        if step_count % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            eta = max(0, RUN_HOURS * 3600 - elapsed)
            sps = step_count / (elapsed / 60)
            gs = agent.workbench.graph.stats()
            ks = agent.knowledge_mapper.get_exploration_stats()
            log = (
                f"\n[{time.strftime('%H:%M:%S')}] "
                f"{step_count}步 ({elapsed/3600:.1f}h) "
                f"{sps:.0f}步/分 "
                f"FG={gs['n_nodes']} "
                f"自发现={ks['explored']}/{ks['total_available']} "
                f"成功={agent.success_count}/{step_count} "
                f"奖励={agent.total_reward:.0f}"
            )
            print(log)
            sys.stdout.flush()

        # 每 500 步保存
        if step_count % CHECKPOINT_INTERVAL == 0:
            stats = {
                "run_id": f"marathon_{int(t_start)}",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                "n_steps": step_count,
                "success_rate": agent.success_count / max(step_count, 1),
                "total_reward": agent.total_reward,
                "fact_graph_nodes": agent.workbench.graph.stats()['n_nodes'],
                "n_intents_covered": len(set(agent.intent_history)),
            }
            agent.pstore.save_all(agent, stats)
            print(f"  [CKPT] 已保存 ({step_count}步)")
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\n[停止] 用户中断")

except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()

finally:
    elapsed = time.time() - t_start
    gs = agent.workbench.graph.stats()
    ks = agent.knowledge_mapper.get_exploration_stats()

    print(f"\n{'='*60}")
    print(f"马拉松结束")
    print(f"运行时间: {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print(f"总步数:   {step_count}")
    print(f"成功率:   {agent.success_count}/{step_count} ({agent.success_count/step_count*100:.1f}%)")
    print(f"总奖励:   {agent.total_reward:.0f}")
    print(f"FactGraph: {gs['n_nodes']}节点, {gs['n_edges']}边")
    print(f"自发现:   {ks['explored']}/{ks['total_available']} 命令")
    print(f"意图:     {len(set(agent.intent_history))}种")
    print(f"{'='*60}")

    # 最终保存
    stats = {
        "run_id": f"marathon_{int(t_start)}",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_steps": step_count,
        "success_rate": agent.success_count / max(step_count, 1),
        "total_reward": agent.total_reward,
        "fact_graph_nodes": gs['n_nodes'],
        "n_intents_covered": len(set(agent.intent_history)),
    }
    agent.pstore.save_all(agent, stats)
    agent.pstore.close()
    print("最终状态已保存")
