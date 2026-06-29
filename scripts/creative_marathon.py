"""
创造力马拉松 — 2小时
评估 3-intent 系统的创造产出
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'
from agent.online_agent import OnlineAgent

RUN_HOURS = 2
t_start = time.time()
step_count = 0
agent = OnlineAgent(conductor_gate=0.6)

print(f"\n3-INTENT 创造力马拉松 — {RUN_HOURS}小时")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    while time.time() - t_start < RUN_HOURS * 3600:
        success, reward = agent.step()
        step_count += 1

        if step_count % 500 == 0:
            elapsed = time.time() - t_start
            from collections import Counter
            c = Counter(agent.intent_history[-500:])
            gs = agent.workbench.graph.stats()
            km = agent.knowledge_mapper
            ks = km.get_exploration_stats() if km else {}
            create_count = c.get("CREATE", 0)
            print(f"[{time.strftime('%H:%M:%S')}] {step_count}步 "
                  f"({elapsed/3600:.1f}h) "
                  f"CREATE={create_count} "
                  f"FG={gs['n_nodes']} "
                  f"自发现={ks.get('explored',0)}/{ks.get('total_available',0)} "
                  f"成功={agent.success_count}/{step_count}")

except KeyboardInterrupt:
    print("\n中断")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback; traceback.print_exc()
finally:
    elapsed = time.time() - t_start
    from collections import Counter
    c = Counter(agent.intent_history)
    print(f"\n{'='*60}")
    print(f"创造力马拉松结束 — {elapsed/3600:.1f}h, {step_count}步")
    print(f"成功率: {agent.success_count}/{step_count} ({agent.success_count/step_count*100:.1f}%)")
    print(f"意图分布:")
    for i, cnt in c.most_common():
        print(f"  {i}: {cnt} ({cnt/step_count*100:.1f}%)")
    gs = agent.workbench.graph.stats()
    print(f"FactGraph: {gs['n_nodes']}节点")
    km = agent.knowledge_mapper
    if km:
        ks = km.get_exploration_stats()
        print(f"自发现: {ks.get('explored',0)}/{ks.get('total_available',0)}")
        cm = km.get_intent_command_map() if hasattr(km, 'get_intent_command_map') else {}
        print(f"意图→命令映射: {len(cm)}个意图")
        for intent, cmds in sorted(cm.items()):
            print(f"  {intent}: {cmds[:6]}")
    recs = getattr(agent, '_recommended_cmds', set())
    print(f"推荐命令: {len(recs)}个: {sorted(recs)[:10]}")
    stats = {
        "run_id": f"creative_{int(t_start)}",
        "n_steps": step_count,
        "success_rate": agent.success_count / max(step_count, 1),
        "total_reward": agent.total_reward,
    }
    agent.pstore.save_all(agent, stats)
    agent.pstore.close()
    print("已保存")
