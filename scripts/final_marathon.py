"""
终极马拉松 — 1 小时
评估完整 3-intent + 推理 + 自生成 + 工具 系统
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HUB_OFFLINE'] = '1'
from agent.online_agent import OnlineAgent

RUN_MINUTES = 60
t_start = time.time()
step_count = 0
agent = OnlineAgent(conductor_gate=0.6)

print(f"\n终极马拉松 — {RUN_MINUTES}分钟")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    while time.time() - t_start < RUN_MINUTES * 60:
        success, reward = agent.step()
        step_count += 1
        if step_count % 1000 == 0:
            elapsed = time.time() - t_start
            from collections import Counter
            c = Counter(agent.intent_history[-1000:])
            g = agent.workbench.graph
            km = agent.knowledge_mapper
            ks = km.get_exploration_stats() if km else {}
            infs = len([1 for n in g.nodes.values() if n.category == 'inference'])
            exps = sum(1 for h in agent.goal_generator.goal_history if 'experiment' in str(h.get('source','')))
            print(f"[{time.strftime('%H:%M:%S')}] {step_count}步 "
                  f"({elapsed/60:.0f}分) "
                  f"O={c.get('OBSERVE',0)} C={c.get('CREATE',0)} T={c.get('TRY',0)} "
                  f"FG={g.node_count()} 推理={infs} 实验={exps} "
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
    g = agent.workbench.graph
    km = agent.knowledge_mapper
    gg = agent.goal_generator
    print(f"\n{'='*60}")
    print(f"终极马拉松结束 — {elapsed/60:.0f}分, {step_count}步, {agent.success_count}/{step_count}成功")
    for i, cnt in c.most_common():
        print(f"  {i}: {cnt} ({cnt/step_count*100:.1f}%)")
    print(f"FactGraph: {g.node_count()}节点")
    print(f"推理: {len([1 for n in g.nodes.values() if n.category=='inference'])}")
    if km:
        ks = km.get_exploration_stats()
        print(f"自发现: {ks.get('explored',0)}/{ks.get('total_available',0)}")
    exps = sum(1 for h in gg.goal_history if 'experiment' in str(h.get('source','')))
    print(f"自生成实验: {exps}")
    inf_aware = len(getattr(gg, '_inferences', {}))
    print(f"推理感知: {inf_aware}条")
    stats = {"run_id": f"final_{int(t_start)}", "n_steps": step_count,
             "success_rate": agent.success_count/max(step_count,1)}
    agent.pstore.save_all(agent, stats)
    agent.pstore.close()
    print("已保存")
