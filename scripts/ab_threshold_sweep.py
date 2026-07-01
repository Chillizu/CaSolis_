"""
A/B 阈值网格扫描 (P0)

对 conductor_gate 做 0.8 → 0.7 → 0.6 三档实验,
每档跑 100 步闭环, 收集统计数据对比。

Usage:
    PYTHONPATH=. python3 scripts/ab_threshold_sweep.py
"""

import sys, os, json, time, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.online_agent import OnlineAgent


THRESHOLDS = [0.8, 0.7, 0.6]
N_STEPS = 100


def run_threshold(gate: float, run_id: int) -> dict:
    """在给定阈值下跑 N_STEPS 步, 返回统计结果"""
    print(f"\n{'='*60}")
    print(f"  实验: gate={gate}  (第 {run_id+1}/{len(THRESHOLDS)} 轮)")
    print(f"{'='*60}")

    agent = OnlineAgent(
        conductor_gate=gate,
        train_interval=20,      # 保持在线训练
        novelty_weight=0.3,
        explore_prob=0.1,
    )

    # 清理历史 (确保每轮独立)
    agent.intent_history = []
    agent.step_count = 0
    agent.total_reward = 0.0
    agent.success_count = 0
    agent.ab_stats = {"conductor": 0, "classifier": 0, "conductor_success": 0, "classifier_success": 0}

    # 每 10 步打印一次进度
    result = agent.run(n_steps=N_STEPS, verbose=True)

    # 提取关键指标
    a = agent.ab_stats
    cond_usage = a["conductor"] / max(N_STEPS, 1)
    cond_success_rate = a["conductor_success"] / max(a["conductor"], 1)
    clf_success_rate = a["classifier_success"] / max(a["classifier"], 1)

    # HELP 占比 (前50 vs 后50)
    first_half = agent.intent_history[:N_STEPS//2]
    second_half = agent.intent_history[N_STEPS//2:]
    help_first = first_half.count("HELP") / max(len(first_half), 1)
    help_second = second_half.count("HELP") / max(len(second_half), 1)

    # 意图多样化
    unique_intents = len(set(agent.intent_history))
    intent_dist = {}
    for intent in agent.intent_history:
        intent_dist[intent] = intent_dist.get(intent, 0) + 1

    summary = {
        "gate": gate,
        "steps": agent.step_count,
        "success_rate": result["success_rate"],
        "total_reward": result["total_reward"],
        "avg_reward": result["avg_reward"],
        "conductor_usage": cond_usage,
        "conductor_usage_count": a["conductor"],
        "conductor_success_rate": cond_success_rate,
        "classifier_success_rate": clf_success_rate,
        "help_first_half": help_first,
        "help_second_half": help_second,
        "unique_intents": unique_intents,
        "intent_distribution": intent_dist,
        "ab_stats": copy.deepcopy(a),
        "rnd_novelty_avg": result.get("rnd_stats", {}).get("running_errors_avg", 0),
    }

    # 关闭 Docker 容器
    if agent.sandbox:
        try:
            agent.sandbox.close()
        except:
            pass

    return summary


def print_report(results: list[dict]):
    """打印最终对比报告"""
    print(f"\n\n{'='*70}")
    print(f"  📊 P0 阈值扫描 — 最终报告")
    print(f"{'='*70}")
    print()

    # 表头
    header = f"{'阈值':>6} | {'成功率':>8} | {'平均奖励':>8} | {'指挥家使用率':>12} | {'指挥家成功率':>12} | {'HELP(前50)':>10} | {'HELP(后50)':>10} | {'意图种类':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['gate']:>6.1f} | "
            f"{r['success_rate']:>7.1%} | "
            f"{r['avg_reward']:>8.2f} | "
            f"{r['conductor_usage']:>10.1%}   | "
            f"{r['conductor_success_rate']:>10.1%}   | "
            f"{r['help_first_half']:>9.1%} | "
            f"{r['help_second_half']:>9.1%} | "
            f"{r['unique_intents']:>8d}"
        )

    print()
    print(f"{'='*70}")
    print()

    # 详细意图分布
    for r in results:
        print(f"\n── gate={r['gate']:.1f} 意图分布:")
        sorted_intents = sorted(r['intent_distribution'].items(), key=lambda x: -x[1])
        for intent, count in sorted_intents:
            pct = count / r['steps'] * 100
            bar = "█" * int(pct / 2)
            print(f"  {intent:15s} {count:4d} ({pct:5.1f}%) {bar}")
        print(f"  指挥家: {r['conductor_usage_count']}次  ({r['conductor_success_rate']:.0%} 成功)")
        print(f"  分类器: {r['steps'] - r['conductor_usage_count']}次  ({r['classifier_success_rate']:.0%} 成功)")

    # 推荐
    print(f"\n{'='*70}")
    print(f"  💡 推荐")
    print(f"{'='*70}")

    # 选择最优: 指挥家使用率最高, 且总成功率下降不超过 10%
    base_sr = results[0]["success_rate"]  # 0.8 为基线
    best = results[0]
    for r in results[1:]:
        sr_drop = base_sr - r["success_rate"]
        if sr_drop < 0.10 and r["conductor_usage"] > best["conductor_usage"]:
            best = r

    print(f"  基线 (gate=0.8): 指挥家使用率 {results[0]['conductor_usage']:.1%}, "
          f"总成功率 {results[0]['success_rate']:.1%}")
    print(f"  推荐阈值: gate={best['gate']:.1f}")
    print(f"    → 指挥家使用率: {best['conductor_usage']:.1%}")
    print(f"    → 总成功率: {best['success_rate']:.1%} "
          f"({'↑' if best['success_rate'] >= base_sr else '↓'}{abs(best['success_rate']-base_sr)*100:.0f}%)")
    print(f"    → HELP 收敛: {best['help_first_half']:.0%} → {best['help_second_half']:.0%}")
    print()


if __name__ == "__main__":
    results = []

    for i, gate in enumerate(THRESHOLDS):
        try:
            summary = run_threshold(gate, i)
            results.append(summary)

            # 即时打印概要
            print(f"\n  ⏺ gate={gate:.1f} 完成: "
                  f"使用率={summary['conductor_usage']:.1%}, "
                  f"成功率={summary['success_rate']:.1%}, "
                  f"平均奖励={summary['avg_reward']:.2f}")
        except KeyboardInterrupt:
            print(f"\n  ⚠️ 被用户中断")
            break
        except Exception as e:
            print(f"\n  ❌ gate={gate:.1f} 失败: {e}")
            import traceback
            traceback.print_exc()
            # 继续下一个阈值
            continue

    if results:
        print_report(results)

        # 保存结果到 JSON
        report_path = "checkpoints/ab_sweep_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {report_path}")
    else:
        print("  ❌ 没有成功的实验")
