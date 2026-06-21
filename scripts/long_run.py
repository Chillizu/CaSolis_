"""
长程闭环运行 + 详细日志

Usage:
    PYTHONPATH=. python3 scripts/long_run.py [--steps 300] [--gate 0.7]
"""

import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.online_agent import OnlineAgent


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--gate", type=float, default=0.7)
    parser.add_argument("--name", type=str, default="long_run")
    args = parser.parse_args()

    log_dir = f"checkpoints/{args.name}"
    os.makedirs(log_dir, exist_ok=True)

    log(f"启动长程闭环: {args.steps}步, gate={args.gate}")
    log(f"日志目录: {log_dir}")

    agent = OnlineAgent(conductor_gate=args.gate)

    # 每 50 步保存一次中间状态
    save_interval = 50
    snapshots = []

    start_time = time.time()

    for i in range(args.steps):
        success, reward = agent.step()

        # 在线训练
        if i > 0 and i % agent.train_interval == 0:
            loss = agent.train_step()
            cond_loss = agent._train_conductor_online()
            if (i + 1) % 50 == 0:
                rnd_s = agent.rnd.get_novelty_stats()
                cond_info = f"  conductor_loss={cond_loss:.4f}" if cond_loss > 0 else ""
                log(f"[{i+1:4d}] loss={loss:.4f}  novelty={rnd_s['running_errors_avg']:.4f}{cond_info}")

        # 每 50 步记录快照
        if (i + 1) % save_interval == 0:
            snapshot = {
                "step": i + 1,
                "success_rate": agent.success_count / max(agent.step_count, 1),
                "total_reward": agent.total_reward,
                "avg_reward": agent.total_reward / max(agent.step_count, 1),
                "buffer_size": agent.buffer.size,
                "ab_stats": dict(agent.ab_stats),
                "multi_cmds": agent.multi_cmds_count,
                "rnd_novelty": agent.rnd.get_novelty_stats().get("running_errors_avg", 0),
                "elapsed_s": time.time() - start_time,
            }

            # 意图分布
            intent_dist = {}
            for h in agent.intent_history:
                intent_dist[h] = intent_dist.get(h, 0) + 1
            snapshot["intent_distribution"] = {
                k: v / max(len(agent.intent_history), 1)
                for k, v in intent_dist.items()
            }

            # A/B 自适应率
            cond_rate = agent.ab_stats["conductor_success"] / max(agent.ab_stats["conductor"], 1)
            clf_rate = agent.ab_stats["classifier_success"] / max(agent.ab_stats["classifier"], 1)
            snapshot["p_conductor"] = 0.2 + 0.6 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))
            snapshot["conductor_success_rate"] = cond_rate
            snapshot["classifier_success_rate"] = clf_rate

            snapshots.append(snapshot)

            # 保存中间 checkpoint
            agent.save(f"{log_dir}/step_{i+1}")

            # 打印摘要
            a = agent.ab_stats
            log(f"[{i+1:4d}] 成功={agent.success_count}/{agent.step_count} "
                f"({snapshot['success_rate']:.0%})  "
                f"奖励={snapshot['avg_reward']:.2f}  "
                f"Conductor={a['conductor']}次({cond_rate:.0%})  "
                f"p_cond={snapshot['p_conductor']:.0%}  "
                f"多命令={snapshot['multi_cmds']}次  "
                f"意图={len(intent_dist)}种  "
                f"耗时={snapshot['elapsed_s']:.0f}s")

    # 最终统计
    total_time = time.time() - start_time
    result = agent.summarize()

    # 保存最终结果
    final_report = {
        "steps": args.steps,
        "gate": args.gate,
        "total_time_s": total_time,
        "steps_per_second": args.steps / total_time,
        "final": {
            "success_rate": result["success_rate"],
            "success_count": result["success"],
            "avg_reward": result["avg_reward"],
            "total_reward": result["total_reward"],
            "buffer_size": result["buffer_size"],
            "intent_distribution": result["intent_distribution"],
            "rnd_novelty_avg": result.get("rnd_stats", {}).get("running_errors_avg", 0),
        },
        "ab_stats": agent.ab_stats,
        "multi_cmds": agent.multi_cmds_count,
        "snapshots": snapshots,
    }

    # A/B 自适应率
    cond_rate = agent.ab_stats["conductor_success"] / max(agent.ab_stats["conductor"], 1)
    clf_rate = agent.ab_stats["classifier_success"] / max(agent.ab_stats["classifier"], 1)
    final_report["p_conductor_final"] = 0.2 + 0.6 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))

    report_path = f"{log_dir}/report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    log(f"报告已保存: {report_path}")
    log(f"总耗时: {total_time:.0f}s ({args.steps/total_time:.1f} 步/秒)")

    # 保存最终 Conductor checkpoint
    if agent.conductor_path_active:
        ckpt_path = f"{log_dir}/conductor_aligned.pt"
        agent.nanny.conductor.save(ckpt_path)
        log(f"Conductor 对齐后权重: {ckpt_path}")

    agent.save(f"{log_dir}/final")
    log("完成!")


if __name__ == "__main__":
    main()
