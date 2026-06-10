#!/usr/bin/env python3
"""
原生交互 — 用 Docker 沙箱真实数据训练

生成"思考→行动→观察"序列，用真实命令输出训练。
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from train.train_native import NativeTrainer
from arch.native import (
    NativeCore, ActionMap, text_to_tokens, tokens_to_text,
    ACTION_START, BOS_TOKEN, N_CHAR, N_ACTION,
)
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


def generate_training_data(
    sandbox, actions: ActionMap, n_interactions: int = 8,
) -> list[int]:
    """用 Docker 沙箱生成真实的思考→行动→观察序列"""
    tokens = [BOS_TOKEN]

    # 一些思考模板（让模型看到多种文字模式）
    thoughts = [
        "what do we have here? ",
        "let me check the system. ",
        "interesting. let me explore. ",
        "i wonder what this does. ",
        "looking around... ",
        "time to investigate. ",
        "what else can i find? ",
        "let me see... ",
        "hmm, what's in here? ",
        "checking something. ",
        "i want to know more. ",
        "let me try this. ",
        "exploring the environment. ",
        "what's happening? ",
        "let me look at this. ",
    ]

    for i in range(n_interactions):
        # 随机选择一个思考模板
        thought = thoughts[i % len(thoughts)]
        tokens.extend(text_to_tokens(thought))

        # 随机选择一个行动
        action_idx = i % N_ACTION
        cmd = actions.get_cmd(action_idx)
        tokens.append(ACTION_START + action_idx)

        # 执行命令，获取真实输出
        try:
            r = sandbox.execute("bash", cmd, None)
            output = (r.stdout or "")[:300]
        except Exception as e:
            output = f"error: {e}"

        tokens.extend(text_to_tokens(output))

        # 加换行和分隔
        tokens.extend(text_to_tokens("\n"))

    return tokens


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="checkpoints/native-v2")
    parser.add_argument("--image", type=str, default="ubuntu:22.04")
    args = parser.parse_args()

    model = NativeCore(hidden_dim=args.hidden)
    actions = ActionMap()

    print(f"🌐 原生交互 — Docker 真实训练")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {N_CHAR} 字符 + {N_ACTION} 行动")
    print(f"   步数: {args.steps} | 隐藏: {args.hidden}")
    print()

    # ── Docker 沙箱 ──────────────────────────────────────
    print("启动 Docker 沙箱...")
    config = DockerSandboxConfig(
        image=args.image,
        network="none",
        memory_limit="256m",
        cpu_limit=2,
        timeout_per_action=10,
    )
    sandbox = DockerSandbox(config)
    sandbox.start()
    print(f"  容器就绪: {sandbox.container.id[:12]}")
    print()

    # ── 生成训练数据 ─────────────────────────────────────
    print("生成训练数据（逐轮实时）...")
    trainer = NativeTrainer(model, actions, lr=args.lr)

    h = model.init_state()
    all_tokens = []

    # 先用 3 轮生成初始数据
    initial_tokens = generate_training_data(sandbox, actions, n_interactions=6)
    all_tokens.extend(initial_tokens)
    print(f"  初始数据: {len(initial_tokens)} tokens")

    # ── 训练循环 ─────────────────────────────────────────
    print(f"\n{'步数':>6} | {'字符损失':>8} | {'好奇':>6} | {'行动率':>6} | {'LR':>8}")
    print("-" * 45)

    start_time = time.time()
    total_steps = 0
    data_pos = 0

    while total_steps < args.steps:
        # 走完当前数据
        for i in range(data_pos, len(all_tokens) - 1):
            if total_steps >= args.steps:
                break

            token = all_tokens[i]
            next_tok = all_tokens[i + 1]
            h, m = trainer.train_step(h, token, next_tok)
            total_steps += 1

            if total_steps % 200 == 0:
                avg_c = sum(trainer.char_losses[-200:]) / 200
                avg_q = sum(trainer.curiosities[-200:]) / 200
                act_rate = sum(trainer.action_taken[-200:]) / 200
                print(
                    f"{total_steps:>6,} | "
                    f"{avg_c:>8.4f} | "
                    f"{avg_q:>6.3f} | "
                    f"{act_rate:>6.1%} | "
                    f"{m['lr']:>8.2e}"
                )

        # 生成新一段数据（用当前模型采样 + Docker 新输出）
        if total_steps < args.steps:
            new_data = generate_training_data(sandbox, actions, n_interactions=3)
            old_len = len(all_tokens)
            all_tokens.extend(new_data)
            data_pos = old_len
            print(f"  ➕ 追加 {len(new_data)} tokens (当前位置: {data_pos})")

    sandbox.stop()

    # ── 最终 ─────────────────────────────────────────────
    torch.save(model.state_dict(), f"{args.output}/model_final.pt")

    final_char = sum(trainer.char_losses[-500:]) / 500
    final_curio = sum(trainer.curiosities[-500:]) / 500
    elapsed = time.time() - start_time

    print(f"\n✅ 训练完成！{total_steps} 步 ({elapsed:.0f}s)")
    print(f"   最终字符损失: {final_char:.4f}（随机 ≈ 4.56）")
    print(f"   最终好奇心: {final_curio:.3f}")
    print(f"   总数据: {len(all_tokens)} tokens")

    # ── 自主生成测试 ─────────────────────────────────────
    print("\n🧪 自主生成测试:")
    h = model.init_state()
    h, seq = trainer.generate_interaction(h, n_steps=50)
    text = tokens_to_text(seq)
    actions_used = [
        actions.get_cmd(t - ACTION_START)
        for t in seq if ACTION_START <= t < ACTION_START + N_ACTION
    ]
    print(f"  思考: {text[:200]}")
    print(f"  行动: {', '.join(actions_used)}")

    # 存一个可视化结果
    with open(f"{args.output}/generated.txt", "w") as f:
        f.write(f"思考: {text}\n行动: {', '.join(actions_used)}\n")


if __name__ == "__main__":
    import argparse
    run()
