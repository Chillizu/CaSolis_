#!/usr/bin/env python3
"""
测试 Stream 架构 — 从零训练一个小型好奇心模型

运行方式:
    OMP_NUM_THREADS=4 python3 scripts/run_stream.py --steps 3000
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch.core import StreamCore, CuriosityLoss
from arch.environments import NumberSequenceEnv
from train.train_stream import StreamTrainer, train_loop


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="训练步数")
    parser.add_argument("--hidden", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--output", type=str, default="checkpoints/stream-v1")
    args = parser.parse_args()

    model = StreamCore(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=args.hidden,
        action_dim=16,
        thought_tokens=4,
    )
    env = NumberSequenceEnv(vocab_size=128)

    print(f"🌊 Stream 架构 — 从零训练")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   步数: {args.steps}")
    print(f"   CPU 线程: {os.environ.get('OMP_NUM_THREADS', 'auto')}")
    print()

    trainer = train_loop(
        model, env,
        steps=args.steps,
        log_every=50,
        save_every=2000,
        output_dir=args.output,
    )
