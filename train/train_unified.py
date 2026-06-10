"""
Unified 训练器 — Token 即操作，持续学习
"""

import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch.unified import UnifiedCore, UnifiedEnv


class UnifiedTrainer:
    def __init__(
        self,
        model: UnifiedCore,
        env: UnifiedEnv,
        lr: float = 3e-4,
    ):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20000, eta_min=1e-5
        )

        self.step_count = 0
        self.loss_history = []
        self.cmd_rate_history = []
        self.surprise_history = []
        self.token_history = []

    def train_step(self, h: torch.Tensor, obs: int) -> tuple[torch.Tensor, int, dict]:
        """单步训练：模型生成 token → 环境解释 → 学习"""
        batch_size = h.size(0)
        obs_tensor = torch.tensor([obs], dtype=torch.long)

        # 模型预测下一个 token
        h_new, outputs = self.model.step(h, obs_tensor)

        # 采样下一个 token
        logits = outputs["token_logits"]
        probs = F.softmax(logits / 0.8, dim=-1)  # 温度采样
        dist = torch.distributions.Categorical(probs)
        pred_token = dist.sample()

        # 环境步进
        target_token, info = self.env.step(pred_token.item())

        # 世界模型损失：预测下一个 token
        target = torch.tensor([target_token], dtype=torch.long)
        world_loss = F.cross_entropy(logits, target, reduction="mean")

        # 预测误差（惊喜度）
        pred_prob = F.softmax(logits, dim=-1)[0, target_token].item()
        surprise = -torch.log(torch.tensor(pred_prob + 1e-10)).item()

        # 反向传播
        self.optimizer.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        # 截断隐藏状态
        h_new = h_new.detach()

        # 统计
        self.step_count += 1
        self.loss_history.append(world_loss.item())
        self.surprise_history.append(surprise)
        is_cmd = 1.0 if info["type"] == "command" else 0.0
        self.cmd_rate_history.append(is_cmd)
        self.token_history.append(pred_token.item())

        metrics = {
            "loss": world_loss.item(),
            "surprise": surprise,
            "is_cmd": is_cmd,
            "token": pred_token.item(),
            "token_type": info["type"],
            "lr": self.scheduler.get_last_lr()[0],
        }

        return h_new, pred_token.item(), target_token, metrics


def train_loop(
    model: UnifiedCore,
    env: UnifiedEnv,
    steps: int = 10000,
    log_every: int = 100,
    save_every: int = 5000,
    output_dir: str = "checkpoints/unified-v1",
):
    os.makedirs(output_dir, exist_ok=True)
    trainer = UnifiedTrainer(model, env, lr=3e-4)

    obs = env.get_initial_obs()
    h = model.init_state(batch_size=1)

    print(f"{'步数':>6} | {'损失':>8} | {'惊喜':>6} | {'命令率':>6} | {'tok':>4} | {'学习率':>8}")
    print("-" * 60)

    start = time.time()

    for step in range(steps):
        h, pred_tok, target_tok, m = trainer.train_step(h, obs)
        obs = target_tok  # 环境输出成为下一个观察

        if (step + 1) % log_every == 0:
            recent_loss = sum(trainer.loss_history[-log_every:]) / log_every
            recent_surprise = sum(trainer.surprise_history[-log_every:]) / log_every
            cmd_rate = sum(trainer.cmd_rate_history[-log_every:]) / log_every
            elapsed = time.time() - start

            print(
                f"{step+1:>6,} | "
                f"{recent_loss:>8.4f} | "
                f"{recent_surprise:>6.3f} | "
                f"{cmd_rate:>6.1%} | "
                f"{pred_tok:>4} | "
                f"{m['lr']:>8.2e}"
            )

        if (step + 1) % save_every == 0:
            torch.save(model.state_dict(), f"{output_dir}/model-{step+1}.pt")

    # 保存最终模型
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ 训练完成！{steps} 步")
    print(f"   最终损失: {sum(trainer.loss_history[-500:])/500:.4f}")
    print(f"   平均命令率: {sum(trainer.cmd_rate_history[-500:])/500:.1%}")

    # 预测 vs 实际 token 序列（最后 50 步）
    recent = trainer.token_history[-50:]
    cmd_rate = sum(1 for t in recent if 96 <= t < 112) / 50
    print(f"   最近 50 步命令率: {cmd_rate:.1%}")

    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="checkpoints/unified-v1")
    args = parser.parse_args()

    model = UnifiedCore(vocab_size=128, embed_dim=64, hidden_dim=args.hidden)
    env = UnifiedEnv()

    print(f"🌊 Unified 架构 (Token 即操作)")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   步数: {args.steps} | 命令: 16 (token 96-111)")
    print()

    trainer = train_loop(model, env, steps=args.steps, output_dir=args.output)
