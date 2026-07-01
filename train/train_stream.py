"""
Stream 训练器 — 持续好奇心训练循环

模型永远在运行，永远在学习。
每个时间步既是推理也是训练。
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arch.core import StreamCore, CuriosityLoss
from arch.environments import NumberSequenceEnv


class StreamTrainer:
    """
    持续训练器

    核心循环：
    1. 模型观察环境状态
    2. 模型生成预测 + 动作
    3. 动作影响环境
    4. 模型观察结果
    5. 计算预测误差（好奇心）
    6. 更新模型参数
    7. 重复
    """

    def __init__(
        self,
        model: StreamCore,
        env: NumberSequenceEnv,
        lr: float = 1e-3,
        hidden_dim: int = 256,
    ):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-5
        )
        self.loss_fn = CuriosityLoss(
            curiosity_alpha=0.1,
            entropy_beta=0.05,
            thought_gamma=0.01,
        )
        self.hidden_dim = hidden_dim

        # 统计
        self.step_count = 0
        self.loss_history = []
        self.curiosity_history = []
        self.world_loss_history = []
        self.explore_rate_history = []
        self.mode_switches = 0
        self.last_switch_step = 0

        # 自动调节
        self.curiosity_alpha = 0.1
        self.entropy_beta = 0.05

    def train_step(
        self,
        h: torch.Tensor,
        obs: int,
        prev_action: int | None,
    ) -> tuple[torch.Tensor, int, dict]:
        """
        单步训练

        返回:
            h_new: 新隐藏状态
            action: 选择的动作
            metrics: 训练指标
        """
        batch_size = h.size(0)

        # 观察值 token
        obs_tensor = torch.tensor([obs], dtype=torch.long)

        # 前一动作
        if prev_action is not None:
            act_tensor = torch.tensor([prev_action], dtype=torch.long)
        else:
            act_tensor = None

        # 模型推理
        h_new, outputs = self.model.step(
            h, obs_tensor, prev_action=act_tensor, return_thoughts=True
        )

        # 截断计算图：阻止跨步反向传播
        # 每个时间步独立训练，不跨越时间步回传梯度
        h_new = h_new.detach()

        # 选择动作（采样）
        action_probs = F.softmax(outputs["action_logits"], dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()  # (batch,)

        # 环境步进
        next_obs, reward, done, info = self.env.step(action.item())

        # 计算损失
        target_obs = torch.tensor([next_obs], dtype=torch.long)
        losses = self.loss_fn(
            outputs,
            target_obs=target_obs,
            target_action=action,
        )

        # 反向传播
        self.optimizer.zero_grad()
        losses["total"].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # 更新统计
        self.step_count += 1
        self.loss_history.append(losses["total"].item())
        self.world_loss_history.append(losses["world"].item())
        self.curiosity_history.append(outputs["curiosity"].item())
        self.explore_rate_history.append(1.0 if info.get("is_explore", False) else 0.0)

        # 检测模式切换
        if info.get("mode_change", False):
            self.mode_switches += 1
            self.last_switch_step = self.step_count

        # 自动调节好奇心强度
        if self.step_count % 100 == 0:
            recent_world_loss = sum(self.world_loss_history[-50:]) / 50
            recent_explore = sum(self.explore_rate_history[-50:]) / 50

            # 如果预测误差很低，增加好奇心
            if recent_world_loss < 0.5:
                self.curiosity_alpha = min(0.3, self.curiosity_alpha + 0.01)
            # 如果探索率太低，增加熵权重
            if recent_explore < 0.2:
                self.entropy_beta = min(0.15, self.entropy_beta + 0.005)
            # 如果探索率太高，降低熵权重
            elif recent_explore > 0.6:
                self.entropy_beta = max(0.02, self.entropy_beta - 0.005)

        metrics = {
            "loss": losses["total"].item(),
            "world_loss": losses["world"].item(),
            "curiosity": outputs["curiosity"].item(),
            "curiosity_loss": losses["curiosity_loss"].item(),
            "entropy": losses["entropy"].item(),
            "action": action.item(),
            "is_explore": 1.0 if info.get("is_explore", False) else 0.0,
            "lr": self.scheduler.get_last_lr()[0],
            "curiosity_alpha": self.curiosity_alpha,
            "entropy_beta": self.entropy_beta,
        }

        return h_new, action.item(), metrics, next_obs


def train_loop(
    model: StreamCore,
    env: NumberSequenceEnv,
    steps: int = 5000,
    log_every: int = 100,
    save_every: int = 1000,
    output_dir: str = "checkpoints/stream-v1",
):
    """完整训练循环"""
    os.makedirs(output_dir, exist_ok=True)

    trainer = StreamTrainer(model, env, lr=1e-3)

    # 重置环境
    obs, info = env.reset()
    h = model.init_state(batch_size=1)
    prev_action = None

    print(f"{'步数':>6} | {'损失':>8} | {'世界':>8} | {'好奇':>6} | {'探索':>6} | {'模式':>8} | {'学习率':>8}")
    print("-" * 70)

    start_time = time.time()

    for step in range(steps):
        h, action, metrics, next_obs = trainer.train_step(h, obs, prev_action)

        obs = next_obs
        prev_action = action

        if (step + 1) % log_every == 0:
            avg_world = sum(trainer.world_loss_history[-log_every:]) / log_every
            avg_curiosity = sum(trainer.curiosity_history[-log_every:]) / log_every
            explore_rate = sum(trainer.explore_rate_history[-log_every:]) / log_every
            mode_str = env.mode[:8]

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            print(
                f"{step+1:>6,} | "
                f"{metrics['world_loss']:>8.4f} | "
                f"{avg_world:>8.4f} | "
                f"{avg_curiosity:>6.3f} | "
                f"{explore_rate:>6.1%} | "
                f"{mode_str:>8} | "
                f"{metrics['lr']:>8.2e}"
            )

        if (step + 1) % save_every == 0:
            checkpoint = {
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "loss_history": trainer.loss_history,
                "curiosity_history": trainer.curiosity_history,
                "world_loss_history": trainer.world_loss_history,
                "explore_rate_history": trainer.explore_rate_history,
            }
            torch.save(checkpoint, f"{output_dir}/checkpoint-{step+1}.pt")
            print(f"  💾 保存检查点: step-{step+1}")

    # 最终保存
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")
    print(f"\n✅ 训练完成！共 {steps} 步")
    print(f"   平均世界损失: {sum(trainer.world_loss_history[-500:])/500:.4f}")
    print(f"   平均好奇心: {sum(trainer.curiosity_history[-500:])/500:.3f}")
    print(f"   平均探索率: {sum(trainer.explore_rate_history[-500:])/500:.1%}")
    print(f"   模式切换数: {trainer.mode_switches}")

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    print(f"Stream 架构 — 从零训练")
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  步数: {args.steps} | 隐藏维度: {args.hidden}")
    print()

    trainer = train_loop(model, env, steps=args.steps, output_dir=args.output)
