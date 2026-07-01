"""
Dual Channel v2 训练器
观察 → 思考(RNN更新) → 决定 → 行动 → 学习
"""

import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from arch.dual_channel import DualChannelCore, DualChannelEnv


class DualTrainer:
    def __init__(self, model: DualChannelCore, env: DualChannelEnv, lr: float = 5e-4):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20000, eta_min=1e-5
        )

        self.step_count = 0
        self.world_losses = []
        self.curiosities = []
        self.action_history = []

    def train_cycle(self, h: torch.Tensor, obs_token: int) -> tuple[torch.Tensor, dict, int]:
        """
        一个周期：
        1. 观察 → 思考（RNN 隐藏状态更新）
        2. 从隐藏状态决定行动
        3. 世界模型预测结果
        4. 环境执行行动
        5. 计算损失 → 学习
        """
        obs_tensor = torch.tensor([obs_token], dtype=torch.long)

        # ── 1+2: 观察并思考 ─────────────────────────────
        h_thought = self.model.observe_and_think(h, obs_tensor)

        # ── 3: 决定行动 ──────────────────────────────────
        action_logits, world_logits, curiosity = self.model.decide(h_thought)

        # 采样行动（温度退火：早期高探索，后期低）
        temp = max(0.6, 1.2 - self.step_count / 5000)
        action_probs = F.softmax(action_logits / temp, dim=-1)
        action = torch.distributions.Categorical(action_probs).sample()

        # ── 4: 环境执行 ──────────────────────────────────
        result_token, cmd = self.env.act(action.item())
        result_idx = result_token - 112

        # ── 5: 损失 ──────────────────────────────────────
        # 世界模型损失
        target = torch.tensor([result_idx], dtype=torch.long)
        world_loss = F.cross_entropy(world_logits, target, reduction="mean")

        # 好奇心门损失（训练好奇心门预测预测误差）
        with torch.no_grad():
            surprise_target = torch.sigmoid(world_loss.detach() - 2.0)
        curiosity_loss = F.mse_loss(
            curiosity.squeeze(-1),
            surprise_target.expand(1),
        )

        # 行动熵（维持多样性）
        action_entropy = -(
            action_probs * torch.log(action_probs + 1e-10)
        ).sum(dim=-1).mean()

        # 行动多样性奖励
        recent = self.action_history[-30:]
        if len(recent) >= 30:
            unique_ratio = len(set(recent)) / 16.0
            diversity_bonus = 0.05 * (1.0 - unique_ratio)
        else:
            diversity_bonus = 0.0

        # 总损失
        total = world_loss + 0.1 * curiosity_loss - 0.05 * action_entropy + diversity_bonus

        # 反向传播
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        # 截断，防止跨步梯度
        h_new = h_thought.detach()

        # 统计
        self.step_count += 1
        self.world_losses.append(world_loss.item())
        self.curiosities.append(curiosity.item())
        self.action_history.append(action.item())

        metrics = {
            "world": world_loss.item(),
            "curiosity": curiosity.item(),
            "entropy": action_entropy.item(),
            "action": action.item(),
            "cmd": cmd,
            "temp": temp,
            "lr": self.scheduler.get_last_lr()[0],
        }

        return h_new, metrics, result_token


def train_loop(
    model: DualChannelCore,
    env: DualChannelEnv,
    cycles: int = 3000,
    log_every: int = 50,
    output_dir: str = "checkpoints/dual-v2",
):
    os.makedirs(output_dir, exist_ok=True)
    trainer = DualTrainer(model, env, lr=5e-4)

    h = model.init_state(batch_size=1)
    obs_token = 112  # 起始 BOS

    print(f"{'轮':>4} | {'世界':>7} | {'好奇':>5} | {'熵':>5} | {'多样':>4} | {'温度':>5} | {'LR':>8}")
    print("-" * 52)

    for cycle in range(cycles):
        h, m, obs_token = trainer.train_cycle(h, obs_token)

        if (cycle + 1) % log_every == 0:
            avg_w = sum(trainer.world_losses[-log_every:]) / log_every
            avg_c = sum(trainer.curiosities[-log_every:]) / log_every
            acts = trainer.action_history[-log_every:]
            unique = len(set(acts))

            print(
                f"{cycle+1:>4,} | "
                f"{avg_w:>7.4f} | "
                f"{avg_c:>5.3f} | "
                f"{m['entropy']:>5.3f} | "
                f"{unique:>3}/{log_every:>3} | "
                f"{m['temp']:>5.2f} | "
                f"{m['lr']:>8.2e}"
            )

    # 最终模型
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ {cycles} 轮完成")
    print(f"   最终世界损失: {sum(trainer.world_losses[-200:])/200:.4f}")
    print(f"   最终好奇心: {sum(trainer.curiosities[-200:])/200:.3f}")
    print(f"   最后 200 步多样性: {len(set(trainer.action_history[-200:]))}/16")
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--output", type=str, default="checkpoints/dual-v2")
    args = parser.parse_args()

    model = DualChannelCore(hidden_dim=args.hidden)
    env = DualChannelEnv()
    print(f"🧠 双通道 — 观察→思考→行动→学习")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   隐藏: {args.hidden} | 轮: {args.cycles}")
    print()

    trainer = train_loop(model, env, cycles=args.cycles, output_dir=args.output)

    # 测试：看模型最后学会了什么
    print("\n🧪 最终测试:")
    h = model.init_state()
    obs = 112
    for i in range(20):
        with torch.no_grad():
            h = model.observe_and_think(h, torch.tensor([obs], dtype=torch.long))
            action_logits, world_logits, curiosity = model.decide(h)
            action = action_logits.argmax(dim=-1).item()
            world_pred = world_logits.argmax(dim=-1).item()
            cmd = env.commands[action]
            obs, _ = env.act(action)
        print(f"  {'→' if i < 19 else '√'} {cmd:20s}  好奇:{curiosity.item():.3f}")
    print(f"  多样性: {len(set(env.commands[a] for a in env.cmd_history[-20:]))}/16")
