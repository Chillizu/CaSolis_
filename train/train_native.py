"""
Native 训练器 — 原生交互

模型在真正的文字流中学习：
- 字符 token = 思考/阅读
- 行动 token = 操作环境
"""

import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from arch.native import (
    NativeCore, ActionMap, text_to_tokens, tokens_to_text,
    CHAR_START, ACTION_START, SPECIAL_START, BOS_TOKEN,
    N_CHAR, N_ACTION, N_SPECIAL, VOCAB_SIZE,
)


class NativeTrainer:
    def __init__(self, model: NativeCore, actions: ActionMap, lr: float = 3e-4):
        self.model = model
        self.actions = actions
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20000, eta_min=1e-5
        )

        self.step_count = 0
        self.char_losses = []
        self.action_losses = []
        self.world_losses = []
        self.curiosities = []
        self.action_taken = []

        # 用于采样行动的分布（温度退火）
        self.temp = 1.0

    def train_step(
        self, h: torch.Tensor, token: int, next_token: int
    ) -> tuple[torch.Tensor, dict]:
        """
        单 token 训练步

        参数:
            h: 当前隐藏状态
            token: 当前输入的 token
            next_token: 下一个实际 token（训练目标）

        返回:
            h_new: 新隐藏状态
            metrics: 指标
        """
        token_t = torch.tensor([token], dtype=torch.long)

        # 模型推理
        h_new, outputs = self.model.step(h, token_t)

        is_action = ACTION_START <= next_token < ACTION_START + N_ACTION

        # ── 1. 字符预测损失 ─────────────────────────────
        char_target = next_token if next_token < N_CHAR else 0
        char_loss = F.cross_entropy(
            outputs["char_logits"],
            torch.tensor([char_target], dtype=torch.long),
            reduction="mean",
        )

        # ── 2. 行动预测损失 ─────────────────────────────
        if is_action:
            action_idx = next_token - ACTION_START
            action_loss = F.cross_entropy(
                outputs["action_logits"],
                torch.tensor([action_idx], dtype=torch.long),
                reduction="mean",
            )
        else:
            action_loss = torch.tensor(0.0)

        # ── 3. 好奇心门 ─────────────────────────────────
        # 世界模型 + 好奇心在这里简化：好奇心门预测"这个 token 我猜得准吗"
        # 高 char_loss 意味着意外 → 好奇心门应该打开
        with torch.no_grad():
            curiosity_target = torch.sigmoid(char_loss.detach() - 3.0)
        curiosity_loss = F.mse_loss(
            outputs["curiosity"].squeeze(-1),
            curiosity_target.expand(1),
        )

        # ── 总损失 ─────────────────────────────────────
        total = char_loss + 0.2 * action_loss + 0.1 * curiosity_loss

        # 反向传播
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        h_new = h_new.detach()

        # 统计
        self.step_count += 1
        self.char_losses.append(char_loss.item())
        if is_action:
            self.action_losses.append(action_loss.item())
        self.curiosities.append(outputs["curiosity"].item())
        self.action_taken.append(1 if is_action else 0)

        metrics = {
            "char_loss": char_loss.item(),
            "action_loss": action_loss.item(),
            "curiosity": outputs["curiosity"].item(),
            "token": token,
            "next_token": next_token,
            "lr": self.scheduler.get_last_lr()[0],
        }

        return h_new, metrics

    def generate_interaction(self, h: torch.Tensor, n_steps: int = 30) -> tuple[torch.Tensor, list[int]]:
        """自主生成一段交互序列"""
        token = BOS_TOKEN
        seq = [BOS_TOKEN]

        with torch.no_grad():
            for _ in range(n_steps):
                token_t = torch.tensor([token], dtype=torch.long)
                h, outputs = self.model.step(h, token_t)
                h = h.detach()

                # ε-贪心：以概率 0.4 强行走一步行动
                if torch.rand(1).item() < 0.4:
                    # 均匀随机选一个行动
                    action = torch.randint(0, N_ACTION, (1,)).item()
                    token = ACTION_START + action
                else:
                    # 否则采样行动头
                    action_probs = F.softmax(
                        outputs["action_logits"].squeeze(0) / 0.8, dim=-1
                    )
                    action = torch.multinomial(action_probs, 1).item()
                    action_conf = action_probs[action].item()

                    if action_conf > 0.2:
                        token = ACTION_START + action
                    else:
                        # 或者输出字符
                        char_probs = F.softmax(
                            outputs["char_logits"].squeeze(0) / 1.0, dim=-1
                        )
                        token = torch.multinomial(char_probs, 1).item()

                seq.append(token)

        return h, seq


def train_loop(
    model: NativeCore,
    actions: ActionMap,
    steps: int = 10000,
    output_dir: str = "checkpoints/native-v1",
):
    os.makedirs(output_dir, exist_ok=True)
    trainer = NativeTrainer(model, actions, lr=3e-4)

    # 初始状态
    h = model.init_state()
    token = BOS_TOKEN

    # 预生成的交互数据（用行动 token 生成真实命令输出）
    print("生成初始交互数据...")

    # 用 Docker 沙箱生成真实命令输出
    use_sandbox = False
    try:
        from sandbox.docker_env import DockerSandbox, DockerSandboxConfig
        sandbox = DockerSandbox(DockerSandboxConfig(network="none", memory_limit="256m", cpu_limit=2))
        sandbox.start()
        use_sandbox = True
    except Exception as e:
        print(f"  无 Docker: {e}")

    # 生成一系列 "思考 → 行动 → 观察结果 → 思考 → 行动 → ..."
    # 我们手动构造训练数据：思考字符 + 行动 token + 输出字符
    training_tokens = [BOS_TOKEN]

    # 第一轮：思考 → 行动(查看目录) → 看结果
    thought1 = "what files are here? "
    training_tokens.extend(text_to_tokens(thought1))
    training_tokens.append(ACTION_START + 0)  # ls -la

    if use_sandbox:
        r = sandbox.execute("bash", "ls -la", None)
        result1 = (r.stdout or "")[:200]
    else:
        result1 = "total 8\ndrwxr-xr-x 2 root root 4096 Jan 1 00:00 .\ndrwxr-xr-x 3 root root 4096 Jan 1 00:00 ..\n-rw-r--r-- 1 root root 0 Jan 1 00:00 test.txt"
    training_tokens.extend(text_to_tokens(result1))

    # 第二轮：思考 → 行动(查看系统信息) → 看结果
    thought2 = "\nlet me check who i am. "
    training_tokens.extend(text_to_tokens(thought2))
    training_tokens.append(ACTION_START + 3)  # whoami

    if use_sandbox:
        r = sandbox.execute("bash", "whoami", None)
        result2 = (r.stdout or "")[:100]
    else:
        result2 = "root"
    training_tokens.extend(text_to_tokens(result2))

    # 第三轮：思考 → 行动(检查磁盘) → 看结果
    thought3 = "\nchecking disk space. "
    training_tokens.extend(text_to_tokens(thought3))
    training_tokens.append(ACTION_START + 4)  # df -h

    if use_sandbox:
        r = sandbox.execute("bash", "df -h /", None)
        result3 = (r.stdout or "")[:200]
    else:
        result3 = "Filesystem      Size  Used Avail Use% Mounted on\noverlay         128G   33G   95G  26% /"
    training_tokens.extend(text_to_tokens(result3))

    # 更多轮次
    for i in range(3):
        thought = f"\ninteresting. let me try another command. "
        training_tokens.extend(text_to_tokens(thought))
        action = (i + 5) % N_ACTION
        training_tokens.append(ACTION_START + action)
        cmd = actions.get_cmd(action)
        if use_sandbox:
            r = sandbox.execute("bash", cmd, None)
            result = (r.stdout or "")[:200]
        else:
            result = f"output of {cmd}"
        training_tokens.extend(text_to_tokens(result))

    if use_sandbox:
        sandbox.stop()

    print(f"  训练序列长度: {len(training_tokens)} tokens")
    print(f"  其中行动 token: {sum(1 for t in training_tokens if t >= ACTION_START and t < ACTION_START + N_ACTION)}")
    print()

    # ── 训练循环 ────────────────────────────────────────
    print(f"{'步数':>6} | {'字符损失':>8} | {'好奇':>6} | {'行动率':>6} | {'LR':>8}")
    print("-" * 45)

    steps_per_epoch = len(training_tokens) - 1
    for epoch in range(steps // steps_per_epoch + 1):
        for i in range(len(training_tokens) - 1):
            token = training_tokens[i]
            next_tok = training_tokens[i + 1]

            h, m = trainer.train_step(h, token, next_tok)
            trainer.scheduler.step()

            if (i + 1) % 200 == 0 and (i == len(training_tokens) - 2):
                avg_char = sum(trainer.char_losses[-200:]) / 200
                avg_curio = sum(trainer.curiosities[-200:]) / 200
                act_rate = sum(trainer.action_taken[-200:]) / 200
                print(
                    f"{trainer.step_count:>6,} | "
                    f"{avg_char:>8.4f} | "
                    f"{avg_curio:>6.3f} | "
                    f"{act_rate:>6.1%} | "
                    f"{m['lr']:>8.2e}"
                )

    # 最终保存
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    final_char = sum(trainer.char_losses[-500:]) / 500
    final_curio = sum(trainer.curiosities[-500:]) / 500

    print(f"\n✅ 训练完成！{trainer.step_count} 步")
    print(f"   最终字符损失: {final_char:.4f}（随机 ≈ {torch.tensor(N_CHAR).float().log().item():.4f}）")
    print(f"   最终好奇心: {final_curio:.3f}")

    # 生成测试
    print("\n🧪 自主生成测试:")
    h = model.init_state()
    h, seq = trainer.generate_interaction(h, n_steps=30)
    text_parts = tokens_to_text(seq)
    action_parts = [
        f"[{actions.get_cmd(t - ACTION_START)}]"
        for t in seq if ACTION_START <= t < ACTION_START + N_ACTION
    ]
    print(f"  思考输出: {text_parts[:200]}")
    print(f"  行动输出: {', '.join(action_parts)}")

    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--output", type=str, default="checkpoints/native-v1")
    args = parser.parse_args()

    model = NativeCore(hidden_dim=args.hidden)
    actions = ActionMap()

    print(f"🌐 原生交互架构 (字符 + 行动 token)")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {N_CHAR} 字符 + {N_ACTION} 行动 + {N_SPECIAL} 特殊")
    print(f"   隐藏: {args.hidden} | 步: {args.steps}")
    print()

    trainer = train_loop(model, actions, steps=args.steps, output_dir=args.output)
