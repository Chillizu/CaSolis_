#!/usr/bin/env python3
"""
Native Online Training — 模型在 Docker 中自主探索学习

核心循环：
  观察(命令输出) → 读字符 → 思考 → 决定行动 → 执行 → 看结果 → 学习
"""

import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from arch.native import (
    NativeCore, text_to_tokens, tokens_to_text,
    N_CHAR, N_ACTION, ACTION_START, BOS_TOKEN, CHAR_START,
)
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


# 多样化的命令池
COMMANDS = [
    "ls", "pwd", "date -u", "whoami", "id",
    "cat /etc/hostname", "uname -a", "df -h /", "free -h",
    "uptime", "echo hello", "hostname", "who -b",
    "ls /tmp", "du -sh /tmp",
    "ls /etc | head -3", "cat /etc/hosts",
    "echo test123", "ls -la /home",
    "cat /etc/issue", "dmesg 2>/dev/null | tail -1",
    "env | head -3", "locale | head -3",
    "ls -d /tmp /etc /var /home",
    "which ls", "echo $HOME",
]


def train_online(steps: int = 100, output_dir: str = "checkpoints/native-online"):
    os.makedirs(output_dir, exist_ok=True)

    model = NativeCore(hidden_dim=256)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    print(f"🧠 在线训练 | 参数: {sum(p.numel() for p in model.parameters()):,}")

    # Docker
    cfg = DockerSandboxConfig(network="none", memory_limit="256m", cpu_limit=2, timeout_per_action=10)
    sandbox = DockerSandbox(cfg)
    sandbox.start()
    print(f"🐳 Docker: {sandbox.container.id[:12]}")

    h = model.init_state()
    step = 0
    losses = []
    rng = random.Random(42)

    print(f"\n{'步':>4} | {'行动':>20} | {'损失':>8} | {'多样':>4} | {'好奇':>5}")
    print("-" * 55)

    action_history = []

    for episode in range(steps):
        # 每次随机选一个命令
        cmd = rng.choice(COMMANDS)

        # ── 1. 模型思考 ──
        # 给模型一些"思考文本"
        thoughts = [
            "what's here? ", "let me check. ", "interesting. ",
            "i wonder. ", "let me see. ", "looking around. ",
            "hmm. ", "exploring. ", "what is this? ",
        ]
        thought = rng.choice(thoughts)
        thought_tokens = text_to_tokens(thought)

        total_loss = 0.0
        n_preds = 0

        # 逐个 token 处理思考（不累积梯度——思考不需要监督）
        for tok in thought_tokens:
            tok_t = torch.tensor([tok], dtype=torch.long)
            h, outputs = model.step(h, tok_t)
            h = h.detach()  # ⚡ 关键：不通过思考步骤反向传播

        # ── 2. 模型决定行动 ──
        # 用模型当前状态采样行动
        action_probs = F.softmax(outputs["action_logits"].squeeze(0) / 0.8, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        action_history.append(action)

        # 输出行动 token
        action_tok = ACTION_START + action
        tok_t = torch.tensor([action_tok], dtype=torch.long)
        h, outputs = model.step(h, tok_t)
        h = h.detach()  # ⚡ 行动后的状态独立于之前的思考

        # ── 3. Docker 执行 ──
        try:
            r = sandbox.execute("bash", cmd, None)
            result = (r.stdout or "").strip()[:800]
        except Exception as e:
            result = f"error: {e}"

        result_text = f"\n{result}\n"

        # ── 4. 观察结果 + 逐字符学习 ──
        result_tokens = text_to_tokens(result_text)

        for i in range(len(result_tokens) - 1):
            # 每个字符独立计算：当前 → 预测下一个
            tok_t = torch.tensor([result_tokens[i]], dtype=torch.long)
            h, outputs = model.step(h.detach(), tok_t)  # ⚡ 每步独立

            char_loss = F.cross_entropy(
                outputs["char_logits"],
                torch.tensor([result_tokens[i+1]], dtype=torch.long),
            )

            # 好奇心门
            with torch.no_grad():
                target = torch.sigmoid(char_loss.detach() - 1.0)
            curio_loss = F.mse_loss(
                outputs["curiosity"].squeeze(-1),
                target.expand(1),
            )

            total = char_loss + 0.05 * curio_loss

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            total_loss += char_loss.item()
            n_preds += 1

        avg_loss = total_loss / max(n_preds, 1)
        losses.append(avg_loss)

        # ── 进度 ──
        if (episode + 1) % 10 == 0:
            uniq = len(set(action_history[-50:])) if len(action_history) >= 50 else len(set(action_history))
            avg_l = sum(losses[-10:]) / 10
            curio = outputs["curiosity"].item()
            print(f"{episode+1:>4} | {cmd:>20} | {avg_l:>8.4f} | {uniq:>4} | {curio:>5.3f}")

        # ── 自动保存 ──
        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), f"{output_dir}/model-e{episode+1}.pt")

    sandbox.stop()
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ 完成！{steps} 轮")
    print(f"   最终损失: {sum(losses[-20:])/20:.4f}")
    print(f"   行动多样性: {len(set(action_history))}/{N_ACTION}")
    print(f"   模型已保存: {output_dir}/model_final.pt")

    # ── 测试 ──
    print("\n🧪 自主生成测试:")
    h = model.init_state()
    token = BOS_TOKEN
    output_cmds = []
    for _ in range(40):
        tok_t = torch.tensor([token], dtype=torch.long)
        h, outputs = model.step(h, tok_t)
        h = h.detach()

        # ε-贪心
        if random.random() < 0.5:
            action = random.randint(0, N_ACTION - 1)
            token = ACTION_START + action
        else:
            ap = F.softmax(outputs["action_logits"].squeeze(0) / 0.8, dim=-1)
            action = torch.multinomial(ap, 1).item()
            if ap[action].item() > 0.15:
                token = ACTION_START + action
            else:
                cp = F.softmax(outputs["char_logits"].squeeze(0) / 1.0, dim=-1)
                token = torch.multinomial(cp, 1).item()

        if ACTION_START <= token < ACTION_START + N_ACTION:
            output_cmds.append(COMMANDS[token - ACTION_START])

    print(f"  自主选择的命令: {', '.join(output_cmds[:10])}")
    print(f"  多样性: {len(set(output_cmds))}/{N_ACTION}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="checkpoints/native-online")
    args = parser.parse_args()
    train_online(steps=args.steps, output_dir=args.output)
