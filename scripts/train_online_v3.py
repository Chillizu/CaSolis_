#!/usr/bin/env python3
"""
原生交互 v3 — 真在线学习

每步都是实时的 Docker 命令执行。
行动有变化参数，输出有差异性，模型无法背诵。
"""

import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from arch.native import (
    NativeCore, ActionMap, text_to_tokens,
    N_CHAR, N_ACTION, ACTION_START, BOS_TOKEN,
    CHAR_START,
)
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


# ── 动态命令池 ────────────────────────────────────────────
# 同一个命令索引生成不同输出（基于随机参数）
COMMAND_POOL = {
    0:  ["ls {path}",           [("/", "/tmp", "/etc", "/var", "/home", "/bin", "/usr")]],
    1:  ["pwd",                 []],
    2:  ["date -u",             []],
    3:  ["id {user}",           [("", "root", "daemon")]],
    4:  ["df -{flag}",          [("h", "h /", "h /tmp")]],
    5:  ["cat /etc/{file}",     [("hostname", "hosts", "issue", "fstab", "os-release")]],
    6:  ["uname -{flag}",       [("a", "r", "s", "n")]],
    7:  ["echo {msg}",          [("hello world", "test", "hi", "ok", "done", "hello", "ABC", "123")]],
    8:  ["ls -la /tmp",         []],
    9:  ["whoami",              []],
    10: ["uptime",              []],
    11: ["free -{flag}",        [("h", "b", "k", "m")]],
    12: ["du -sh {path}",       [("/tmp", "/", "/etc", "/var")]],
    13: ["echo {msg}",          [("$HOME", "$USER", "$SHELL", "$PWD")]],
    14: ["ls -d {path}",        [(".", "/tmp", "/etc", "/var", "/home")]],
    15: ["who -{flag}",         [("b", "r", "d")]],
}


class DynamicActionMap(ActionMap):
    """扩展的行动映射，支持动态参数"""

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def get_cmd(self, action_idx: int) -> str:
        if action_idx in COMMAND_POOL:
            template, args_list = COMMAND_POOL[action_idx]
            if args_list:
                args = self.rng.choice(args_list[0]) if args_list else ""
                return template.format(path=args, flag=args, user=args, file=args, msg=args)
        return self.actions[action_idx % len(self.actions)]

    def reseed(self, seed: int = None):
        self.rng = random.Random(seed)


def execute_command(sandbox, cmd: str) -> str:
    """执行命令并返回输出"""
    try:
        r = sandbox.execute("bash", cmd, None)
        output = (r.stdout or "").strip()[:500]
        if not output:
            output = "(empty)"
        return output
    except Exception as e:
        return f"(error: {e})"


def run_training_cycle(
    sandbox, model, h, actions: DynamicActionMap,
) -> tuple:
    """
    一个完整的训练周期：

    1. 模型收到当前观察文本
    2. 模型逐个 token 处理（思考）
    3. 模型决定行动
    4. Docker 执行，返回新文本
    5. 学习
    """
    char_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    h = h.detach()
    obs_text = "\n$ "  # 初始"提示符"
    obs_tokens = text_to_tokens(obs_text)

    total_loss = 0.0
    n_tokens = 0
    action_count = 0
    recent_actions = []

    for step in range(50):  # 每段处理50个思考行动
        # 逐个 token 喂给模型
        for tok in obs_tokens:
            tok_t = torch.tensor([tok], dtype=torch.long)
            h, outputs = model.step(h, tok_t)

            # 下一步预测
            next_pred = outputs["char_logits"]
            # 训练目标：学会阅读（预测文本中的下一个字符）
            # 略过因为我们在"阅读"模式，不需要反向传播
            h = h.detach()

        # ── 决定是否行动 ──
        # 用行动头决定下一步
        action_probs = F.softmax(outputs["action_logits"].squeeze(0) / 0.7, dim=-1)
        # 温度退火
        action = torch.multinomial(action_probs, 1).item()

        # 如果模型选择行动
        cmd = actions.get_cmd(action)
        cmd_tokens = text_to_tokens(cmd)

        # ⚡ 关键：模型"思考"决策过程
        # 模型内部推理：决定用什么命令
        # 输出命令字符作为"思考"，然后输出行动 token
        for cmd_tok in cmd_tokens:
            tok_t = torch.tensor([cmd_tok], dtype=torch.long)
            h, outputs = model.step(h, tok_t)
            h = h.detach()

        # 输出行动 token
        action_token = ACTION_START + action
        tok_t = torch.tensor([action_token], dtype=torch.long)
        h, outputs = model.step(h, tok_t)

        # ── 环境执行 ──
        result = execute_command(sandbox, cmd)
        result_text = f"\n{result}\n$ "

        recent_actions.append(action)
        action_count += 1

        # ── 训练：世界模型应该能预测结果 ──
        result_tokens = text_to_tokens(result_text)
        total_result_loss = 0.0

        for i, pred_tok in enumerate(result_tokens):
            if i == 0:
                continue  # 跳过第一个（预测下一个）
            tok_t = torch.tensor([result_tokens[i-1]], dtype=torch.long)
            h, outputs = model.step(h, tok_t)

            # 损失：预测下一个字符
            loss = char_loss_fn(
                outputs["char_logits"],
                torch.tensor([pred_tok], dtype=torch.long),
            )

            # 好奇心门损失
            with torch.no_grad():
                curiosity_target = torch.sigmoid(loss.detach() - 1.0)
            curiosity_loss = F.mse_loss(
                outputs["curiosity"].squeeze(-1),
                curiosity_target.expand(1),
            )

            # 行动损失（当遇到下一个行动 token 时）
            action_loss = F.cross_entropy(
                outputs["action_logits"],
                torch.tensor([action], dtype=torch.long) if i < len(result_tokens)-1 and ACTION_START <= result_tokens[i+1] < ACTION_START+N_ACTION else torch.tensor([0], dtype=torch.long),
                reduction="mean",
            ) if i < len(result_tokens)-1 and ACTION_START <= result_tokens[i+1] < ACTION_START+N_ACTION else torch.tensor(0.0)

            # 多样性奖励
            unique_ratio = len(set(recent_actions[-20:])) / N_ACTION
            diversity_bonus = 0.05 * (1.0 - unique_ratio) if len(recent_actions) >= 5 else 0.0

            total = loss + 0.05 * curiosity_loss + 0.02 * action_loss - diversity_bonus

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            h = h.detach()
            total_loss += loss.item()
            n_tokens += 1

        obs_text = result_text
        obs_tokens = text_to_tokens(obs_text)

        # 输出进度
        if (step + 1) % 5 == 0:
            avg_loss = total_loss / max(n_tokens, 1)
            act_unique = len(set(recent_actions[-20:]))
            print(f"  [步骤{step+1}] 损失: {avg_loss:.3f} | 行动: {cmd:20s} | 多样: {act_unique}/20 | 好奇: {outputs['curiosity'].item():.3f}")

    return h, {
        "avg_loss": total_loss / max(n_tokens, 1),
        "actions_taken": action_count,
        "unique_actions": len(set(recent_actions)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--output", type=str, default="checkpoints/native-v3")
    args = parser.parse_args()

    model = NativeCore(hidden_dim=args.hidden)
    actions = DynamicActionMap()

    print(f"🌐 原生交互 v3 — 真在线学习")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {N_CHAR} 字符 + {N_ACTION} 行动")
    print()

    # Docker 沙箱
    print("启动 Docker...", end=" ", flush=True)
    config = DockerSandboxConfig(
        network="none", memory_limit="256m",
        cpu_limit=2, timeout_per_action=10,
    )
    sandbox = DockerSandbox(config)
    sandbox.start()
    print(f"OK ({sandbox.container.id[:12]})")

    h = model.init_state()
    all_actions = []

    for cycle in range(args.cycles):
        print(f"\n{'='*50}")
        print(f"周期 {cycle + 1}/{args.cycles}")
        print(f"{'='*50}")

        actions.reseed(seed=cycle * 100)

        h, metrics = run_training_cycle(sandbox, model, h, actions)
        all_actions.extend([metrics.get("unique_actions", 0)])

        print(f"  平均损失: {metrics['avg_loss']:.4f}")

        if (cycle + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{args.output}/model-c{cycle+1}.pt")

    sandbox.stop()
    torch.save(model.state_dict(), f"{args.output}/model_final.pt")

    print(f"\n✅ 完成！{args.cycles} 个周期")
    print(f"   最终行动多样性: {all_actions[-1]}/20")


if __name__ == "__main__":
    main()
