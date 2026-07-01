#!/usr/bin/env python3
"""
元学习世界模型 — 学"命令→输出"的函数，不是背答案

核心:
  1. 编码命令文本 → 预测输出(embedding)
  2. 预测误差 = 好奇度
  3. 无聊时(低误差) → 主动尝试奇怪的新命令
  4. 自由命令生成 — 不只是 16 个预定义命令
"""

import os, sys, time, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.pretrain_offline import WordModel, encode, decode, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


# ── 基础命令池（用于初始探索） ──────────────────────────────
BASE_COMMANDS = [
    "ls", "ls -la", "pwd", "date", "whoami", "id", "hostname",
    "cat /etc/hostname", "uname -a", "uptime", "echo hello",
    "df -h /", "free -h", "who -b", "ls /tmp", "du -sh /tmp",
]

# 奇怪命令池（用于无聊时的探索）
WEIRD_COMMANDS = [
    "ls /dev", "ls /proc", "ls /sys", "cat /etc/passwd",
    "head -5 /etc/services", "wc /etc/hostname",
    "ls -la /dev/null", "cat /dev/null",
    "true", "false", "yes n 2>/dev/null | head -3",
    "seq 1 5", "printf 'hello\nworld\n'",
    "ls -R /etc 2>/dev/null | head -20",
    "find /etc -name '*.conf' 2>/dev/null | head -5",
    "grep root /etc/passwd 2>/dev/null",
    "sort /etc/hostname 2>/dev/null",
    "od -c /etc/hostname 2>/dev/null",
    "env | grep PATH", "echo $HOME $USER $SHELL",
]


class WorldModelTrainer:
    """
    世界模型训练器

    学习的是"命令→输出"的函数关系，不是特定命令的输出内容。
    好奇心驱动：预测误差低 → 无聊 → 试奇怪命令 → 高误差 → 学习
    """

    def __init__(self, pretrained_path="checkpoints/word-offline-v1/model_best.pt"):
        # 编码器（冻结）
        self.encoder = WordModel(hidden_dim=256)
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(sd, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # 世界模型（可训练）
        # 从命令编码 h → 预测输出编码
        self.world_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

        # 好奇心门（可训练）
        self.curiosity_gate = nn.Sequential(
            nn.Linear(512, 64),  # 拼接命令 h + 输出 h
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.params = list(self.world_model.parameters()) + list(self.curiosity_gate.parameters())

    def encode_text(self, text: str) -> torch.Tensor:
        """用冻结的 RNN 编码文本为向量"""
        tokens = encode(text)
        h = torch.zeros(1, 256)
        with torch.no_grad():
            for tok in tokens[:50]:
                tok_t = torch.tensor([tok], dtype=torch.long)
                h, _ = self.encoder.step(h, tok_t)
        return h.squeeze(0)

    def encode_and_train(self, cmd: str, output: str) -> tuple[float, float, float]:
        """编码命令+输出，学习预测"""
        # 编码命令
        cmd_h = self.encode_text(cmd)

        # 编码输出
        out_h = self.encode_text(output)

        # 世界模型预测输出
        pred_h = self.world_model(cmd_h)

        # 预测误差 = 好奇度
        pred_loss = F.mse_loss(pred_h, out_h)

        # 好奇心门
        h_concat = torch.cat([cmd_h, out_h])
        curiosity = self.curiosity_gate(h_concat).squeeze(0)

        # 训练
        curiosity_loss = F.mse_loss(
            curiosity,
            torch.sigmoid(pred_loss.detach() - 0.1),
        )

        total = pred_loss + 0.1 * curiosity_loss
        return total, pred_loss.item(), curiosity.item()

    def curiosity_for(self, cmd: str) -> float:
        """计算命令的好奇度（预测误差估计）"""
        cmd_h = self.encode_text(cmd)

        # 如果没有输出，用世界模型预测一个虚拟输出
        dummy = torch.zeros(256)
        pred_h = self.world_model(cmd_h)
        pred_err = F.mse_loss(pred_h, dummy).item()
        return pred_err


def select_action(wm: WorldModelTrainer, sandbox, known_commands, cycle: int, max_cycles: int):
    """智能选择行动 — 无聊时做奇怪的事"""

    # 动态无聊阈值：周期越大越爱探索
    boredom_threshold = max(0.05, 0.2 - cycle / max_cycles * 0.15)

    # 计算对所有已知命令的好奇度
    curiosities = {}
    for cmd in known_commands:
        c = wm.curiosity_for(cmd)
        curiosities[cmd] = c

    avg_curiosity = sum(curiosities.values()) / max(len(curiosities), 1)

    # 决策
    if avg_curiosity < boredom_threshold:
        # 😴 无聊了 → 试奇怪的东西
        weird_cmd = random.choice(WEIRD_COMMANDS)
        print(f"    😴 无聊(好奇{avg_curiosity:.3f}) → 试奇怪的: {weird_cmd}")
        return weird_cmd, "weird"
    else:
        # 🤔 好奇 → 选最意外的已知命令
        best_cmd = max(curiosities, key=curiosities.get)
        if random.random() < 0.3:
            # 偶尔还是试奇怪的
            weird_cmd = random.choice(WEIRD_COMMANDS)
            return weird_cmd, "curious_explore"
        return best_cmd, "curious"


def train_meta(
    cycles=300,
    lr=1e-3,
    output_dir="checkpoints/world-model-v1",
):
    os.makedirs(output_dir, exist_ok=True)

    # 加载世界模型
    wm = WorldModelTrainer()
    opt = torch.optim.AdamW(wm.params, lr=lr, weight_decay=1e-5)

    print(f"🧠 元学习世界模型 — 学命令→输出函数")
    print(f"   词表: {VOCAB_SIZE}")
    print(f"   冻结编码器: {sum(p.numel() for p in wm.encoder.parameters()):,}")
    print(f"   可训练: {sum(p.numel() for p in wm.params):,}")
    print()

    # Docker
    cfg = DockerSandboxConfig(network="none", memory_limit="512m", cpu_limit=2, timeout_per_action=15)
    sandbox = DockerSandbox(cfg)
    sandbox.start()
    print(f"   Docker: {sandbox.container.id[:12]}\n")

    # 已知命令集合
    known = set(BASE_COMMANDS)
    rng = random.Random(42)

    stats = {"loss": [], "curiosity": [], "weird_attempts": []}

    print(f"{'轮':>4} | {'命令':>30} | {'损失':>8} | {'好奇':>5} | {'类型':>12}")
    print("-" * 65)

    for cycle in range(cycles):
        # ── 选行动 ──
        cmd, action_type = select_action(wm, sandbox, list(known), cycle, cycles)
        known.add(cmd)

        # ── Docker 执行 ──
        try:
            r = sandbox.execute("bash", cmd, None)
            output = (r.stdout or "").strip()[:2000]
        except Exception as e:
            output = f""

        if not output:
            output = "(empty)"

        # ── 训练世界模型 ──
        total_loss, pred_loss, curiosity = wm.encode_and_train(cmd, output)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.params, max_norm=1.0)
        opt.step()

        stats["loss"].append(pred_loss)
        stats["curiosity"].append(curiosity)
        if action_type == "weird":
            stats["weird_attempts"].append(1)

        # ── 日志 ──
        if (cycle + 1) % 15 == 0:
            avg_l = sum(stats["loss"][-15:]) / 15
            avg_c = sum(stats["curiosity"][-15:]) / 15
            n_weird = sum(stats["weird_attempts"][-15:]) if len(stats["weird_attempts"]) > 15 else len(stats["weird_attempts"])
            print(f"{cycle+1:>4} | {cmd[:30]:>30} | {avg_l:>8.4f} | {avg_c:>5.3f} | {action_type:>12}")

    sandbox.stop()
    torch.save(wm.world_model.state_dict(), f"{output_dir}/world_model.pt")
    torch.save(wm.curiosity_gate.state_dict(), f"{output_dir}/curiosity_gate.pt")

    print(f"\n✅ {cycles} 轮完成")
    print(f"   好奇度趋势: {stats['curiosity'][0]:.3f} → {stats['curiosity'][-1]:.3f}")
    print(f"   奇怪尝试: {sum(stats['weird_attempts'])}/{cycles}")
    print(f"   已知命令: {len(known)}")

    # ── 展示学到的世界模型 ──
    print(f"\n🧪 世界模型预测测试:")
    test_cmds = ["ls", "date -u", "cat /etc/hostname", "ls /nonexistent 2>&1"]
    for c in test_cmds:
        cmd_h = wm.encode_text(c)
        pred = wm.world_model(cmd_h)
        loss = F.mse_loss(pred, torch.zeros(256)).item()
        print(f"   {c:30s} 预测确定性: {1.0/(loss+0.01):.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--output", type=str, default="checkpoints/world-model-v1")
    args = parser.parse_args()
    train_meta(cycles=args.cycles, output_dir=args.output)
