#!/usr/bin/env python3
"""
好奇心驱动探索 v3 — 用预训练 LM 的困惑度作为好奇心信号

核心循环:
  1. 模型对每个命令的输出计算困惑度 (=好奇度)
  2. 好奇度低 → 无聊 → 尝试奇怪命令
  3. 执行命令 → 看到真实输出 → 放进记忆
  4. 下次遇到类似命令就能预测得更好
"""

import os, sys, time, random, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from scripts.pretrain_offline import WordModel, encode, BOS_TOKEN
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


# ── 命令池 ────────────────────────────────────────────────
BASIC_CMDS = ["ls", "pwd", "whoami", "hostname", "date -u", "uptime",
              "echo hello", "uname -a", "id", "df -h /", "free -h",
              "ls /tmp", "who -b", "cat /etc/hostname", "du -sh /tmp"]

WEIRD_CMDS = [
    "cat /etc/passwd | head -5", "ls /proc | head -10",
    "ls /dev | head -10", "cat /etc/services | head -10",
    "env | head -10", "locale",
    "ls /nonexistent 2>&1",
    "sl 2>/dev/null; echo 'not found'",
    "cat /dev/urandom | head -c 50 2>/dev/null",
    "ls -R /etc 2>/dev/null | head -20",
    "find /etc -name '*.conf' 2>/dev/null | head -5",
    "grep root /etc/passwd 2>/dev/null",
    "head -20 /etc/group",
    "ls -la /bin | head -10",
]


class CuriosityDrivenAgent:
    """好奇心驱动的探索代理"""

    def __init__(self):
        # 冻结的预训练语言模型
        self.lm = WordModel(hidden_dim=256)
        self.lm.load_state_dict(torch.load(
            "checkpoints/word-offline-v1/model_best.pt", map_location="cpu",
            weights_only=True
        ))
        for p in self.lm.parameters():
            p.requires_grad = False
        self.lm.eval()

        # 经验记忆: {命令: (输出文本, 困惑度)}
        self.memory = {}

        # 探索统计
        self.novelty_bonus = {}  # 命令 → 尝试次数
        self.boredom_count = 0

    def perplexity(self, text: str) -> float:
        """计算模型对文本的困惑度"""
        tokens = encode(text)[:100]
        if len(tokens) < 2:
            return 0.0
        h = torch.zeros(1, 256)
        total = 0.0
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                h, logits = self.lm.step(h, torch.tensor([tokens[i]]))
                loss = F.cross_entropy(logits, torch.tensor([tokens[i+1]]), reduction="mean")
                total += loss.item()
        return total / (len(tokens) - 1)

    def curiosity_for(self, cmd: str, output: str) -> float:
        """计算对命令输出文本的好奇度"""
        if not output:
            return 0.5
        ppl = self.perplexity(output)
        # 记忆折扣：见过的命令好奇心降低
        if cmd in self.memory:
            times = self.novelty_bonus.get(cmd, 1)
            ppl = ppl / (1 + 0.3 * times)  # 见过的越多越无聊
        return ppl

    def remember(self, cmd: str, output: str):
        """记住命令输出"""
        self.memory[cmd] = output
        self.novelty_bonus[cmd] = self.novelty_bonus.get(cmd, 0) + 1

    def select_action(self, available_cmds: list) -> tuple[str, str]:
        """选择下一步行动"""
        # 对已知命令算好奇度
        curiosities = {}
        for cmd in available_cmds:
            last_out = self.memory.get(cmd, "")
            c = self.curiosity_for(cmd, last_out)
            curiosities[cmd] = c

        avg_c = sum(curiosities.values()) / max(len(curiosities), 1)

        if avg_c < 0.5:
            # 😴 无聊 — 尝试奇怪命令
            self.boredom_count += 1
            weird = [c for c in WEIRD_CMDS if c not in self.memory or self.novelty_bonus.get(c, 0) < 3]
            if weird:
                chosen = random.choice(weird)
            else:
                chosen = random.choice(WEIRD_CMDS + BASIC_CMDS)
            return chosen, "weird"
        else:
            # 🤔 好奇 — 选最意外的
            best = max(curiosities, key=curiosities.get)
            if random.random() < 0.2:
                weird = random.choice(WEIRD_CMDS)
                return weird, "curious_explore"
            return best, "familiar"


async def train_curious_v3(cycles=500, output_dir="checkpoints/curious-v3"):
    os.makedirs(output_dir, exist_ok=True)

    agent = CuriosityDrivenAgent()
    print(f"🧠 好奇心驱动探索 v3")
    print(f"   冻结 LM: {sum(p.numel() for p in agent.lm.parameters()):,}")

    cfg = DockerSandboxConfig(network="none", memory_limit="256m", cpu_limit=2, timeout_per_action=15)
    sandbox = DockerSandbox(cfg)
    sandbox.start()
    print(f"   Docker: {sandbox.container.id[:12]}\n")

    available = list(BASIC_CMDS)
    stats = {"ppl": [], "curiosity": [], "weird_ratio": []}

    print(f"{'轮':>4} | {'命令':>40} | {'困惑度':>7} | {'好奇':>5} | {'类型':>14}")
    print("-" * 70)

    for cycle in range(cycles):
        # ── 选行动 ──
        cmd, action_type = agent.select_action(available)

        # ── Docker 执行 ──
        try:
            r = await sandbox.execute("bash", cmd, None)
            output = (r.stdout or "").strip()[:2000]
        except Exception:
            output = ""

        # ── 计算好奇度并记忆 ──
        ppl = agent.perplexity(output) if output else 0
        curio = agent.curiosity_for(cmd, output)
        agent.remember(cmd, output)

        # 加入可用命令集
        if cmd not in available:
            available.append(cmd)

        stats["ppl"].append(ppl)
        stats["curiosity"].append(curio)
        stats["weird_ratio"].append(1 if action_type == "weird" else 0)

        # ── 日志 ──
        if (cycle + 1) % 25 == 0:
            avg_p = sum(stats["ppl"][-25:]) / 25
            avg_c = sum(stats["curiosity"][-25:]) / 25
            wr = sum(stats["weird_ratio"][-25:]) / 25 if len(stats["weird_ratio"]) >= 25 else 0
            print(f"{cycle+1:>4} | {cmd[:40]:>40} | {avg_p:>7.2f} | {avg_c:>5.2f} | {action_type:>14}")

    sandbox.stop()

    print(f"\n✅ {cycles} 轮完成")
    print(f"   困惑度趋势: {stats['ppl'][0]:.2f} → {stats['ppl'][-1]:.2f}")
    print(f"   好奇度趋势: {stats['curiosity'][0]:.2f} → {stats['curiosity'][-1]:.2f}")
    print(f"   奇怪尝试比例: {sum(stats['weird_ratio'])/len(stats['weird_ratio']):.1%}")
    print(f"   已知命令: {len(agent.memory)}")

    with open(f"{output_dir}/memory.json", "w") as f:
        json.dump(agent.memory, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--output", type=str, default="checkpoints/curious-v3")
    args = parser.parse_args()
    asyncio.run(train_curious_v3(cycles=args.cycles, output_dir=args.output))
