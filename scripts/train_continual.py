#!/usr/bin/env python3
"""
边学边做 — 持续学习循环

模型永远不冻结，永远在适应。
好奇心(困惑度)驱动探索，旧数据重放防止遗忘。
"""

import os, sys, time, random, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from scripts.pretrain_offline import WordModel, encode, decode, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE


# ── Docker 帮助函数 ────────────────────────────────────────
DOCKER_CONTAINER = None

def docker_start():
    global DOCKER_CONTAINER
    import uuid
    name = f"continual-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--name", name, "--rm",
                    "ubuntu:22.04", "sleep", "3600"],
                   capture_output=True, timeout=30)
    DOCKER_CONTAINER = name
    return name

def docker_exec(cmd: str) -> str:
    r = subprocess.run(
        ["docker", "exec", "-i", DOCKER_CONTAINER, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=15
    )
    return (r.stdout or "").strip()[:2000]

def docker_stop():
    if DOCKER_CONTAINER:
        subprocess.run(["docker", "kill", DOCKER_CONTAINER], capture_output=True)


class ContinualLearner:
    """
    持续学习者

    核心:
    - 模型永远在线学习
    - 困惑度 = 好奇心 = 探索动力
    - 旧数据重放 = 防遗忘
    - 检查点 = 安全网
    """

    def __init__(self, pretrained_path="checkpoints/word-offline-v1/model_best.pt"):
        self.model = WordModel(hidden_dim=256)
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.train()

        # 优化器（低学习率，保护已学知识）
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-5)

        # 经验缓存
        self.experience = {}  # cmd → [output1, output2, ...]

        # 探索统计
        self.visit_counts = {}  # cmd → 访问次数
        self.prev_losses = []   # 用于检测污染

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """模型对文本的困惑度"""
        tokens = encode(text)[:100]
        if len(tokens) < 2:
            return 0.0
        self.model.eval()
        h = torch.zeros(1, 256)
        total = 0.0
        for i in range(len(tokens) - 1):
            h, logits = self.model.step(h, torch.tensor([tokens[i]]))
            h = h.detach()
            total += F.cross_entropy(logits, torch.tensor([tokens[i+1]]), reduction="mean").item()
        self.model.train()
        return total / (len(tokens) - 1)

    def learn(self, text: str, n_steps: int = 1) -> float:
        """从文本中学习（多步训练）"""
        tokens = encode(text)[:200]
        if len(tokens) < 3:
            return 0.0

        total_loss = 0.0
        self.model.train()

        for _ in range(n_steps):
            h = torch.zeros(1, 256)
            for i in range(len(tokens) - 1):
                h, logits = self.model.step(h, torch.tensor([tokens[i]]))
                loss = F.cross_entropy(logits, torch.tensor([tokens[i+1]]), reduction="mean")
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.opt.step()
                total_loss += loss.item()
                h = h.detach()

        return total_loss / max(len(tokens) - 1, 1)

    def replay_old_data(self, jsonl_path="data/offline_raw.jsonl", n_samples: int = 5) -> float:
        """重放旧数据防遗忘"""
        import json
        if not os.path.exists(jsonl_path):
            return 0.0

        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        samples = random.sample(lines, min(n_samples, len(lines)))

        total = 0.0
        for d in samples:
            text = f"$ {d['cmd']}\n{d['output']}\n"
            total += self.learn(text, n_steps=1)
        return total / max(len(samples), 1)

    def select_action(self, available_cmds: list, weird_cmds: list) -> tuple[str, str]:
        """基于好奇心选择行动"""
        # 计算所有命令的好奇度
        curiosities = {}
        for cmd in available_cmds:
            history = self.experience.get(cmd, [])
            if history:
                ppl = self.perplexity(history[-1][:500])
                # 访问折扣
                visits = self.visit_counts.get(cmd, 0)
                curiosities[cmd] = ppl / (1 + 0.2 * visits)
            else:
                curiosities[cmd] = 10.0  # 没见过的 = 高好奇

        avg_c = sum(curiosities.values()) / max(len(curiosities), 1)
        best_cmd = max(curiosities, key=curiosities.get)

        # 决策
        if avg_c < 1.0:
            # 😴 无聊了 — 试真奇怪的
            return random.choice(weird_cmds), "weird"
        elif random.random() < 0.2:
            # 偶尔随机探索
            return random.choice(available_cmds + weird_cmds), "random"
        else:
            # 选最意外的
            return best_cmd, "curious"


def train_continual(
    cycles=1000, output_dir="checkpoints/continual-v1"
):
    os.makedirs(output_dir, exist_ok=True)

    agent = ContinualLearner()
    print(f"🧠 持续学习 — 边学边做")
    print(f"   参数: {sum(p.numel() for p in agent.model.parameters()):,}")
    print(f"   学习率: 5e-5 (低 = 保护旧知识)")

    # Docker
    name = docker_start()
    print(f"   Docker: {name}\n")

    # 命令池
    basic = [
        "ls", "ls -la", "pwd", "whoami", "hostname", "date -u",
        "uptime", "echo hello", "uname -a", "id", "df -h /",
        "free -h", "cat /etc/hostname", "who -b",
    ]
    weird = [
        "cat /etc/passwd | head -5", "ls /proc | head -10",
        "ls /dev | head -10", "cat /etc/services | head -10",
        "env | head -10", "ls /nonexistent 2>&1",
        "head -20 /etc/group", "ls -la /bin | head -10",
        "grep root /etc/passwd 2>/dev/null",
        "find /etc -name '*.conf' 2>/dev/null | head -5",
        "sl 2>/dev/null; echo not found",
        "cat /dev/urandom | head -c 50 2>/dev/null",
    ]

    available = list(basic)

    print(f"{'轮':>5} | {'命令':>35} | {'损失':>8} | {'困惑度':>7} | {'类型':>8}")
    print("-" * 70)

    stats = {"loss": [], "ppl": [], "type": []}

    for cycle in range(cycles):
        # ── 选行动 ──
        cmd, atype = agent.select_action(available, weird)
        if cmd not in available:
            available.append(cmd)

        # ── Docker 执行 ──
        try:
            output = docker_exec(cmd)
        except Exception:
            output = ""
        if not output:
            output = "(empty)"

        # ── 学习 ──
        text = f"$ {cmd}\n{output}\n"
        loss = agent.learn(text, n_steps=2)  # 每段学2遍

        # ── 记录 ──
        if cmd not in agent.experience:
            agent.experience[cmd] = []
        agent.experience[cmd].append(output)
        agent.visit_counts[cmd] = agent.visit_counts.get(cmd, 0) + 1

        # 当前困惑度
        ppl = agent.perplexity(output[:500])

        stats["loss"].append(loss)
        stats["ppl"].append(ppl)
        stats["type"].append(atype)

        # ── 日志 ──
        if (cycle + 1) % 25 == 0:
            avg_l = sum(stats["loss"][-25:]) / 25
            avg_p = sum(stats["ppl"][-25:]) / 25
            print(f"{cycle+1:>5} | {cmd[:35]:>35} | {avg_l:>8.4f} | {avg_p:>7.2f} | {atype:>8}")

        # ── 防遗忘 ──
        if (cycle + 1) % 100 == 0:
            replay_loss = agent.replay_old_data(n_samples=3)
            print(f"  ↻ 重放: loss={replay_loss:.4f}")
            torch.save(agent.model.state_dict(), f"{output_dir}/model-c{cycle+1}.pt")
            # 记录污染检测
            agent.prev_losses.append(avg_l)

    # ── 完成 ──
    docker_stop()
    torch.save(agent.model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ {cycles} 轮持续学习完成")
    print(f"   已知命令: {len(agent.experience)}")
    print(f"   总交互: {sum(agent.visit_counts.values())}")
    print(f"   好奇尝试: {stats['type'].count('curious')} / {stats['type'].count('weird')} / {stats['type'].count('random')}")

    # ── 测试 ├─
    test_cmds = ["ls", "pwd", "echo hello", "id", "cat /etc/passwd | head -3"]
    print("\n🧪 学后困惑度测试:")
    for c in test_cmds:
        out = agent.experience.get(c, [""])[-1]
        ppl = agent.perplexity(out[:500]) if out else 99
        visits = agent.visit_counts.get(c, 0)
        print(f"   {c:30s} 困惑度: {ppl:.2f} (见过{visits}次)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1500)
    parser.add_argument("--output", type=str, default="checkpoints/continual-v2")
    args = parser.parse_args()
    train_continual(cycles=args.cycles, output_dir=args.output)
