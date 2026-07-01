#!/usr/bin/env python3
"""
在线好奇心训练 v2 — 基于预训练 WordModel + 嫁接行动能力

流程: 观察 → 预测 → 行动 → Docker执行 → 对比 → 学习
"""

import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.pretrain_offline import WordModel, encode, decode, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE, TOTAL_VOCAB
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


# ── 命令集 ────────────────────────────────────────────────
COMMANDS = [
    "ls", "ls -la", "pwd", "date -u", "whoami", "id",
    "cat /etc/hostname", "uname -a", "df -h /", "free -h",
    "uptime", "echo hello", "hostname", "who -b",
    "ls /tmp", "du -sh /tmp",
]
N_ACTION = len(COMMANDS)

# 行动 token 放在词表之外
ACTION_START = VOCAB_SIZE  # 1500
TOTAL_VOCAB_WITH_ACTIONS = ACTION_START + N_ACTION + 3  # 1500 + 16 + 3 = 1519


class CuriousWordModel(nn.Module):
    """
    预训练 WordModel + 行动/世界模型/好奇心嫁接

    保留预训练权重，新增轻量级模块
    """

    def __init__(self, pretrained_path="checkpoints/word-offline-v1/model_best.pt"):
        super().__init__()
        self.vocab_size = VOCAB_SIZE

        # 复制预训练模型的结构
        self.token_embed = nn.Embedding(TOTAL_VOCAB_WITH_ACTIONS, 96)
        self.rnn = nn.GRUCell(96, 256)
        self.shared = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.lm_head = nn.Linear(256, TOTAL_VOCAB_WITH_ACTIONS)

        # 加载预训练权重（只加载匹配的部分）
        pretrained = WordModel(hidden_dim=256)
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        pretrained.load_state_dict(sd, strict=False)

        # 复制匹配的权重（用 copy_ 保持参数注册）
        with torch.no_grad():
            self.token_embed.weight[:VOCAB_SIZE].copy_(pretrained.token_embed.weight[:VOCAB_SIZE])
            self.rnn.weight_ih.copy_(pretrained.rnn.weight_ih)
            self.rnn.weight_hh.copy_(pretrained.rnn.weight_hh)
            self.rnn.bias_ih.copy_(pretrained.rnn.bias_ih)
            self.rnn.bias_hh.copy_(pretrained.rnn.bias_hh)
            self.shared[0].weight.copy_(pretrained.shared[0].weight)
            self.shared[0].bias.copy_(pretrained.shared[0].bias)
            self.shared[1].weight.copy_(pretrained.shared[1].weight)
            self.shared[1].bias.copy_(pretrained.shared[1].bias)
            self.lm_head.weight[:VOCAB_SIZE].copy_(pretrained.lm_head.weight[:VOCAB_SIZE])
            self.lm_head.bias[:VOCAB_SIZE].copy_(pretrained.lm_head.bias[:VOCAB_SIZE])

        # ── 新增模块（从零初始化） ──
        # 行动头
        self.action_head = nn.Linear(256, N_ACTION)
        # 世界模型（预测行动输出的首几个 token）
        self.world_head = nn.Linear(256, VOCAB_SIZE)
        # 好奇心门
        self.curiosity_gate = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        nn.init.orthogonal_(self.action_head.weight, gain=0.5)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.world_head.weight, gain=0.5)
        nn.init.zeros_(self.world_head.bias)

    def init_state(self, batch=1):
        return torch.zeros(batch, 256)

    def step(self, h, token):
        if token.dim() == 0:
            token = token.unsqueeze(0)
        emb = self.token_embed(token)
        h_new = self.rnn(emb, h)
        s = self.shared(h_new)
        lm_logits = self.lm_head(s)
        action_logits = self.action_head(s)
        world_logits = self.world_head(s)
        curiosity = self.curiosity_gate(s)
        return h_new, lm_logits, action_logits, world_logits, curiosity


def run_cmd(sandbox, cmd):
    try:
        r = sandbox.execute("bash", cmd, None)
        return (r.stdout or "").strip()[:2000]
    except:
        return ""


def train(cycles=300, lr=5e-5, output_dir="checkpoints/word-curious-v2"):
    os.makedirs(output_dir, exist_ok=True)

    model = CuriousWordModel()
    print(f"🧠 好奇心在线训练")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {VOCAB_SIZE} 词 + {N_ACTION} 行动")

    cfg = DockerSandboxConfig(network="none", memory_limit="512m", cpu_limit=2, timeout_per_action=15)
    sandbox = DockerSandbox(cfg)
    sandbox.start()
    print(f"   Docker: {sandbox.container.id[:12]}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    h = model.init_state()
    rng = random.Random(42)
    curio_log = []
    action_log = []

    print(f"{'轮':>4} | {'行动':>22} | {'损失':>8} | {'好奇':>5} | {'多样':>4}")
    print("-" * 55)

    for cycle in range(cycles):
        # ── 选择行动（ε-贪心） ──
        explore = rng.random() < max(0.1, 0.6 - cycle / cycles * 0.5)
        if explore:
            action_idx = rng.randint(0, N_ACTION - 1)
        else:
            h_t, _, action_logits, _, _ = model.step(h.detach(), torch.tensor([BOS_TOKEN]))
            ap = F.softmax(action_logits.squeeze(0) / 0.8, dim=-1)
            action_idx = torch.multinomial(ap, 1).item()

        cmd = COMMANDS[action_idx]
        action_log.append(action_idx)

        # ── Docker 执行 ──
        output = run_cmd(sandbox, cmd)
        full_text = f"$ {cmd}\n{output}\n"
        full_tokens = encode(full_text)

        # ── 逐 token 处理 + 学习 ──
        total_loss = 0
        n_ok = 0
        h_seq = h.detach()

        for i in range(min(len(full_tokens) - 1, 120)):
            tok_t = torch.tensor([full_tokens[i]], dtype=torch.long)
            h_seq, lm_logits, act_logits, world_logits, curiosity = model.step(h_seq, tok_t)

            next_tok = full_tokens[i + 1]
            if next_tok >= VOCAB_SIZE:
                continue

            # 语言模型损失（保留预训练能力）
            lm_loss = F.cross_entropy(
                lm_logits[:, :VOCAB_SIZE],
                torch.tensor([next_tok]),
            )

            # 世界模型损失（预测行动输出）
            if i < len(full_tokens) - 5:  # 只对前几个输出 token 算世界模型
                wm_loss = F.cross_entropy(
                    world_logits,
                    torch.tensor([next_tok]),
                )
            else:
                wm_loss = torch.tensor(0.0)

            # 好奇心门
            with torch.no_grad():
                curio_target = torch.sigmoid(lm_loss.detach() - 0.5)
            curio_loss = F.mse_loss(
                curiosity.squeeze(-1),
                curio_target.expand(1),
            )

            total = lm_loss + 0.2 * wm_loss + 0.05 * curio_loss
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            total_loss += lm_loss.item()
            n_ok += 1
            h_seq = h_seq.detach()

        avg_loss = total_loss / max(n_ok, 1)
        curio_log.append(avg_loss)

        if (cycle + 1) % 15 == 0:
            uniq = len(set(action_log[-60:])) if len(action_log) >= 60 else len(set(action_log))
            avg_c = sum(curio_log[-15:]) / 15
            print(f"{cycle+1:>4} | {cmd:>22} | {avg_loss:>8.4f} | {avg_c:>5.3f} | {uniq:>4}")

        if (cycle + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{output_dir}/model-c{cycle+1}.pt")

    sandbox.stop()
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ {cycles} 轮完成")
    print(f"   好奇度: {sum(curio_log[-20:])/20:.4f}")
    print(f"   多样性: {len(set(action_log))}/{N_ACTION}")

    # ── 测试 ──
    print("\n🧪 生成:")
    h = model.init_state()
    token = BOS_TOKEN
    out = []
    with torch.no_grad():
        for _ in range(80):
            h, lm_logits, *_ = model.step(h, torch.tensor([token]))
            h = h.detach()
            lp = F.softmax(lm_logits[:, :VOCAB_SIZE].squeeze(0) / 0.7, dim=-1)
            token = torch.multinomial(lp, 1).item()
            if token == EOS_TOKEN: break
            out.append(token)
    print(f"  {decode(out)[:200]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--output", type=str, default="checkpoints/word-curious-v2")
    args = parser.parse_args()
    train(cycles=args.cycles, output_dir=args.output)
