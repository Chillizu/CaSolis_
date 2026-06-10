#!/usr/bin/env python3
"""Word-level 在线训练 — 思考链"""

import os, sys, time, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from arch.word_level import (
    WordLevelCore, encode, decode, COMMANDS, N_ACTION,
    ACTION_START, BOS_TOKEN, VOCAB_SIZE, TOTAL_VOCAB, SPECIAL_TOKENS,
)
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


def run_cmd(sandbox, cmd: str) -> str:
    """执行 Docker 命令并返回输出"""
    try:
        r = sandbox.execute("bash", cmd, None)
        return (r.stdout or "").strip()[:2000]
    except Exception as e:
        return f""


def train_online(cycles: int = 50, hidden_dim: int = 512, output_dir: str = "checkpoints/word-v1"):
    os.makedirs(output_dir, exist_ok=True)

    model = WordLevelCore(hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    print(f"🧠 词级思考链架构")
    print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   词表: {VOCAB_SIZE} 词 + {N_ACTION} 行动")

    # Docker
    cfg = DockerSandboxConfig(network="none", memory_limit="512m", cpu_limit=2, timeout_per_action=15)
    sandbox = DockerSandbox(cfg)
    sandbox.start()
    print(f"   Docker: {sandbox.container.id[:12]}")

    # ── 训练 ──
    h = model.init_state()
    rng = random.Random(42)
    total_steps = 0
    recent_losses = []
    action_history = []
    thought_lens = []

    print(f"\n{'轮':>4} | {'行动':>22} | {'损失':>8} | {'思考len':>6} | {'多样':>4} | {'好奇':>5}")
    print("-" * 65)

    for cycle in range(cycles):
        # ── 1. 生成思考链 ──
        # 模型得到 BOS token，然后开始"想"
        token = BOS_TOKEN
        h = h.detach()

        # 生成一个行动（探索初期用 ε-贪心）
        explore = rng.random() < max(0.3, 1.0 - cycle / cycles * 0.7)

        if explore:
            action_idx = rng.randint(0, N_ACTION - 1)
        else:
            tok_t = torch.tensor([token], dtype=torch.long)
            h, outputs = model.step(h, tok_t)
            action_probs = F.softmax(outputs["action_logits"].squeeze(0) / 0.8, dim=-1)
            action_idx = torch.multinomial(action_probs, 1).item()

        cmd = COMMANDS[action_idx]
        action_history.append(action_idx)

        # 生成思考 token（模型"想"一下）
        h = h.detach()
        thought_tokens = []
        for _ in range(3):  # 简短思考
            tok_t = torch.tensor([token], dtype=torch.long)
            h, outputs = model.step(h, tok_t)
            h = h.detach()
            # 采样一个词 token
            lm_probs = F.softmax(outputs["lm_logits"].squeeze(0) / 1.0, dim=-1)
            next_token = torch.multinomial(lm_probs, 1).item()
            # 保证是词 token
            next_token = next_token % VOCAB_SIZE
            thought_tokens.append(next_token)
            token = next_token

        thought_lens.append(len(thought_tokens))

        # ── 2. 执行行动 ──
        # 输出行动 token
        action_tok = ACTION_START + action_idx
        tok_t = torch.tensor([action_tok], dtype=torch.long)
        h, outputs = model.step(h, tok_t)
        h = h.detach()

        # Docker 执行
        result = run_cmd(sandbox, cmd)

        # ── 3. 观察结果 + 学习 ──
        if result:
            result_tokens = encode(result)
        else:
            result_tokens = []

        # 世界模型预测
        batch_loss = 0.0
        n_preds = 0

        if result_tokens:
            # 用世界模型预测第一个输出 token
            h_wm, wm_out = model.step_with_action(h, tok_t, torch.tensor([action_idx]))
            pred_logits = wm_out["world_logits"]

            # 对每个输出 token 计算损失
            for i in range(min(len(result_tokens), 50)):  # 限制长度
                target = torch.tensor([result_tokens[i]], dtype=torch.long)

                # 词预测损失（下一个词）
                lm_loss = F.cross_entropy(wm_out["lm_logits"], target, reduction="mean")

                # 世界模型损失（预测行动输出）
                if target.item() < VOCAB_SIZE:
                    world_loss = F.cross_entropy(
                        wm_out["world_logits"].unsqueeze(0),
                        torch.tensor([target.item()]),
                    )
                else:
                    world_loss = lm_loss.clone()

                # 好奇心门
                with torch.no_grad():
                    curio_target = torch.sigmoid(world_loss.detach() - 1.0)
                curio_loss = F.mse_loss(
                    wm_out["curiosity"].squeeze(-1),
                    curio_target.expand(1),
                )

                total = lm_loss + 0.3 * world_loss + 0.05 * curio_loss

                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()

                batch_loss += world_loss.item()
                n_preds += 1

                # 下一步
                if i < len(result_tokens) - 1:
                    tok_t = torch.tensor([result_tokens[i]], dtype=torch.long)
                    h_wm, wm_out = model.step_with_action(
                        h_wm.detach(), tok_t,
                        torch.tensor([action_idx]),
                    )
        else:
            # 无输出时至少学点别的
            lm_loss = F.cross_entropy(
                outputs["lm_logits"],
                torch.tensor([rng.randint(0, VOCAB_SIZE - 1)]),
            )
            opt.zero_grad()
            lm_loss.backward()
            opt.step()
            batch_loss = lm_loss.item()
            n_preds = 1

        total_steps += n_preds
        avg_loss = batch_loss / max(n_preds, 1)
        recent_losses.append(avg_loss)

        # ── 进度 ──
        if (cycle + 1) % 5 == 0:
            uniq = len(set(action_history[-30:])) if action_history else 0
            avg_l = sum(recent_losses[-5:]) / 5
            curio = outputs["curiosity"].item() if n_preds > 0 else 0
            avg_thought = sum(thought_lens[-5:]) / 5
            print(
                f"{cycle+1:>4} | {cmd:>22} | {avg_l:>8.4f} | {avg_thought:>6.1f} | "
                f"{uniq:>4} | {curio:>5.3f}"
            )

        # ── 保存 ──
        if (cycle + 1) % 25 == 0:
            torch.save(model.state_dict(), f"{output_dir}/model-c{cycle+1}.pt")

    sandbox.stop()
    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    print(f"\n✅ {cycles} 轮完成")
    print(f"   最终损失: {sum(recent_losses[-10:])/10:.4f}")
    print(f"   行动多样性: {len(set(action_history))}/{N_ACTION}")

    # ── 测试 ──
    print("\n🧪 思考链生成测试:")
    h = model.init_state()
    token = BOS_TOKEN
    thoughts = []
    actions = []

    for step in range(30):
        tok_t = torch.tensor([token], dtype=torch.long)
        h, outputs = model.step(h, tok_t)
        h = h.detach()

        # 决定是否行动
        ap = F.softmax(outputs["action_logits"].squeeze(0) / 0.8, dim=-1)
        action = torch.multinomial(ap, 1).item()

        if ap[action].item() > 0.15:
            token = ACTION_START + action
            actions.append(COMMANDS[action])
        else:
            lp = F.softmax(outputs["lm_logits"].squeeze(0) / 1.0, dim=-1)
            token = torch.multinomial(lp, 1).item()
            if token < VOCAB_SIZE:
                thoughts.append(token)

    thought_text = decode(thoughts)
    print(f"  思考: {thought_text[:200]}")
    print(f"  行动: {', '.join(actions[:8])}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--output", type=str, default="checkpoints/word-v1")
    args = parser.parse_args()
    train_online(cycles=args.cycles, hidden_dim=args.hidden, output_dir=args.output)
