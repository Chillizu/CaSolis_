"""测试不同命令对预训练模型的困惑度（= 好奇度）"""
import asyncio, torch, sys
sys.path.insert(0, ".")
from scripts.pretrain_offline import WordModel, encode
from sandbox.docker_env import DockerSandbox, DockerSandboxConfig


def perplexity(model, text):
    tokens = encode(text)[:100]
    if not tokens:
        return 0.0
    h = torch.zeros(1, 256)
    total = 0.0
    with torch.no_grad():
        for i in range(len(tokens) - 1):
            h, logits = model.step(h, torch.tensor([tokens[i]]))
            h = h.detach()
            loss = torch.nn.functional.cross_entropy(
                logits, torch.tensor([tokens[i+1]]), reduction="mean"
            )
            total += loss.item()
    return total / max(len(tokens) - 1, 1)


async def main():
    model = WordModel(hidden_dim=256)
    model.load_state_dict(torch.load(
        "checkpoints/word-offline-v1/model_best.pt", map_location="cpu", weights_only=True
    ))
    model.eval()

    cfg = DockerSandboxConfig(network="none", memory_limit="256m", cpu_limit=2, timeout_per_action=10)
    s = DockerSandbox(cfg)
    s.start()

    tests = [
        "ls", "pwd", "whoami", "hostname", "date -u",
        "cat /etc/hostname", "uname -a", "uptime", "echo hello",
        "id", "df -h /", "free -h", "who -b",
        "cat /etc/passwd | head -5",
        "ls /proc | head -10",
        "ls /dev | head -10",
        "cat /etc/services | head -10",
        "cat /etc/hosts",
        "env | head -10",
        "cat /dev/urandom | head -c 50 2>/dev/null",
        "ls /nonexistent 2>&1",
        "sl 2>/dev/null; echo 'not found'",
    ]

    print(f"{'困惑度':>8} | {'命令':>40} | {'输出预览':>30}")
    print("-" * 80)
    for cmd in tests:
        r = await s.execute("bash", cmd, None)
        out = (r.stdout or "").strip()[:500]
        ppl = perplexity(model, out) if out else 9.999
        print(f"{ppl:>8.3f} | {cmd:>40} | {out[:30]}")

    s.stop()

asyncio.run(main())
