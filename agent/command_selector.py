"""
命令选择器 — 在安全约束下探索新命令

CUSTOM 意图的核心: 从 SAFE_COMMANDS 中根据状态、新颖度、历史选择命令
"""

import random
import re
import math
from collections import defaultdict

import json, os

# P8.4c: 从 config/command_registry.json 加载 CUSTOM_COMMANDS
# 保留旧名引兼容性
_CUSTOM_COMMANDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "command_registry.json"
)
def _load_custom_commands():
    try:
        with open(_CUSTOM_COMMANDS_PATH) as f:
            registry = json.load(f)
        return registry.get("custom_commands", {})
    except Exception:
        # 回退到内联
        from benchmark.template_engine import _EMBEDDED_CUSTOM_COMMANDS
        return dict(_EMBEDDED_CUSTOM_COMMANDS)

CUSTOM_COMMANDS = _load_custom_commands()


class CommandSelector:
    """
    自适应命令选择器
    
    跟踪每个命令的使用历史和效果, 根据当前状态选命令:
    - 未探索的命令优先 (exploration bonus)
    - 在状态上下文中有相关性的命令加分
    - 成功率低的命令降权
    """

    def __init__(self):
        # cmd_name → {successes, tries, total_novelty, last_output}
        self.history: dict[str, dict] = {}
        for cmd_name in CUSTOM_COMMANDS:
            self.history[cmd_name] = {
                "tries": 0,
                "successes": 0,
                "novelty_sum": 0.0,
                "last_novelty": 0.0,
                "satiation": 0.0,  # 厌腻度: 0~1, 越高越不想选
            }
        self.last_chosen: list[str] = []  # 最近 N 个选择的命令
        self._step_counter = 0

    def _decay_satiation(self):
        """每步衰减所有命令的厌腻度"""
        self._step_counter += 1
        for meta in self.history.values():
            # 每步恢复 2% (更持久, 强迫换命令)
            meta["satiation"] = max(0.0, meta["satiation"] * 0.98)

    def select(self, state_text: str, rnd_novelty: float = 0.0, forced: str | None = None) -> list[str]:
        """
        根据状态和 RND 新颖度选择一个自定义命令, 返回 args list
        """
        self._decay_satiation()

        if forced and forced in CUSTOM_COMMANDS:
            cmd_name = forced
        else:
            cmd_name = self._score_and_select(state_text, rnd_novelty)

        # 选中 → 大幅增加厌腻度 (防锁死)
        # 选中 → 厌腻度指数增长 (每选一次几乎翻倍的惩罚)
        old = self.history[cmd_name]["satiation"]
        self.history[cmd_name]["satiation"] = min(1.0, old * 2.0 + 0.1)

        # 构建 args
        template = CUSTOM_COMMANDS[cmd_name]
        args = list(template["args"])
        args = self._fill_args(args, state_text)

        self.last_chosen.append(cmd_name)
        if len(self.last_chosen) > 10:
            self.last_chosen.pop(0)

        return args

    def _score_and_select(self, state_text: str, rnd_novelty: float) -> str:
        """对所有候选命令打分并选择"""
        scores = {}

        for cmd_name, meta in self.history.items():
            template = CUSTOM_COMMANDS[cmd_name]
            score = 0.0

            # 探索奖励: 从未试过 → 高分
            if meta["tries"] == 0:
                score += 3.0
            else:
                # 成功率
                success_rate = meta["successes"] / max(meta["tries"], 1)
                score += success_rate * 1.0

                # 新颖度衰减: 很久没选 → 高探索价值
                recency_bonus = 0.0
                for i, last in enumerate(reversed(self.last_chosen)):
                    if last == cmd_name:
                        recency_bonus = (i + 1) / 10.0  # 越久远越高
                        break
                    if i == len(self.last_chosen) - 1:
                        recency_bonus = 1.0  # 从未见过
                score += recency_bonus * 0.5

            # 状态相关性
            desc = template.get("desc", "")
            if desc:
                for word in state_text.lower().split():
                    if word in desc.lower():
                        score += 0.3

            # 厌腻感: 刚用过的命令大幅降低分数
            satiation = meta.get("satiation", 0.0)
            score *= (1.0 - satiation * 0.85)  # 最多降 85%

            # 随机抖动 (+/- 30%) 增加多样性
            score *= 0.7 + random.random() * 0.6

            scores[cmd_name] = score

        # Softmax 采样 (替代 argmax, 防止字典顺序倾斜)
        temp = 0.3  # 温度: 越低越确定性
        names = list(scores.keys())
        vals = [scores[n] for n in names]
        max_v = max(vals) if vals else 1.0
        probs = [math.exp((v - max_v) / temp) for v in vals]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            best = random.choices(names, weights=probs, k=1)[0]
        else:
            best = random.choice(names)
        return best

    def _fill_args(self, args: list[str], state_text: str) -> list[str]:
        """填充 {path}, {cmd}, {target} 等模板变量"""
        filled = []
        for arg in args:
            if "{path}" in arg:
                # 从 state_text 提取路径
                path = self._extract_path(state_text) or "/etc/hostname"
                arg = arg.replace("{path}", path)
            if "{cmd}" in arg:
                cmd = self._extract_cmd(state_text) or "python3"
                arg = arg.replace("{cmd}", cmd)
            filled.append(arg)
        return filled

    def _extract_path(self, text: str) -> str | None:
        """从状态文本中提取路径"""
        m = re.search(r"/[\w./-]+", text)
        return m.group(0) if m else None

    def _extract_cmd(self, text: str) -> str | None:
        """从状态文本中提取命令名"""
        # 常见命令关键词
        cmds = ["python3", "git", "docker", "curl", "node", "gcc", "java", "nginx"]
        for cmd in cmds:
            if cmd in text.lower():
                return cmd
        return None

    def record_result(self, args: list[str], novelty: float, success: bool):
        """记录命令执行结果"""
        if not args:
            return
        cmd_name = None
        for name, meta in CUSTOM_COMMANDS.items():
            if args and args[0] == meta["args"][0]:
                cmd_name = name
                break
        if cmd_name is None:
            return

        self.history[cmd_name]["tries"] += 1
        self.history[cmd_name]["novelty_sum"] += novelty
        self.history[cmd_name]["last_novelty"] = novelty
        if success:
            self.history[cmd_name]["successes"] += 1

    def stats(self) -> dict:
        """返回命令选择统计"""
        explored = sum(1 for m in self.history.values() if m["tries"] > 0)
        available = len(self.history)
        return {
            "explored": explored,
            "available": available,
            "exploration_rate": f"{explored/available*100:.0f}%",
            "last_chosen": self.last_chosen[-5:] if self.last_chosen else [],
        }


def test():
    sel = CommandSelector()
    # 模拟几步选择
    states = [
        "当前目录: /etc 已知文件: passwd 上步: 查看文件信息 历史: 无",
        "当前目录: /proc 已知文件: cpuinfo 上步: 查看 CPU 信息 历史: 无",
        "当前目录: / 已知文件:  上步: 查看系统进程 历史: 无",
    ]
    for s in states:
        args = sel.select(s, rnd_novelty=random.random())
        print(f"  state: {s[:40]}...")
        print(f"    args: {' '.join(args)}")
        sel.record_result(args, novelty=random.random(), success=True)

    print(f"\n统计: {sel.stats()}")


if __name__ == "__main__":
    test()
