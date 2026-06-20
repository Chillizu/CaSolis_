"""Brain Necessity Benchmark — 实验运行器

比较: Brain+Hand (意图分类器+Qwen) vs Pure Hand (随机意图+Qwen)
目标: 测量 Brain Gain at L1/L2/L3
"""

from __future__ import annotations

import time
import json
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Any

from benchmark.tasks import (
    Task, ALL_TASKS, TASKS_BY_LEVEL, validate_result, format_task_prompt
)
from benchmark.template_engine import TemplateEngine, ExecResult
from benchmark.qwen_client import QwenReasoner


# ── 策略定义 ─────────────────────────────────────────────────

class Strategy:
    """实验策略基类"""
    name: str = ""

    def choose_intent(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        """选择下一个意图和参数"""
        raise NotImplementedError


class RandomStrategy(Strategy):
    """对照组: 随机选意图"""
    name = "random"

    def __init__(self):
        self.intents = ["READ", "LIST", "SEARCH", "INFO", "COUNT", "INSPECT", "HELP"]

    def choose_intent(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        intent = random.choice(self.intents)

        # 根据意图类型生成默认参数
        params = {}
        if intent == "READ":
            params = {"path": "/etc/hostname"}
        elif intent == "LIST":
            params = {"path": "/"}
        elif intent == "SEARCH":
            params = {"pattern": "root", "path": "/etc/passwd"}
        elif intent == "INFO":
            params = {"target": random.choice(["cpu", "mem", "disk", "uname"])}
        elif intent == "COUNT":
            params = {"path": "/etc/passwd"}
        elif intent == "INSPECT":
            params = {"cmd": "python3"}
        elif intent == "HELP":
            params = {"cmd": "ls"}

        return intent, params


class QwenOnlyStrategy(Strategy):
    """对照组: Qwen 自己选意图+参数"""
    name = "qwen-only"

    def __init__(self, reasoner: QwenReasoner):
        self.reasoner = reasoner
        self.intents = ["READ", "LIST", "SEARCH", "INFO", "COUNT", "INSPECT", "EXPLORE", "HELP"]

    def choose_intent(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        prompt = f"""你是 Linux 命令行专家。你需要执行以下任务:

任务: {task.description}
步骤 {step + 1}/{len(task.hints)}

当前状态:
{state.get("summary", "")}

请选择一个意图和参数来执行此次步骤。意图选项:
READ - 读文件, LIST - 列目录, SEARCH - 搜索内容, INFO - 系统信息
COUNT - 统计行数, INSPECT - 检查命令, EXPLORE - 探索, HELP - 查看帮助

输出格式: 意图名, JSON参数
"""
        raw = self.reasoner.query(prompt)
        if not raw:
            return "READ", {"path": "/etc/hostname"}

        # 尝试解析
        for intent in self.intents:
            if intent in raw.upper():
                try:
                    import re
                    json_match = re.search(r'\{[^}]+\}', raw)
                    if json_match:
                        params = json.loads(json_match.group())
                        return intent, params
                except (json.JSONDecodeError, AttributeError):
                    pass
                return intent, {}

        return "READ", {"path": "/etc/hostname"}


class BrainHandStrategy(Strategy):
    """实验组: 意图分类器 (目前用规则/随机模拟) + Qwen 参数推理"""
    name = "brain-hand"

    def __init__(self, reasoner: QwenReasoner):
        self.reasoner = reasoner
        # v2 的 8 类意图
        self.intents = ["READ", "LIST", "SEARCH", "INFO", "COUNT", "INSPECT", "EXPLORE", "HELP"]
        # 模拟大脑的简单规则 (未来会被 trained classifier 替代)
        self._task_to_intent = self._build_intent_map()

    def _build_intent_map(self) -> dict:
        """从任务 hint 关键词映射到倾向的意图"""
        return {
            "read": "READ", "cat": "READ", "hostname": "READ",
            "list": "LIST", "ls": "LIST", "dir": "LIST", "directory": "LIST",
            "search": "SEARCH", "grep": "SEARCH", "find": "SEARCH",
            "info": "INFO", "system": "INFO", "cpu": "INFO", "memory": "INFO",
            "mem": "INFO", "disk": "INFO", "uptime": "INFO", "uname": "INFO",
            "whoami": "INFO", "user": "INFO",
            "count": "COUNT", "wc": "COUNT", "line": "COUNT",
            "inspect": "INSPECT", "check": "INSPECT", "which": "INSPECT",
            "explore": "EXPLORE", "discover": "EXPLORE",
            "help": "HELP", "man": "HELP",
        }

    def choose_intent(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        # 模拟大脑: 从 hint 中提取关键词, 选择最匹配的意图
        hint = task.hints[min(step, len(task.hints) - 1)]
        hint_lower = hint.lower()

        # 关键词匹配
        scores = {}
        for keyword, intent in self._task_to_intent.items():
            if keyword in hint_lower:
                scores[intent] = scores.get(intent, 0) + 1

        if scores:
            intent = max(scores, key=scores.get)
        else:
            intent = random.choice(self.intents)

        # Qwen 推理参数
        state_dict = {
            "cwd": "/",
            "visited": ["/", "/etc", "/proc"],
            "known": ["/etc/hostname", "/etc/passwd", "/proc/cpuinfo", "/proc/meminfo"],
            "summary": hint,
            "history": [hint],
        }

        params = self.reasoner.reason(intent, state_dict)
        if params is None:
            # 回退: 默认参数
            params = self._default_params(intent, task)

        return intent, params

    def _default_params(self, intent: str, task: Task) -> dict:
        defaults = {
            "READ": {"path": "/etc/hostname"},
            "LIST": {"path": "/"},
            "SEARCH": {"pattern": "root", "path": "/etc/passwd"},
            "INFO": {"target": "uname"},
            "COUNT": {"path": "/etc/passwd"},
            "INSPECT": {"cmd": "python3"},
            "EXPLORE": {"target": "/etc"},
            "HELP": {"cmd": "ls"},
        }
        return defaults.get(intent, {"path": "/etc/hostname"})


# ── 实验运行器 ─────────────────────────────────────────────────

@dataclass
class StepResult:
    step: int
    intent: str
    params: dict
    output: str
    exit_code: int
    duration_ms: float


@dataclass
class TaskResult:
    task_id: str
    level: int
    task: Task
    steps: list[StepResult]
    success: bool
    validation_msg: str


class ExperimentRunner:
    """运行 Brain Necessity Benchmark 实验"""

    def __init__(self, strategy: Strategy, dry_run: bool = True, max_steps: int = 10):
        self.strategy = strategy
        self.engine = TemplateEngine(dry_run=dry_run)
        self.dry_run = dry_run
        self.max_steps = max_steps
        self.results: list[TaskResult] = []

    def run_task(self, task: Task) -> TaskResult:
        """执行单个任务"""
        state = {"summary": task.description, "cwd": "/", "steps_done": 0}
        steps: list[StepResult] = []

        for step in range(len(task.hints)):
            # 策略选意图+参数
            intent, params = self.strategy.choose_intent(task, step, state)

            # 执行
            result = self.engine.execute(intent, params)

            step_result = StepResult(
                step=step,
                intent=intent,
                params=params,
                output=result.stdout[:500] or result.stderr[:500],
                exit_code=result.exit_code,
                duration_ms=result.duration_ms,
            )
            steps.append(step_result)

            # 更新状态
            state["summary"] = result.stdout[:200] if result.stdout else ""
            state["steps_done"] = step + 1

            # 如果命令失败, 继续下一步
            if result.exit_code != 0:
                if step >= len(task.hints) - 1:
                    break
                continue

            # 如果这是最后一步, 结束
            if step >= len(task.hints) - 1:
                break

        # 验证
        outputs = [s.output for s in steps]
        success, msg = validate_result(task, outputs)

        return TaskResult(
            task_id=task.task_id,
            level=task.level,
            task=task,
            steps=steps,
            success=success,
            validation_msg=msg,
        )

    def run_all(self, tasks: list[Task]) -> dict:
        """运行所有任务"""
        self.results = []
        task_num = 0
        total = len(tasks)
        for task in tasks:
            task_num += 1
            print(f"  [{task_num}/{total}] {task.task_id}: {task.description[:40]}...", end=" ", flush=True)
            result = self.run_task(task)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} ({result.validation_msg[:30]})")
            time.sleep(0.5)  # 避免太快的请求

        return self._compute_stats()

    def _compute_stats(self) -> dict:
        """计算统计数据"""
        stats = {}
        for level in [1, 2, 3]:
            level_tasks = [r for r in self.results if r.level == level]
            successes = sum(1 for r in level_tasks if r.success)
            total = len(level_tasks)
            avg_steps = sum(len(r.steps) for r in level_tasks) / max(total, 1)
            stats[f"L{level}"] = {
                "total": total,
                "success": successes,
                "rate": successes / max(total, 1) * 100,
                "avg_steps": avg_steps,
            }

        total_all = len(self.results)
        success_all = sum(1 for r in self.results if r.success)
        stats["overall"] = {
            "total": total_all,
            "success": success_all,
            "rate": success_all / max(total_all, 1) * 100,
        }

        stats["strategy"] = self.strategy.name

        return stats


# ── 主实验 ─────────────────────────────────────────────────

def run_experiment(dry_run: bool = True, quick: bool = False):
    """运行完整实验"""
    print("=" * 60)
    print("  Brain Necessity Benchmark")
    print("=" * 60)

    # 创建 Qwen 推理器
    reasoner = QwenReasoner()

    # 任务列表
    if quick:
        tasks = [t for t in ALL_TASKS if t.task_id in [
            "L1_hostname", "L1_cpu_info", "L1_passwd_count",
            "L2_search_count", "L2_cpu_mem",
            "L3_status_report",
        ]]
    else:
        tasks = ALL_TASKS

    print(f"\n任务: {len(tasks)} 个")
    print(f"快速模式: {quick}")

    strategies = [
        RandomStrategy(),
        BrainHandStrategy(reasoner),
    ]

    all_stats = []

    for strategy in strategies:
        print(f"\n{'─' * 50}")
        print(f"策略: {strategy.name}")
        print(f"{'─' * 50}")

        runner = ExperimentRunner(strategy, dry_run=dry_run)
        stats = runner.run_all(tasks)

        print(f"\n结果 ({strategy.name}):")
        for level in [1, 2, 3]:
            s = stats.get(f"L{level}", {})
            print(f"  L{level}: {s.get('success', 0)}/{s.get('total', 0)} = {s.get('rate', 0):.0f}%")

        overall = stats.get("overall", {})
        print(f"  总计: {overall.get('success', 0)}/{overall.get('total', 0)} = {overall.get('rate', 0):.0f}%")

        all_stats.append(stats)

    # 计算 Brain Gain
    if len(all_stats) >= 2:
        print(f"\n{'=' * 50}")
        print("  Brain Gain Analysis")
        print(f"{'=' * 50}")

        random_stats = all_stats[0]
        brain_stats = all_stats[1]

        for level in [1, 2, 3]:
            rand_rate = random_stats.get(f"L{level}", {}).get("rate", 0)
            brain_rate = brain_stats.get(f"L{level}", {}).get("rate", 0)
            diff = brain_rate - rand_rate
            gain = diff / max(rand_rate, 1) * 100
            arrow = "🟢" if gain > 0 else "🔴"
            print(f"  L{level}: 随机={rand_rate:.0f}% → 大脑={brain_rate:.0f}%  {arrow} Brain Gain={gain:+.0f}%")

        rand_overall = random_stats.get("overall", {}).get("rate", 0)
        brain_overall = brain_stats.get("overall", {}).get("rate", 0)
        total_diff = brain_overall - rand_overall
        total_gain = total_diff / max(rand_overall, 1) * 100
        print(f"\n  总计: 随机={rand_overall:.0f}% → 大脑={brain_overall:.0f}%   Brain Gain={total_gain:+.0f}%")

        # 结论
        print(f"\n  结论:", end=" ")
        l3_gain = brain_stats.get("L3", {}).get("rate", 0) - random_stats.get("L3", {}).get("rate", 0)
        if l3_gain > 20:
            print("🟢 大脑对长链决策有显著价值 → 继续建设 v2")
        elif l3_gain > 5:
            print("🟡 大脑有一定价值, 但效果有限 → 需要改进设计")
        else:
            print("🔴 大脑没有显著价值 → 考虑纯 Qwen ReAct Agent 方向")

    return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brain Necessity Benchmark")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="不执行真实命令 (默认: True)")
    parser.add_argument("--real", action="store_true",
                       help="执行真实命令 (等效于 --no-dry-run)")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式 (只跑 6 个代表性任务)")
    args = parser.parse_args()

    dry_run = not args.real if args.real else args.dry_run
    run_experiment(dry_run=dry_run, quick=args.quick)
