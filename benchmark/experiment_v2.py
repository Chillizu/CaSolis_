"""Brain Necessity Benchmark — 实验运行器

核心比较:
  实验组 (Brain): 关键词匹配选意图 + 确定性参数
  对照组 (Random): 随机选意图 + 确定性参数
  
不依赖 LLM。参数从任务描述中提取。
目标是验证: 意图选择本身是否有价值 (Brain Gain)
"""

from __future__ import annotations

import time
import json
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.tasks import (
    Task, ALL_TASKS, TASKS_BY_LEVEL, validate_result
)
from benchmark.template_engine import TemplateEngine, ExecResult
from benchmark.param_extractor import ParameterExtractor


# ── Trained Classifier ───────────────────────────────────

class TrainedClassifier:
    """Sentence Transformer + Linear(384,8) 意图分类器"""
    
    def __init__(self, checkpoint: str = "checkpoints/intent_classifier/best_head.pt"):
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer
        
        self.INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE"]
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        
        # MLP(384 → 128 → 8) + LayerNorm + Dropout
        class IntentHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(384),
                    nn.Linear(384, 128),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 11),
                )
            def forward(self, x):
                return self.net(x)
        
        self.head = IntentHead()
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.head.load_state_dict(sd)
        self.head.eval()

    def predict(self, state_text: str) -> str:
        import torch
        emb = self.encoder.encode(state_text, convert_to_tensor=True, show_progress_bar=False)
        with torch.no_grad():
            logits = self.head(emb)
            pred = logits.argmax().item()
        return self.INTENTS[pred]


# ── 策略 ─────────────────────────────────────────────────

class Strategy:
    name: str = ""
    desc: str = ""

    def choose_step(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        raise NotImplementedError


class BrainStrategy(Strategy):
    """实验组: 关键词匹配意图 (模拟训练好的分类器)"""
    name = "brain-keyword"
    desc = "关键词匹配选意图"

    INTENT_KEYWORDS = {
        "READ":   ["read", "cat", "hostname", "file", "content", "看", "读", "查看"],
        "LIST":   ["list", "ls", "dir", "directory", "列", "目录"],
        "SEARCH": ["search", "grep", "find", "搜", "搜索", "查", "包含"],
        "INFO":   ["info", "cpu", "memory", "mem", "disk", "uptime", "uname", "whoami",
                   "system", "系统", "信息", "型号", "内存", "磁盘", "主机名"],
        "COUNT":  ["count", "wc", "line", "行", "多少", "统计", "数"],
        "INSPECT":["inspect", "check", "which", "检查", "有没有", "安装"],
        "EXPLORE":["explore", "探索", "发现", "找出"],
        "HELP":   ["help", "man", "--help"],
    }

    def _score_intents(self, text: str) -> list[tuple[str, int]]:
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] = scores.get(intent, 0) + 1
        return sorted(scores.items(), key=lambda x: -x[1])

    def __init__(self):
        self.param_extractor = ParameterExtractor()

    def choose_step(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        hint = task.hints[min(step, len(task.hints) - 1)]
        scored = self._score_intents(hint)

        if not scored:
            scored = self._score_intents(task.description)

        intent = scored[0][0] if scored else "INFO"

        params = self.param_extractor.extract(intent, hint)
        task_params = task.params.copy() if hasattr(task.params, 'copy') else {}
        for k in ["path", "pattern", "cmd"]:
            if k in task_params and k not in params:
                params[k] = task_params[k]

        return intent, params


class TrainedBrainStrategy(Strategy):
    """实验组 v2: 真实训练的 Sentence Transformer 分类器"""
    name = "brain-trained"
    desc = "MiniLM 分类器选意图 (65% acc)"

    def _build_state_text(self, task: Task, step: int, state: dict) -> str:
        # 使用 task.description 作为上步描述 (与训练数据格式一致)
        desc = task.description
        history = state.get("summary", "")[:200] if state.get("summary") and step > 0 else "无"
        path_raw = task.params.get('path', '')
        # 使用相对路径 (与训练数据格式一致)
        known_file = path_raw.lstrip('/') if path_raw else '未知'
        return f"当前目录: / 已知文件: {known_file} 上步: {desc} 历史: {history}"

    def __init__(self):
        self.classifier = TrainedClassifier()
        self.param_extractor = ParameterExtractor()

    def choose_step(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        state_text = self._build_state_text(task, step, state)
        intent = self.classifier.predict(state_text)

        # 使用参数提取器从任务描述的 hints 中提取参数
        hint = task.hints[min(step, len(task.hints) - 1)] if task.hints else task.description
        params = self.param_extractor.extract(intent, hint)

        # 多步上下文: 如果需要 path 但 hint 没提, 从 task.params 继承
        task_params = task.params.copy() if hasattr(task.params, 'copy') else {}
        for k in ["path", "pattern", "cmd"]:
            if k in task_params and k not in params:
                params[k] = task_params[k]

        return intent, params


class RandomStrategy(Strategy):
    """对照组: 随机选意图"""
    name = "random"
    desc = "随机选意图 + 参数提取器"

    def __init__(self):
        self.param_extractor = ParameterExtractor()

    def choose_step(self, task: Task, step: int, state: dict) -> tuple[str, dict]:
        intents = ["READ", "LIST", "SEARCH", "INFO", "COUNT", "INSPECT", "EXPLORE", "HELP"]
        intent = random.choice(intents)

        hint = task.hints[min(step, len(task.hints) - 1)] if task.hints else task.description
        params = self.param_extractor.extract(intent, hint)
        task_params = task.params.copy() if hasattr(task.params, 'copy') else {}
        for k in ["path", "pattern", "cmd"]:
            if k in task_params and k not in params:
                params[k] = task_params[k]

        return intent, params


# ── 实验运行器 ─────────────────────────────────────────────────

@dataclass
class StepResult:
    step: int
    intent: str
    params: dict
    result: ExecResult


@dataclass
class TaskResult:
    task_id: str
    level: int
    description: str
    steps: list[StepResult]
    success: bool
    validation_msg: str


class ExperimentRunner:

    def __init__(self, strategy: Strategy, real: bool = True):
        self.strategy = strategy
        self.engine = TemplateEngine(dry_run=not real)
        self.results: list[TaskResult] = []

    def run_task(self, task: Task) -> TaskResult:
        state: dict = {
            "summary": task.description,
            "cwd": "/",
            "steps_done": 0,
        }
        steps: list[StepResult] = []

        n_steps = max(len(task.hints), len(task.expected_intents))

        for step in range(n_steps):
            intent, params = self.strategy.choose_step(task, step, state)
            result = self.engine.execute(intent, params)

            steps.append(StepResult(
                step=step,
                intent=intent,
                params=params,
                result=result,
            ))

            # 更新状态 (自然语言摘要, 与训练数据格式一致)
            output = (result.stdout or result.stderr or "")[:200]
            # 根据意图生成结构化摘要
            if intent == "COUNT":
                n = len(output.strip().splitlines())
                state["summary"] = f"上一步: {intent} → 统计结果: 约 {max(n,1)} 行"
            elif intent == "SEARCH":
                n = len([l for l in output.strip().splitlines() if l.strip()])
                state["summary"] = f"上一步: {intent} → 找到 {max(n,1)} 行匹配"
            elif intent == "READ":
                state["summary"] = f"上一步: {intent} → 文件内容已读取, 共 {len(output.strip().splitlines())} 行"
            elif intent == "LIST":
                n = len([l for l in output.strip().splitlines() if l.strip()])
                state["summary"] = f"上一步: {intent} → 目录列表已获取, {max(n,1)} 项"
            elif intent == "INFO":
                state["summary"] = f"上一步: {intent} → 系统信息已获取"
            elif intent == "INSPECT":
                exists = "已安装" if output else "未找到"
                state["summary"] = f"上一步: {intent} → 命令{exists}"
            elif intent == "HELP":
                state["summary"] = f"上一步: {intent} → 帮助信息已显示"
            elif intent == "EXPLORE":
                state["summary"] = f"上一步: {intent} → 已浏览目录"
            else:
                state["summary"] = f"上一步: {intent} → 操作完成"
            state["steps_done"] = step + 1

        # 验证
        outputs = [s.result.stdout or s.result.stderr for s in steps]
        success, msg = validate_result(task, outputs)

        return TaskResult(
            task_id=task.task_id,
            level=task.level,
            description=task.description,
            steps=steps,
            success=success,
            validation_msg=msg,
        )

    def run_all(self, tasks: list[Task]) -> dict:
        self.results = []
        for i, task in enumerate(tasks, 1):
            short = task.description[:35]
            print(f"  [{i}/{len(tasks)}] {task.task_id:20s} {short}...", end=" ", flush=True)
            result = self.run_task(task)
            self.results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status}")
            time.sleep(0.1)

        return self._stats()

    def _stats(self) -> dict:
        stats = {}
        for level in [1, 2, 3]:
            tasks = [r for r in self.results if r.level == level]
            ok = sum(1 for r in tasks if r.success)
            total = len(tasks)
            stats[f"L{level}"] = {
                "total": total,
                "success": ok,
                "rate": (ok / total * 100) if total else 0,
            }
        all_r = self.results
        ok = sum(1 for r in all_r if r.success)
        stats["overall"] = {
            "total": len(all_r),
            "success": ok,
            "rate": (ok / len(all_r) * 100) if all_r else 0,
        }
        stats["strategy"] = self.strategy.name
        return stats


# ── 入口 ─────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Brain Necessity Benchmark")
    print("  验证: 意图选择是否产生价值")
    print("=" * 55)

    strategies: list[Strategy] = [
        RandomStrategy(),
        BrainStrategy(),
    ]

    # 如果训练好的分类器存在, 加入
    ckpt = "checkpoints/intent_classifier/best_head.pt"
    if os.path.exists(ckpt):
        strategies.append(TrainedBrainStrategy())
    else:
        print("  (训练好的分类器不存在, 跳过 TrainedBrain)")

    # 是否快速模式
    quick = "--quick" in sys.argv
    if quick:
        tasks = [t for t in ALL_TASKS if t.task_id in [
            "L1_hostname", "L1_cpu_info", "L1_passwd_count",
            "L2_search_count", "L2_cpu_mem",
            "L3_status_report",
        ]]
    else:
        tasks = ALL_TASKS

    real = "--dry-run" not in sys.argv
    mode = "真实" if real else "DRY RUN"
    print(f"\n模式: {mode} | 任务: {len(tasks)} 个 | 快速: {quick}\n")

    all_stats = []

    for strategy in strategies:
        print(f"\n── {strategy.desc} ({strategy.name}) ──")
        runner = ExperimentRunner(strategy, real=real)
        stats = runner.run_all(tasks)
        all_stats.append(stats)

        for level in [1, 2, 3]:
            s = stats.get(f"L{level}", {})
            bar = "█" * int(s.get("rate", 0) / 5) + "░" * (20 - int(s.get("rate", 0) / 5))
            print(f"    L{level}: {s.get('success', 0):2d}/{s.get('total', 0):2d}  {bar} {s.get('rate', 0):5.1f}%")
        o = stats.get("overall", {})
        print(f"    总计: {o.get('success', 0):2d}/{o.get('total', 0):2d}  成功率 {o.get('rate', 0):.1f}%")

    # Brain Gain
    if len(all_stats) >= 2:
        print(f"\n{'=' * 55}")
        print("  Brain Gain Analysis (vs Random)")
        print(f"{'=' * 55}")

        rand = all_stats[0]

        for i, stats in enumerate(all_stats[1:], 1):
            name = stats.get("strategy", f"策略{i}")
            print(f"\n  ── {name} ──")
            for level in [1, 2, 3]:
                r = rand.get(f"L{level}", {}).get("rate", 0)
                b = stats.get(f"L{level}", {}).get("rate", 0)
                gain = b - r
                pct = (gain / max(r, 1)) * 100
                icon = "🟢" if gain > 20 else ("🟡" if gain > 5 else "🔴")
                print(f"    L{level}: 随机={r:.0f}% → {name}={b:.0f}%  {icon} Δ={gain:+.0f}pp  ({pct:+.0f}%)")

            ro = rand.get("overall", {}).get("rate", 0)
            bo = stats.get("overall", {}).get("rate", 0)
            total_gain = bo - ro
            print(f"    总计: 随机={ro:.0f}% → {name}={bo:.0f}%   Δ={total_gain:+.0f}pp")

        # 最终结论
        print(f"\n  {'─' * 45}")
        best = max(all_stats[1:], key=lambda s: s.get('overall',{}).get('rate',0))
        best_name = best.get('strategy', '?')
        best_gain = best.get('overall',{}).get('rate',0) - rand.get('overall',{}).get('rate',0)
        if best_gain > 20:
            print(f"  结论: 🟢 {best_name} 显著优于随机 → 继续建设")
        elif best_gain > 5:
            print(f"  结论: 🟡 {best_name} 略优于随机, 但提升有限")
        else:
            print(f"  结论: 🔴 所有策略未体现显著价值 → 考虑纯手脚路线")


if __name__ == "__main__":
    main()
