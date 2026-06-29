"""
SelfModel: 自我意识统计追踪器

追踪每类意图的成功率、高价值产出、自我认识特征。
用于给 qwen 自省 prompt 提供「你是怎样的 Agent」的数据。
"""

import json
import os
from collections import defaultdict
from typing import Any, Optional


class SelfModel:
    """
    轻量自我模型: 追踪行为统计, 生成自我描述。

    数据保留在内存 + 可持久化到 JSON。
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or "data/persistent/self_model.json"
        self.step: int = 0
        self.total_steps: int = 0
        self.total_success: int = 0
        self.total_reward: float = 0.0

        # per-intent 统计
        self.intent_stats: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "success": 0, "reward": 0.0,
                     "last_step": 0, "output_bytes": [], "facts_added": []}
        )
        # 高光时刻 (top-N 高奖励)
        self.highlights: list[dict] = []
        self.max_highlights: int = 10

        # 创作记录
        self.creations: list[dict] = []
        self.max_creations: int = 20

        # 认识更新步数
        self.last_self_reflect_step: int = 0

        self._load()

    # ── 记录 ──

    def record(self, intent: str, success: bool, reward: float,
               step: int, output_len: int = 0, n_facts: int = 0,
               path: str = "", content_len: int = 0):
        """每步后调用"""
        self.step = step
        self.total_steps += 1
        if success:
            self.total_success += 1
        self.total_reward += reward

        s = self.intent_stats[intent]
        s["calls"] += 1
        s["success"] += 1 if success else 0
        s["reward"] += reward
        s["last_step"] = step
        if output_len > 10:
            s["output_bytes"].append(output_len)
        if n_facts > 0:
            s["facts_added"].append(n_facts)

        # 高光 (高奖励 + 成功)
        if success and reward > 0.5:
            self.highlights.append({
                "step": step, "intent": intent, "reward": round(reward, 2),
                "output_len": output_len, "n_facts": n_facts,
            })
            self.highlights = self.highlights[-self.max_highlights:]

        # 创作记录
        if intent == "CREATE" and success and path:
            self.creations.append({
                "step": step, "path": path, "size": content_len or output_len,
            })
            self.creations = self.creations[-self.max_creations:]

    # ── 查询 ──

    def success_rate(self, intent: Optional[str] = None) -> float:
        if intent:
            s = self.intent_stats.get(intent, {})
            calls = s.get("calls", 0)
            return s.get("success", 0) / calls if calls > 0 else 0.0
        return self.total_success / max(self.total_steps, 1)

    def best_intents(self, top_k: int = 3) -> list[str]:
        scored = [(name, s["success"] / max(s["calls"], 1))
                  for name, s in self.intent_stats.items()
                  if s["calls"] >= 3]
        scored.sort(key=lambda x: -x[1])
        return [n for n, _ in scored[:top_k]]

    def worst_intents(self, top_k: int = 2) -> list[str]:
        best_set = set(self.best_intents(top_k))
        scored = [(name, s["success"] / max(s["calls"], 1))
                  for name, s in self.intent_stats.items()
                  if s["calls"] >= 3 and name not in best_set]
        scored.sort(key=lambda x: x[1])
        return [n for n, _ in scored[:top_k]]

    def avg_output_len(self, intent: str) -> float:
        s = self.intent_stats.get(intent, {})
        obs = s.get("output_bytes", [])
        return sum(obs) / len(obs) if obs else 0.0

    def total_facts_added(self) -> int:
        return sum(sum(s.get("facts_added", [])) for s in self.intent_stats.values())

    # ── 自我描述 (给 qwen 用) ──

    def build_self_description(self) -> str:
        """生成一段自我描述, 用作 LLM prompt 的上下文"""
        lines = []
        lines.append(f"- 共执行 {self.total_steps} 步, 成功率 {self.success_rate()*100:.0f}%")
        lines.append(f"- 总奖励 {self.total_reward:.1f}")
        lines.append(f"- 擅长: {', '.join(self.best_intents(3))}")
        worst = self.worst_intents(2)
        if worst:
            lines.append(f"- 不擅长: {', '.join(worst)}")
        lines.append(f"- 共挖掘事实 {self.total_facts_added()} 条")
        if self.creations:
            lines.append(f"- 创作 {len(self.creations)} 次, "
                         f"总大小 {sum(c['size'] for c in self.creations)}B")
        if self.highlights:
            last_hl = self.highlights[-1]
            lines.append(f"- 最近高光: 步{last_hl['step']} {last_hl['intent']} "
                         f"(奖励 {last_hl['reward']})")
        return "\n".join(lines)

    def build_self_prompt(self) -> str:
        """自省 prompt: 让 LLM 思考「我想做什么」"""
        desc = self.build_self_description()
        return (
            "You are an autonomous Linux agent. Here is what you know about yourself:\n"
            f"{desc}\n\n"
            "Based on this self-knowledge, what do you feel like doing next?\n"
            "- What are you good at and want to practice?\n"
            "- What haven't you done in a while?\n"
            "- Is there something you've been curious about?\n\n"
            "Output ONE short sentence describing your intention. "
            "Be specific (e.g. 'I want to explore network commands because I "
            "haven't tried them yet' or 'I want to write a script that "
            "summarizes system info').\n"
            "Intention:"
        )

    # ── 持久化 ──

    def _load(self):
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.step = data.get("step", 0)
                self.total_steps = data.get("total_steps", 0)
                self.total_success = data.get("total_success", 0)
                self.total_reward = data.get("total_reward", 0.0)
                self.intent_stats.update(data.get("intent_stats", {}))
                self.highlights = data.get("highlights", [])
                self.creations = data.get("creations", [])
            except Exception:
                pass

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "step": self.step,
            "total_steps": self.total_steps,
            "total_success": self.total_success,
            "total_reward": self.total_reward,
            "intent_stats": dict(self.intent_stats),
            "highlights": self.highlights,
            "creations": self.creations,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
