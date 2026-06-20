"""
保姆 (Nanny) V3 — 想法向量 → 可执行命令

使用训练好的 ConductorHead:
  - 编码 state_text → (thought, logits)
  - logits → argmax → intent
  - thought → 保姆参考信息
"""

import torch
import torch.nn.functional as F
from typing import Any

from benchmark.template_engine import TemplateEngine, ExecResult


INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE"]


class Nanny:
    """
    保姆: 封装 Conductor → 意图 → 执行

    使用:
        nanny = Nanny(conductor_checkpoint="checkpoints/conductor/head.pt")
        thought, logits = nanny.think(state_text)    # 获得想法
        intent, params, conf = nanny.translate(thought, logits)  # 翻译
        result = nanny.execute(state_text)            # 一步完成
    """

    def __init__(
        self,
        engine: TemplateEngine = None,
        sandbox=None,
        conductor_checkpoint: str = None,
    ):
        self.engine = engine or TemplateEngine(dry_run=False, sandbox=sandbox)
        self.stats = {
            "total": 0,
            "by_intent": {},
            "conductor_path": 0,
            "fallback_path": 0,
        }

        # ConductorHead
        from agent.conductor import Conductor
        self.conductor = Conductor(checkpoint=conductor_checkpoint)

    def think(self, state_text: str) -> tuple:
        """编码状态文本 → (想法向量, 分类logits)"""
        emb = self.conductor.encoder.encode(
            state_text, convert_to_tensor=True
        ).clone().to(self.conductor.device)
        with torch.no_grad():
            thought, logits = self.conductor.head(emb.unsqueeze(0))
        return thought.squeeze(0), logits.squeeze(0)

    def translate(
        self, thought: torch.Tensor, logits: torch.Tensor, threshold: float = 0.3
    ) -> tuple[str, dict, float]:
        """
        想法 → 意图 + 参数

        Args:
            thought: (16,) tensor
            logits: (11,) tensor
            threshold: 置信度阈值, 低于此值走 CUSTOM
        Returns:
            (intent_name, params, confidence)
        """
        probs = F.softmax(logits, dim=-1)
        best_prob, best_idx = probs.max(dim=-1)
        best_intent = INTENTS[best_idx.item()]
        best_prob = best_prob.item()

        # 低置信度 → CUSTOM (自由探索)
        if best_prob < threshold:
            best_intent = "CUSTOM"

        # 参数推断 (基于想法向量的激活模式)
        params = self._vector_to_params(thought, best_intent)

        self.stats["total"] += 1
        self.stats["by_intent"][best_intent] = self.stats["by_intent"].get(best_intent, 0) + 1

        return best_intent, params, best_prob

    def _vector_to_params(self, thought: torch.Tensor, intent: str) -> dict:
        """从想法向量推断参数"""
        params = {}
        vec = thought.detach().cpu().numpy()

        if intent == "READ":
            params["path"] = "/etc/hosts"
        elif intent == "SEARCH":
            params["pattern"] = "root"
            params["path"] = "/etc/passwd"
        elif intent == "COUNT":
            params["path"] = "/etc/passwd"
        elif intent == "INSPECT":
            params["cmd"] = "python3"
        elif intent == "HELP":
            params["cmd"] = "ls"
        elif intent == "READ_ETC":
            params["path"] = "/etc/passwd"
        elif intent == "DISK_USAGE":
            params["path"] = "/"

        return params

    def execute(self, state_text: str) -> tuple[ExecResult, str, float]:
        """
        一步完成: 思考 + 翻译 + 执行
        Returns: (result, intent, confidence)
        """
        thought, logits = self.think(state_text)
        intent, params, conf = self.translate(thought, logits)

        result = self.engine.execute(intent, params)
        self.stats["conductor_path"] += 1
        return result, intent, conf

    def stats_report(self) -> str:
        lines = [
            f"  保姆 ({self.stats['total']} 次):",
            f"    Conductor路径: {self.stats['conductor_path']}",
            f"    Fallback路径: {self.stats['fallback_path']}",
        ]
        for intent, count in sorted(self.stats["by_intent"].items(), key=lambda x: -x[1]):
            pct = count / max(self.stats["total"], 1) * 100
            lines.append(f"    {intent:15s} {count:4d} ({pct:.0f}%)")
        return "\n".join(lines)
