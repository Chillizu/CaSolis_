"""
保姆 (Nanny) V3 — 想法向量 → 可执行命令

使用训练好的 ConductorHead:
  - 编码 state_text → (thought, logits)
  - logits → argmax → intent
  - thought → 保姆参考信息
"""

import torch
import torch.nn.functional as F
from typing import Any, Optional

from benchmark.template_engine import TemplateEngine, ExecResult
from benchmark.param_extractor import ParameterExtractor


INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE", "LS_TMP", "ARCH_INFO", "CUSTOM"]


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
        self.param_extractor = ParameterExtractor()

    def think(self, state_text: str) -> tuple:
        """编码状态文本 → (想法向量, 分类logits)"""
        emb = self.conductor.encoder.encode(
            state_text, convert_to_tensor=True
        ).clone().to(self.conductor.device)
        with torch.no_grad():
            thought, logits = self.conductor.head(emb.unsqueeze(0))
        return thought.squeeze(0), logits.squeeze(0)

    def translate(
        self, thought: torch.Tensor, logits: torch.Tensor,
        threshold: float = 0.3, state_text: str = ""
    ) -> tuple[str, dict, float]:
        """
        想法 → 意图 + 参数

        Args:
            thought: (16,) tensor
            logits: (11,) tensor
            threshold: 置信度阈值, 低于此值走 CUSTOM
            state_text: 当前状态文本 (用于参数提取)
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

        # 参数推断 (基于想法向量 + 状态文本规则提取)
        params = self._vector_to_params(thought, best_intent, state_text)

        # P1: 从想法向量计算探索深度
        thought_norm = float(thought.norm().item())
        depth = min(3, max(1, int(thought_norm * 0.6 + 0.5)))
        params["depth"] = depth

        self.stats["total"] += 1
        self.stats["by_intent"][best_intent] = self.stats["by_intent"].get(best_intent, 0) + 1

        return best_intent, params, best_prob

    def _vector_to_params(self, thought: torch.Tensor, intent: str, state_text: str = "") -> dict:
        """从想法向量 + 状态文本推断参数"""
        params = {}

        # 1. 先用规则提取器从状态文本中提取参数
        if state_text:
            rule_params = self.param_extractor.extract(intent, state_text)
            params.update(rule_params)

        # 2. 只有确实没提取到时, 才用默认兜底
        if intent == "READ" and "path" not in params:
            params["path"] = "/etc/hosts"
        elif intent == "SEARCH":
            if "pattern" not in params:
                params["pattern"] = "root"
            if "path" not in params:
                params["path"] = "/etc/passwd"
        elif intent == "COUNT" and "path" not in params:
            params["path"] = "/etc/passwd"
        elif intent == "INSPECT" and "cmd" not in params:
            params["cmd"] = "python3"
        elif intent == "HELP" and "cmd" not in params:
            params["cmd"] = "ls"
        elif intent == "READ_ETC" and "path" not in params:
            params["path"] = "/etc/passwd"
        elif intent == "DISK_USAGE" and "path" not in params:
            params["path"] = "/"

        # 3. 想法向量微调: 高激活维度影响参数选择
        vec = thought.detach().cpu().numpy()
        if intent == "SEARCH" and abs(vec[0]) > 0.8:
            # 如果想法向量 dim[0] 高激活, 换搜索关键词
            alt_patterns = ["bash", "nobody", "daemon", "www-data"]
            idx = int(abs(vec[0]) * 3) % len(alt_patterns)
            params["pattern"] = alt_patterns[idx]

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
