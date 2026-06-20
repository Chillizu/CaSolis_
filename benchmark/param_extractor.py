"""
参数提取器 — 基于规则的确定性参数提取

从任务描述/提示中提取意图所需的具体参数。
不依赖 LLM，完全基于 config/param_rules.json 的规则。
"""

import json
import re
import os
from typing import Any


class ParameterExtractor:
    """从自然语言描述中提取参数"""

    def __init__(self, rules_path: str = "config/param_rules.json"):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)

    def extract(self, intent: str, text: str) -> dict[str, Any]:
        """从描述文本中提取指定意图的参数"""
        intent_rules = self.rules["intents"].get(intent)
        if not intent_rules:
            return {}

        params = {}
        for rule in intent_rules["rules"]:
            param_name = rule["param"]
            if param_name in params:
                continue  # 已提取

            # 1. 尝试模式匹配
            patterns = rule.get("patterns", [])
            for i in range(0, len(patterns), 2):
                pattern = patterns[i]
                if not pattern:
                    continue
                # 如果有相邻的下一个元素是 flags (空字符串 = 无 flag)
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    # 如果有捕获组, 用第一个; 否则用整个匹配
                    value = m.group(1) if m.lastindex else m.group(0)
                    # 特殊处理: single-word command names
                    if param_name in ("cmd",) and len(value.split()) > 1:
                        value = value.split()[0]
                    params[param_name] = value
                    break

            # 2. 尝试 keyword_map (按关键词长度降序, 最精确的优先匹配)
            if param_name not in params and "keyword_map" in intent_rules:
                # 按长度降序排列, 确保 "当前时间" 优先于 "时间"
                sorted_kws = sorted(intent_rules["keyword_map"].items(), key=lambda x: -len(x[0]))
                for kw, value in sorted_kws:
                    if kw.lower() in text.lower():
                        params[param_name] = value
                        break

            # 3. 使用默认值
            if param_name not in params and "default" in rule:
                params[param_name] = rule["default"]

        return params

    def extract_from_hints(
        self, intent: str, hints: list[str], step: int
    ) -> dict[str, Any]:
        """从任务提示列表中提取参数 (支持多步)"""
        if step < len(hints):
            text = " ".join(hints)
        else:
            text = hints[-1] if hints else ""

        params = self.extract(intent, text)

        # 多步上下文增强: 如果前一步有结果, 继承
        if step > 0:
            # 第一步可能是 SEARCH → 第二步 COUNT (继承 path)
            if intent in ("COUNT",) and "path" not in params:
                # 尝试从前一步的 hint 中找文件路径
                for h in hints[:step]:
                    m = re.search(r"(/[\w./-]+)", h)
                    if m:
                        params["path"] = m.group(1)
                        break

        return params
