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

    def extract(self, intent: str, text: str,
                 workbench: Any = None,
                 known_files: set[str] | None = None) -> dict[str, Any]:
        """
        P8.1: 从描述文本中提取意图参数, 支持工作栏事实推断
        
        Args:
            intent: 意图名称
            text: 描述文本
            workbench: 可选, 用于从已发现事实推断参数
            known_files: 可选, 已探索过的文件路径集合
        """
        intent_rules = self.rules["intents"].get(intent)
        if not intent_rules:
            return {}

        params = {}
        for rule in intent_rules["rules"]:
            param_name = rule["param"]
            if param_name in params:
                continue

            # 1. 尝试模式匹配
            patterns = rule.get("patterns", [])
            for i in range(0, len(patterns), 2):
                pattern = patterns[i]
                if not pattern:
                    continue
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    value = m.group(1) if m.lastindex else m.group(0)
                    if param_name in ("cmd",) and len(value.split()) > 1:
                        value = value.split()[0]
                    params[param_name] = value
                    break

            # 2. 尝试 keyword_map
            if param_name not in params and "keyword_map" in intent_rules:
                sorted_kws = sorted(intent_rules["keyword_map"].items(), key=lambda x: -len(x[0]))
                for kw, value in sorted_kws:
                    if kw.lower() in text.lower():
                        params[param_name] = value
                        break

            # 3. P8.1: 从工作栏事实推断 (替代硬编码默认值)
            if param_name not in params and workbench:
                params[param_name] = self._infer_from_facts(
                    param_name, intent, workbench, known_files
                )

            # 4. 使用默认值 (仅当工作栏也没推断出来)
            if param_name not in params and "default" in rule:
                params[param_name] = rule["default"]

        return params

    def _infer_from_facts(self, param_name: str, intent: str,
                          workbench: Any,
                          known_files: set[str] | None = None) -> str | None:
        """P8.1: 从工作栏事实推断参数, 返回推断值或 None"""
        facts = getattr(workbench, 'facts', {})
        if not facts:
            return None

        if param_name == "path":
            # 优先: 从已知目录选未探索的文件
            known_dirs = {k: v['value'] for k, v in facts.items()
                          if k.startswith('dir_')}
            for k, v in known_dirs.items():
                items = v.split(',')
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                    candidate = f"/{k[4:]}/{item}" if k != 'dir_root' else f"/{item}"
                    if known_files and candidate not in known_files:
                        return candidate
            # 次优: 用已知文件
            for f_k in ('etchosts_hosts', 'hostname', 'kernel', 'os_pretty_name'):
                if f_k in facts:
                    return f"/etc/{f_k}"
            # 兜底: 用已有事实的路径
            for fk in facts:
                if fk.startswith('dir_'):
                    path = fk.replace('dir_', '/').replace('_', '/')
                    return path
            return None

        if param_name == "pattern":
            # 从事实提取搜索关键词
            for key in ('hostname', 'kernel', 'os_name', 'os_pretty_name', 'users'):
                if key in facts:
                    val = str(facts[key]['value'])[:20]
                    if len(val) >= 3:
                        return val.split()[0]
            return None

        if param_name == "cmd":
            # 从事实推断应检查的命令
            cmds = []
            if 'kernel' in facts:
                cmds.append('python3')
            if 'os_name' in facts and 'debian' in facts['os_name']['value'].lower():
                cmds.append('bash')
            cmds.append('ls')
            return cmds[0] if cmds else None

        return None

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
