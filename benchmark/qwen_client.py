"""Qwen 参数推理客户端 — Ollama API 调用

功能:
  输入: 意图名 + 当前状态 → 输出: 参数 JSON
  不生成 bash 命令, 只推理参数值
"""

import json
import subprocess
import time
from typing import Any

# ── Prompt 模板 ─────────────────────────────────────────────────

PARAM_PROMPTS = {
    "READ": """你是系统的参数推理器。给定一个意图和当前系统状态，输出执行该意图所需的参数。

意图: READ — 读文件内容
参数: {{"path": "文件路径"}}

当前状态:
{cwd}
已访问目录: {visited}
已知文件: {known}
上步摘要: {summary}
历史: {history}

输出 JSON:""",

    "LIST": """意图: LIST — 列目录内容
参数: {{"path": "目录路径"}}

当前状态:
{cwd}
已访问目录: {visited}
已知文件: {known}

输出 JSON:""",

    "SEARCH": """意图: SEARCH — 在文件中搜索内容
参数: {{"pattern": "搜索模式", "path": "文件路径"}}

当前状态:
{cwd}
已访问目录: {visited}
已知文件: {known}
历史: {history}

输出 JSON:""",

    "INFO": """意图: INFO — 获取系统信息 (cpu/mem/disk/uptime/whoami/uname)
参数: {{"target": "cpu|mem|disk|uptime|whoami|uname"}}

输出 JSON:""",

    "COUNT": """意图: COUNT — 统计行数/词数
参数: {{"path": "文件路径"}}

当前状态:
{cwd}
已知文件: {known}

输出 JSON:""",

    "INSPECT": """意图: INSPECT — 检查命令是否存在
参数: {{"cmd": "命令名"}}

当前状态:
{cwd}
历史: {history}

输出 JSON:""",

    "EXPLORE": """意图: EXPLORE — 探索新路径或新命令
参数: {{"target": "要探索的路径或命令"}}

当前状态:
{cwd}
已访问目录: {visited}
已知文件: {known}

输出 JSON:""",

    "HELP": """意图: HELP — 查看命令帮助
参数: {{"cmd": "命令名"}}

当前状态:
{cwd}
历史: {history}

输出 JSON:""",
}

# ── JSON Schema 校验 ────────────────────────────────────────────

PARAM_SCHEMAS = {
    "READ": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
    "LIST": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
    "SEARCH": {"type": "object", "required": ["pattern", "path"], "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}},
    "INFO": {"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
    "COUNT": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
    "INSPECT": {"type": "object", "required": ["cmd"], "properties": {"cmd": {"type": "string"}}},
    "EXPLORE": {"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
    "HELP": {"type": "object", "required": ["cmd"], "properties": {"cmd": {"type": "string"}}},
}

# ── 参数验证器 ─────────────────────────────────────────────────

def validate_params(intent: str, params: dict) -> tuple[bool, str]:
    """验证参数是否合法"""
    try:
        import jsonschema
        schema = PARAM_SCHEMAS.get(intent)
        if schema:
            jsonschema.validate(params, schema)
    except ImportError:
        # 用简单规则验证
        pass
    except jsonschema.ValidationError as e:
        return False, str(e)

    # 安全验证
    for key, value in params.items():
        if not isinstance(value, str):
            continue
        # 禁止 shell 特殊字符
        for char in [';', '|', '`', '$', '>', '<', '&', '!']:
            if char in value:
                return False, f"参数 {key} 包含非法字符: {char}"
        # 路径安全检查
        if key in ('path', 'target'):
            if not value.startswith('/'):
                value = '/' + value
            # 只允许安全路径
            safe_prefixes = ('/proc/', '/etc/', '/tmp/', '/usr/', '/var/', '/sys/', '/')
            if not any(value.startswith(p) for p in safe_prefixes):
                return False, f"路径不安全: {value}"

    return True, ""


# ── Qwen 推理客户端 ────────────────────────────────────────────

class QwenReasoner:
    """通过 Ollama 调用 Qwen 2.5 1.5B 进行参数推理"""

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self._stats = {"calls": 0, "total_ms": 0, "errors": 0}

    def query(self, prompt: str) -> str | None:
        """调用 Ollama API"""
        import requests
        try:
            start = time.monotonic()
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 128, "temperature": 0.3}
                },
                timeout=15,
            )
            elapsed = (time.monotonic() - start) * 1000
            self._stats["calls"] += 1
            self._stats["total_ms"] += elapsed

            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                self._stats["errors"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            return None

    def reason(self, intent: str, state: dict) -> dict | None:
        """给定意图和状态, 推理参数"""
        prompt_template = PARAM_PROMPTS.get(intent)
        if not prompt_template:
            return {}

        prompt = prompt_template.format(
            cwd=state.get("cwd", "/"),
            visited=state.get("visited", []),
            known=state.get("known", []),
            summary=state.get("summary", ""),
            history=state.get("history", []),
        )

        raw = self.query(prompt)
        if not raw:
            return None

        # 提取 JSON
        try:
            # 尝试直接解析
            params = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON 块
            import re
            matches = re.findall(r'\{[^}]+\}', raw, re.DOTALL)
            if not matches:
                return None
            try:
                params = json.loads(matches[0])
            except json.JSONDecodeError:
                return None

        # 验证
        ok, msg = validate_params(intent, params)
        if not ok:
            return None

        return params

    def stats(self) -> dict:
        s = self._stats
        avg = s["total_ms"] / s["calls"] if s["calls"] else 0
        return {"calls": s["calls"], "avg_ms": f"{avg:.0f}", "errors": s["errors"]}


# ── 快速测试 ─────────────────────────────────────────────────

def test():
    """测试 Qwen 推理"""
    import os
    import json

    reasoner = QwenReasoner()

    # 测试连接
    print("测试 Ollama 连接...")
    try:
        import requests
        resp = requests.get(f"{reasoner.base_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        print(f"Ollama 可用, 模型列表:")
        for m in models:
            print(f"  - {m['name']}")
    except Exception as e:
        print(f"⚠️ Ollama 不可用: {e}")
        print("请确保 Ollama 正在运行: ollama serve")
        return

    # 测试参数推理
    test_state = {
        "cwd": "/home/user",
        "visited": ["/", "/etc"],
        "known": ["/etc/passwd", "/etc/hostname"],
        "summary": "文件30行, 包含 'root'",
        "history": ["READ /etc/passwd", "SEARCH root"],
    }

    for intent in ["READ", "SEARCH", "INFO"]:
        print(f"\n意图: {intent}")
        params = reasoner.reason(intent, test_state)
        if params:
            print(f"  参数: {json.dumps(params)}")
        else:
            print(f"  ❌ 推理失败")


if __name__ == "__main__":
    test()
