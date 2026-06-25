"""
CreativeWriter — LLM 创作插件

基于 FactGraph 事实, 调用本地 Ollama 模型自由生成内容。
只参与「写什么」, 不参与「做什么/执行什么」。

P0: 独立可用 (Ollama + prompt + 4风格 + fallback)
P1: 接入 GoalGenerator CREATE mode
P2: 异步 + 质量评估

安全红线:
  - 不参与意图选择
  - 输出只作为 content 参数, 不直接执行
  - path 仍由 GoalGenerator 决定并受白名单约束
  - prompt 不传敏感信息
"""

import json
import os
import subprocess
import time
from typing import Optional


# ── Ollama 调用 ──

def _ollama_generate(prompt: str, model: str = "gemma4:e4b",
                     base_url: str = "http://localhost:11434",
                     timeout: float = 15.0,
                     max_tokens: int = 1024,
                     temperature: float = 0.5) -> Optional[str]:
    """调用 Ollama /api/generate, 超时返回 None"""
    try:
        import urllib.request
        import urllib.error

        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()

    except Exception:
        return None


def _ollama_health(base_url: str = "http://localhost:11434") -> bool:
    """检查 Ollama 是否运行且模型可用"""
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            # 检查 gemma4:e4b 或类似名字
            return any("gemma" in m.lower() for m in models)
    except Exception:
        return False


def _thermal_ok(threshold_c: float = 95.0) -> bool:
    """检查 CPU 温度是否安全 (读 /sys/class/thermal/)"""
    try:
        for entry in os.listdir("/sys/class/thermal/"):
            if entry.startswith("thermal_zone"):
                temp_path = f"/sys/class/thermal/{entry}/temp"
                if os.path.exists(temp_path):
                    with open(temp_path) as f:
                        temp_mc = int(f.read().strip())
                        temp_c = temp_mc / 1000.0
                        if temp_c < threshold_c:
                            return True
        return True  # 无传感器时默认 OK
    except Exception:
        return True


# ── Prompt 构建 ──

def _build_facts_section(workbench) -> str:
    """从 FactGraph 构建 facts 文本"""
    lines = []
    graph = getattr(workbench, 'graph', None)
    if graph and graph.nodes:
        nodes = sorted(graph.nodes.items(), key=lambda x: -x[1].step)
        for key, node in nodes[:30]:  # 最多 30 个
            val = node.value[:80]
            cat = node.category
            conf = node.confidence
            src = node.source_cmd[:30]
            lines.append(f"[{cat}] {key} = {val} (confidence={conf}, source={src})")
    elif hasattr(workbench, 'facts') and workbench.facts:
        for key, fact in sorted(workbench.facts.items(),
                                key=lambda x: -x[1].get("step", 0))[:30]:
            val = fact.get("value", "")[:80]
            cat = fact.get("category", "general")
            conf = fact.get("confidence", 0.5)
            src = fact.get("source_cmd", "")[:30]
            lines.append(f"[{cat}] {key} = {val} (confidence={conf}, source={src})")

    if not lines:
        return "(no facts available)"
    return "\n".join(lines)


def _build_relationships_section(workbench) -> str:
    """从 FactGraph 构建 relationships 文本"""
    lines = []
    graph = getattr(workbench, 'graph', None)
    if graph and graph.edges:
        for src, edges in graph.edges.items():
            for e in edges[:5]:  # 每节点最多 5 条边
                lines.append(f"{src} --{e['rel']}--> {e['to']}")
    if not lines:
        return "(no relationships)"
    return "\n".join(lines)


def _build_gaps_section(workbench) -> str:
    """从 FactGraph 构建 gaps 文本"""
    lines = []
    graph = getattr(workbench, 'graph', None)
    if graph and hasattr(graph, 'find_gaps'):
        gaps = graph.find_gaps()
        for src, missing, rel in gaps[:10]:
            lines.append(f"- {missing} ({rel}, required by {src})")
    if not lines:
        return "(no gaps)"
    return "\n".join(lines)


# ── 质量检查 ──

def _check_hallucination(text: str, facts_text: str) -> float:
    """检查生成文本中是否有幻觉 (未出现在 prompt 中的数值)
    返回 0~1 的分数 (0=纯幻觉, 1=完全基于事实)
    """
    # 提取事实中的所有具体值
    fact_values = set()
    for line in facts_text.split("\n"):
        if "=" in line:
            val = line.split("=", 1)[1].split("(")[0].strip().strip('"')
            if val and not val.startswith("(no"):
                fact_values.add(val.lower())

    # 提取生成文本中的所有具体值 (数字、路径、版本号等)
    import re
    text_values = set()
    for match in re.finditer(r'\b(\d+\.?\d*[KMG]?|[a-z]+\.[a-z]+\.[0-9]+|/[a-z/]+)\b',
                             text.lower()):
        text_values.add(match.group(1))

    if not text_values:
        return 1.0  # 没有具体数值, 无法判断

    # 检查有多少生成文本中的值在事实中出现过
    matched = sum(1 for v in text_values if any(v in fv or fv in v for fv in fact_values))
    return matched / len(text_values)


# ── CreativeWriter 类 ──

class CreativeWriter:
    """
    LLM 创作插件

    P0: 独立可用
      writer = CreativeWriter()
      result = writer.generate_content(workbench, style="report")

    P1: 接入 GoalGenerator
      goal_generator = GoalGenerator(creative_writer=writer)

    P2: 异步
      writer.enable_async()
    """

    def __init__(self,
                 model: str = "gemma4:e4b",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 15.0,
                 max_tokens: int = 1024,
                 temperature: float = 0.5,
                 thermal_threshold: float = 95.0,
                 enabled: bool = True):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thermal_threshold = thermal_threshold
        self.enabled = enabled

        # 统计
        self.stats = {
            "total_calls": 0,
            "llm_success": 0,
            "fallback": 0,
            "thermal_skip": 0,
            "timeout": 0,
        }

        # 加载 prompt 模板
        self._prompts = self._load_prompts()

        # P2: 异步
        self._async_enabled = False
        self._async_result: Optional[dict] = None

    def _load_prompts(self) -> dict:
        """加载 config/creative_prompts.yaml"""
        try:
            import yaml
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "config", "creative_prompts.yaml")
            if os.path.exists(path):
                with open(path) as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    def health_check(self) -> bool:
        """Ollama 模型可用?"""
        return _ollama_health(self.base_url)

    def is_thermal_ok(self) -> bool:
        """CPU 温度安全?"""
        return _thermal_ok(self.thermal_threshold)

    def build_prompt(self, workbench, style: str = "report") -> str:
        """构建 LLM prompt"""
        template = self._prompts.get(style, self._prompts.get("report", ""))
        facts_text = _build_facts_section(workbench)
        rels_text = _build_relationships_section(workbench)
        gaps_text = _build_gaps_section(workbench)

        prompt = template.replace("{facts_section}", facts_text)
        prompt = prompt.replace("{relationships_section}", rels_text)
        prompt = prompt.replace("{gaps_section}", gaps_text)

        # 缓存 facts_text 用于质量检查
        self._last_facts_text = facts_text

        return prompt

    def generate(self, prompt: str, timeout: Optional[float] = None) -> Optional[str]:
        """调用 Ollama 生成"""
        return _ollama_generate(
            prompt=prompt,
            model=self.model,
            base_url=self.base_url,
            timeout=timeout or self.timeout,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def generate_content(self, workbench, style: Optional[str] = None,
                         timeout: Optional[float] = None) -> dict:
        """
        主入口: 生成内容

        Returns:
          {"content": str, "path": str, "desc": str, "size": int, "source": "llm"|"fallback"}
        """
        self.stats["total_calls"] += 1

        # 1. P2 异步结果检查
        if self._async_enabled and self._async_result:
            result = self._async_result
            self._async_result = None
            if result and result["source"] == "llm":
                return result

        # 2. 安全检查
        if not self.enabled:
            return self._fallback(workbench, style, "disabled")

        if not self.health_check():
            return self._fallback(workbench, style, "ollama_unavailable")

        if not self.is_thermal_ok():
            self.stats["thermal_skip"] += 1
            return self._fallback(workbench, style, "thermal_throttle")

        # 3. 构建 prompt
        effective_style = style or "report"
        prompt = self.build_prompt(workbench, effective_style)

        # 4. 生成
        text = self.generate(prompt, timeout)

        if text is None:
            self.stats["timeout"] += 1
            return self._fallback(workbench, style, "timeout")

        # 5. 质量检查
        quality = _check_hallucination(text, self._last_facts_text)
        if quality < 0.3:
            # 严重幻觉, 降级
            return self._fallback(workbench, style, f"hallucination({quality:.2f})")

        # 6. 构建返回
        self.stats["llm_success"] += 1
        step = getattr(workbench, '_step_counter', 0) or 0
        desc_map = {
            "report": "LLM生成:系统报告",
            "analysis": "LLM生成:分析",
            "story": "LLM生成:叙事",
            "code": "LLM生成:Python脚本",
        }
        path_map = {
            "report": f"/tmp/llm_report_{step}.md",
            "analysis": f"/tmp/llm_analysis_{step}.md",
            "story": f"/tmp/llm_story_{step}.md",
            "code": f"/tmp/llm_script_{step}.py",
        }
        return {
            "content": text,
            "path": path_map.get(effective_style, f"/tmp/llm_output_{step}.md"),
            "desc": desc_map.get(effective_style, "LLM生成内容"),
            "size": len(text),
            "source": "llm",
        }

    def _fallback(self, workbench, style: Optional[str] = None,
                  reason: str = "") -> dict:
        """回退到 Workbench 模板生成"""
        self.stats["fallback"] += 1

        if not hasattr(workbench, 'build_write_content'):
            return {"content": "(fallback)", "path": "/tmp/fallback.txt",
                    "desc": "回退", "size": 0, "source": f"fallback:{reason}"}

        if style == "code":
            # code 风格: 用脚本生成
            if hasattr(workbench, 'generate_script'):
                result = workbench.generate_script()
                if result:
                    script, combo = result
                    step = getattr(workbench, '_step_counter', 0) or 0
                    return {"content": script, "path": f"/tmp/fallback_script_{step}.sh",
                            "desc": f"模板脚本({combo})", "size": len(script),
                            "source": f"fallback:{reason}"}

        # 默认 fallback: build_write_content
        ci = workbench.build_write_content()
        ci.setdefault("size", len(ci.get("content", "")))
        if ci.get("size", 0) > 0:
            ci["source"] = f"fallback:{reason}"
            return ci

        return {"content": "(fallback)", "path": "/tmp/fallback.txt",
                "desc": "回退", "size": 0, "source": f"fallback:{reason}"}

    # ── P2: 异步 ──

    def enable_async(self):
        """启用异步模式"""
        self._async_enabled = True

    def generate_async(self, workbench, style: str = "report"):
        """后台预生成 (P2)"""
        import threading
        def _run():
            self._async_result = self.generate_content(workbench, style)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── 统计 ──

    def stats_report(self) -> str:
        s = self.stats
        total = s["total_calls"]
        llm_pct = s["llm_success"] / max(total, 1) * 100
        fb_pct = s["fallback"] / max(total, 1) * 100
        return (
            f"  CreativeWriter ({total}次):\n"
            f"    LLM成功: {s['llm_success']} ({llm_pct:.0f}%)\n"
            f"    Fallback: {s['fallback']} ({fb_pct:.0f}%)\n"
            f"    温度跳过: {s['thermal_skip']}\n"
            f"    超时: {s['timeout']}"
        )

    def get_stats(self) -> dict:
        return dict(self.stats)
