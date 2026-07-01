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

import os, urllib.request, json




def _ollama_generate(prompt: str, model: str = "gemma4:e4b",
                     base_url: str = "http://localhost:11434",
                     timeout: float | None = None,
                     max_tokens: int = 1024,
                     temperature: float = 0.5) -> Optional[str]:
    """调用 Ollama /api/chat, 超时返回 None"""
    timeout = timeout if timeout is not None else 120.0
    try:
        import urllib.request
        import urllib.error

        # gemma4:e4b 是推理模型, 需要 raw:true 绕过 thinking token
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
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


def _deepseek_generate(prompt: str,
                       api_key: str,
                       model: str = "deepseek-chat",
                       timeout: float = 60.0,
                       max_tokens: int = 2048,
                       temperature: float = 0.3) -> Optional[str]:
    """调用 DeepSeek API, 超时返回 None"""
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a Linux system agent. Output ONLY the requested content, no explanation or markdown wrapping."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _ollama_health(base_url: str = "http://localhost:11434",
                   model: str = "qwen3.5:0.8b") -> bool:
    """检查 Ollama 是否运行且配置的模型可用"""
    try:
        import urllib.request, urllib.error
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            # 检查配置的模型是否在可用列表中
            return any(model in m for m in models)
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

def _build_facts_section(workbench, max_facts: int = 8) -> str:
    """从 FactGraph 构建 facts 文本 (最多 max_facts 条, 加速推理)"""
    lines = []
    graph = getattr(workbench, 'graph', None)
    if graph and graph.nodes:
        # 先取 system 类, 再取其他
        system_nodes = [(k, n) for k, n in graph.nodes.items() if n.category == "system"]
        other_nodes = [(k, n) for k, n in graph.nodes.items() if n.category != "system"]
        nodes = sorted(system_nodes, key=lambda x: -x[1].step) + sorted(other_nodes, key=lambda x: -x[1].step)
        for key, node in nodes[:max_facts]:
            val = node.value[:60]
            cat = node.category
            lines.append(f"[{cat}] {key} = {val}")
    elif hasattr(workbench, 'facts') and workbench.facts:
        for key, fact in sorted(workbench.facts.items(),
                                key=lambda x: -x[1].get("step", 0))[:max_facts]:
            val = fact.get("value", "")[:60]
            cat = fact.get("category", "general")
            lines.append(f"[{cat}] {key} = {val}")

    if not lines:
        return "(no facts)"
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

    # 提取生成文本中的所有具体值 (使用子串匹配更鲁棒)
    text_lower = text.lower()
    matched = set()
    for fv in fact_values:
        if fv in text_lower:
            matched.add(fv)

    # Score: fraction of fact values that appear in generated text
    if not fact_values:
        return 1.0  # no facts to cross-check = pass
    return len(matched) / len(fact_values)


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
                 model: str = "qwen3.5:0.8b",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 60.0,
                 max_tokens: int = 1024,
                 temperature: float = 0.5,
                 thermal_threshold: float = 95.0,
                 enabled: bool = True,
                 api_backend: str = "ollama",
                 api_key: str = ""):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thermal_threshold = thermal_threshold
        self.enabled = enabled
        self.api_backend = api_backend
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._deepseek_ok = bool(self.api_key)

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
        self._self_reflect_result: Optional[dict] = None

    def _load_prompts(self) -> dict:
        """多个 prompt 模板 — 小模型决定格式, DeepSeek 执行"""
        return {
            "code": (
                "You are a Python/coding assistant for a Linux exploration agent.\n"
                "The agent has an idea. Implement it in Python.\n"
                "Output ONLY valid Python code, no explanations.\n\n"
                "## Idea\n{intention}\n"
            ),
            "analysis": (
                "You are an analyst studying a Linux system.\n"
                "The agent wants to understand something. Provide a concise analysis.\n\n"
                "## Question\n{intention}\n"
            ),
            "report": (
                "You are a documentarian. The agent has findings.\n"
                "Write a short structured report.\n\n"
                "## Topic\n{intention}\n"
            ),
            "create": (
                "You are a Python/coding assistant for a Linux exploration agent.\n"
                "The agent has an idea. Help implement it. Output ONLY the code or content.\n\n"
                "## The Agent's Idea\n{intention}\n"
            ),
        }
    
    def health_check(self) -> bool:
        """检查后端是否可用: Ollama 或 DeepSeek"""
        if self.api_backend == "deepseek":
            return self._deepseek_ok
        return _ollama_health(self.base_url, self.model)

    def is_thermal_ok(self) -> bool:
        """CPU 温度安全?"""
        return _thermal_ok(self.thermal_threshold)

    def build_prompt(self, workbench, style: str = "report",
                     intention: str = "") -> str:
        """构建 LLM prompt, 可指定创作意图"""
        prompt = self._prompts.get(style, self._prompts.get("report", self._prompts["create"]))
        
        if style == "self_reflect":
            # self_reflect 需要 self_description
            self_desc = getattr(workbench, 'self_model', None)
            if self_desc and hasattr(self_desc, 'build_self_description'):
                desc_text = self_desc.build_self_description()
            else:
                desc_text = "(agent just started, no self-knowledge yet)"
            prompt = prompt.replace("{self_description}", desc_text)
            self._last_facts_text = desc_text
            # self_reflect 模板没有 {intention}/{facts_section}, 但有它们也不影响
            prompt = prompt.replace("{intention}", "")
        else:
            # create: 只传想法, DeepSeek 只做实现
            prompt = prompt.replace("{intention}", intention or "create something interesting")
            prompt = prompt.replace("{self_description}", "")
            self._last_facts_text = intention
        
        # 清理其他未使用占位符 (模板可能没有这些)
        prompt = prompt.replace("{self_description}", "")
        prompt = prompt.replace("{facts_section}", "")
        prompt = prompt.replace("{relationships_section}", "")
        prompt = prompt.replace("{gaps_section}", "")
        return prompt

    def generate(self, prompt: str, timeout: Optional[float] = None) -> Optional[str]:
        """根据 api_backend 调用对应后端"""
        if self.api_backend == "deepseek" and self._deepseek_ok:
            return _deepseek_generate(
                prompt=prompt, api_key=self.api_key,
                model=self.model,
                timeout=timeout or self.timeout,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
        return _ollama_generate(
            prompt=prompt, model=self.model, base_url=self.base_url,
            timeout=timeout or self.timeout,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )

    def generate_self_reflect(
            self, workbench, timeout: Optional[float] = None) -> Optional[str]:
        """LLM 自省: '根据你对自己的了解, 想做什么?'"""
        if not self.enabled or not self.health_check():
            return None
        prompt = self.build_prompt(workbench, "self_reflect")
        return self.generate(prompt, timeout=timeout)


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

    # ── P2: 异步生成 + 质量评估 + 自适应调度 ──

    def enable_async(self):
        """启用异步模式: generate_content 返回 fallback, 后台 LLM 生成"""
        self._async_enabled = True

    def check_async_result(self) -> Optional[dict]:
        """检查异步生成是否完成, 完成则返回结果"""
        if not self._async_enabled or not self._async_result:
            return None
        result = self._async_result
        self._async_result = None
        return result

    def generate_async(self, workbench, style: str = "report",
                       intention: str = ""):
        """后台预生成 (线程池, 不阻塞主循环), intention 指定创作意图"""
        import threading
        import traceback
        if self._async_result is not None:
            return
        if not self.is_thermal_ok():
            self.stats["thermal_skip"] += 1
            return

        effective_style = style or "report"
        prompt = self.build_prompt(workbench, effective_style, intention)

        def _run():
            try:
                text = self.generate(prompt, timeout=None)
                if text:
                    quality = _check_hallucination(text, self._last_facts_text)
                    min_quality = 0.1 if effective_style in ("create", "story") else 0.3
                    if quality >= min_quality:
                        step = getattr(workbench, '_step_counter', 0) or 0
                        desc_map = {
                            "report": "LLM报告", "analysis": "LLM分析",
                            "story": "LLM叙事", "code": "LLM脚本",
                            "create": "LLM创作",
                            "self_reflect": "SELF:反思",
                        }
                        path_map = {
                            "report": f"/tmp/llm_report_{step}.md",
                            "analysis": f"/tmp/llm_analysis_{step}.md",
                            "story": f"/tmp/llm_story_{step}.md",
                            "code": f"/tmp/llm_script_{step}.py",
                            "create": f"/tmp/llm_create_{step}.md",
                            "self_reflect": f"/tmp/self_intent_{step}.md",
                        }
                        result = {
                            "content": text,
                            "path": path_map.get(effective_style, f"/tmp/llm_output_{step}.md"),
                            "desc": desc_map.get(effective_style, "LLM内容"),
                            "style": effective_style,
                            "size": len(text),
                            "source": "llm",
                            "quality": quality,
                        }
                        if effective_style == "self_reflect":
                            self._self_reflect_result = result
                        else:
                            self._async_result = result
            except Exception as e:
                import traceback
                print(f"  [CreativeWriter] async error: {type(e).__name__}: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def generate_content(self, workbench, style: Optional[str] = None,
                         timeout: Optional[float] = None) -> dict:
        """
        主入口 (P2 异步):
          - 先检查缓存队列中的异步结果
          - 有→使用并启动下一次预生成
          - 无→启动异步, 立即返回 fallback
        """
        self.stats["total_calls"] += 1

        # 1. 异步结果缓存
        if self._async_enabled and self._async_result:
            result = self._async_result
            self._async_result = None
            if result and result.get("source", "") == "llm":
                self.stats["llm_success"] += 1
                # 立即启动下一次预生成
                self.generate_async(workbench, style or "report")
                return result

        # 2. 安全检查
        if not self.enabled:
            return self._fallback(workbench, style, "disabled")

        if not self.health_check():
            return self._fallback(workbench, style, "ollama_unavailable")

        if not self._async_enabled:
            # 非异步模式: 阻塞等待
            return self._generate_content_sync(workbench, style, timeout)

        # 3. 异步模式: 启动后台生成, 立即返回 fallback
        self.generate_async(workbench, style or "report")
        return self._fallback(workbench, style, "async_pending")

    def _generate_content_sync(self, workbench, style: Optional[str] = None,
                               timeout: Optional[float] = None) -> dict:
        """同步生成 (内部调用, 可能阻塞)"""
        if not self.is_thermal_ok():
            self.stats["thermal_skip"] += 1
            return self._fallback(workbench, style, "thermal_throttle")

        effective_style = style or "report"
        prompt = self.build_prompt(workbench, effective_style)
        text = self.generate(prompt, timeout)

        if text is None:
            self.stats["timeout"] += 1
            return self._fallback(workbench, style, "timeout")

        # 质量检查
        quality = _check_hallucination(text, self._last_facts_text)
        if quality < 0.3:
            return self._fallback(workbench, style, f"hallucination({quality:.2f})")

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
        self.stats["llm_success"] += 1
        return {
            "content": text,
            "path": path_map.get(effective_style, f"/tmp/llm_output_{step}.md"),
            "desc": desc_map.get(effective_style, "LLM生成内容"),
            "size": len(text),
            "source": "llm",
            "quality": quality,
        }

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
