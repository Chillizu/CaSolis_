"""测试各 LLM 模型对结构化 prompt 的响应能力"""
import requests
import json

models = [
    "deepseek-r1:1.5b",
    "qwen3.5:0.8b",
    "gemma4:e4b",
]

prompts = [
    "hi",
    "say hello",
    "what is 2+2?",
    'output JSON: {"x": 1}',
    "意图: READ\n参数: {\"path\": \"/etc\"}",
    "意图: READ — 读文件\n输出 JSON:",
]

for model in models:
    print(f"\n{'='*50}")
    print(f"模型: {model}")
    print(f"{'='*50}")
    for prompt in prompts:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100}
                },
                timeout=30,
            )
            raw = resp.json().get("response", "")
            display = raw[:80].replace("\n", "\\n") if raw else "(空)"
            print(f"  {prompt[:30]:30s} → {display}")
        except Exception as e:
            print(f"  {prompt[:30]:30s} → ❌ {e}")
