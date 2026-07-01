#!/usr/bin/env python3
"""
测试 CreativeWriter 的独立功能

用法:
  python scripts/test_creative_writer.py              # 仅测试 CI/fallback
  python scripts/test_creative_writer.py --online      # 真实 Ollama 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.creative_writer import (
    CreativeWriter, _ollama_health, _thermal_ok,
    _build_facts_section, _build_relationships_section, _build_gaps_section,
    _check_hallucination,
)


def test_ollama_health():
    """Ollama 健康检查"""
    ok = _ollama_health()
    print(f"[TEST] _ollama_health() = {ok}")
    return ok


def test_thermal():
    """温度检查"""
    ok = _thermal_ok()
    print(f"[TEST] _thermal_ok() = {ok}")
    return True


def test_prompt_building():
    """用 mock Workbench 测试 prompt 构建"""
    from unittest.mock import MagicMock
    from agent.fact_graph import FactGraph

    wb = MagicMock()
    g = FactGraph()
    g.add_node("os_name", "Debian", category="system", step=1)
    g.add_node("os_version_id", "12", category="system", step=2)
    g.add_node("kernel", "7.0.12-arch1-1", category="system", step=3)
    g.add_node("cpu_cores", "22", category="system", step=4)
    g.add_node("mem_total", "7.6Gi", category="system", step=5)
    g.add_node("hostname", "c9c341a36f9f", category="system", step=6)
    g.add_edge("os_name", "os_version_id", "requires")
    g.add_edge("cpu_cores", "cpu_model", "extends")
    wb.graph = g
    wb.facts = {}

    writer = CreativeWriter(enabled=True)
    prompt = writer.build_prompt(wb, style="report")

    assert "Debian" in prompt, "事实应出现在 prompt 中"
    assert "os_name" in prompt, "事实 key 应出现在 prompt 中"
    assert "requires" in prompt, "关系应出现在 prompt 中"
    assert "Report" in prompt or "report" in prompt, "风格指令应出现"

    print(f"[TEST] Prompt 构建 OK ({len(prompt)} chars)")
    return True


def test_hallucination_check():
    """幻觉检测"""
    facts = "[system] cpu_cores = 22 (confidence=1.0, source=cat /proc/cpuinfo)\n[system] kernel = 7.0.12"
    good_text = "The system has 22 cores running kernel 7.0.12"
    bad_text = "The system has 999 cores and runs on Mars OS"

    score_good = _check_hallucination(good_text, facts)
    score_bad = _check_hallucination(bad_text, facts)

    assert score_good > score_bad, "正常文本的幻觉分应高于幻觉文本"

    print(f"[TEST] 幻觉检测: good={score_good:.2f}, bad={score_bad:.2f}")
    return True


def test_fallback():
    """Fallback 回模板"""
    from unittest.mock import MagicMock

    wb = MagicMock()
    # Simulate build_write_content
    wb.build_write_content.return_value = {
        "content": "# Fallback Report\n\nSystem: Debian",
        "path": "/tmp/fallback.md",
        "desc": "Markdown分析报告",
        "size": 35,
    }

    writer = CreativeWriter(enabled=False)  # disabled → force fallback
    result = writer.generate_content(wb, style="report")

    assert result["source"].startswith("fallback:"), "应为 fallback"
    assert result["size"] > 0, "应有内容"
    print(f"[TEST] Fallback OK: source={result['source']}, size={result['size']}")
    return True


def test_stats():
    """统计"""
    from unittest.mock import MagicMock
    wb = MagicMock()
    wb.build_write_content.return_value = {
        "content": "test", "path": "/tmp/t.txt",
        "desc": "t", "size": 4,
    }
    writer = CreativeWriter(enabled=False)
    writer.generate_content(wb)
    writer.generate_content(wb)

    s = writer.get_stats()
    assert s["total_calls"] == 2
    assert s["fallback"] == 2
    print(f"[TEST] Stats OK: total_calls={s['total_calls']}, fallback={s['fallback']}")
    return True


def test_online():
    """真实 Ollama 生成测试 (需要 --online)"""
    from agent.workbench import Workbench

    wb = Workbench(max_facts=40)
    wb._add_fact("os_name", "Debian", "INFO", "cat /etc/os-release", 1, category="system")
    wb._add_fact("kernel", "7.0.12", "INFO", "uname -a", 2, category="system")
    wb._add_fact("cpu_cores", "4", "INFO", "cat /proc/cpuinfo", 3, category="system")

    writer = CreativeWriter(enabled=True)
    if not writer.health_check():
        print("[SKIP] Ollama 不可用, 跳过在线测试")
        return True

    result = writer.generate_content(wb, style="report")
    print(f"[TEST] Online 生成: source={result['source']}, size={result['size']}")
    if result["source"] == "llm":
        print(f"[CONTENT]\n{result['content'][:500]}")
    return True


if __name__ == "__main__":
    tests = [
        ("thermal", test_thermal),
        ("prompt_building", test_prompt_building),
        ("hallucination_check", test_hallucination_check),
        ("fallback", test_fallback),
        ("stats", test_stats),
    ]

    if "--online" in sys.argv:
        tests.append(("online(llm)", test_online))

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"通过: {passed}/{len(tests)}, 失败: {failed}")
    sys.exit(1 if failed else 0)
