"""
ToolRegistry — 工具注册表

核心职责:
  1. 扫描 data/persistent/tools/ 发现可用工具
  2. 注册工具 (名称/类型/描述/使用统计)
  3. 提供工具列表供 TOOL 意图使用
  4. 记录使用次数和成功率
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


class ToolRegistry:
    """工具注册表: 发现、注册、统计"""

    def __init__(self, tools_dir: str = "data/persistent/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}  # name → metadata
        self._stats_file = self.tools_dir / "_registry.json"

        # 加载持久化的统计
        self._load_stats()

        # 扫描工具目录
        self._scan()

    def _load_stats(self):
        """加载持久化的使用统计"""
        if self._stats_file.exists():
            try:
                stats = json.loads(self._stats_file.read_text())
                self._index = stats
            except Exception:
                self._index = {}

    def _save_stats(self):
        """保存使用统计"""
        try:
            self._stats_file.write_text(json.dumps(self._index, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _scan(self):
        """扫描 tools/ 目录, 发现所有 .py 文件"""
        for f in self.tools_dir.glob("tool_*.py"):
            if f.name not in self._index:
                self._index[f.name] = {
                    "name": f.name,
                    "tool_type": self._infer_type(f),
                    "description": "",
                    "use_count": 0,
                    "success_count": 0,
                    "last_used_step": 0,
                    "created_step": 0,
                    "avg_bytes": 0,
                }
        self._save_stats()

    def _infer_type(self, path: Path) -> str:
        """从文件名推断工具类型"""
        name = path.name.lower()
        if "list" in name or "scan" in name or "gather" in name:
            return "data_gather"
        if "analyze" in name or "check" in name or "validate" in name:
            return "analysis"
        if "write" in name or "generate" in name:
            return "creative"
        return "utility"

    def register(self, name: str, description: str = "",
                 tool_type: str = "utility", created_step: int = 0):
        """注册一个新的工具"""
        file_path = self.tools_dir / name
        if not file_path.exists():
            return False

        if name not in self._index:
            self._index[name] = {
                "name": name,
                "tool_type": tool_type,
                "description": description,
                "use_count": 0,
                "success_count": 0,
                "last_used_step": created_step,
                "created_step": created_step,
                "avg_bytes": 0,
            }
        else:
            self._index[name].update({
                "description": description or self._index[name].get("description", ""),
                "tool_type": tool_type or self._index[name].get("tool_type", "utility"),
            })
        self._save_stats()
        return True

    def log_use(self, name: str, step: int, success: bool, bytes_created: int = 0):
        """记录工具使用结果"""
        if name not in self._index:
            return
        meta = self._index[name]
        meta["use_count"] += 1
        if success:
            meta["success_count"] += 1
        meta["last_used_step"] = step
        # 滑动平均字节数
        old_avg = meta.get("avg_bytes", 0)
        n = meta["use_count"]
        meta["avg_bytes"] = (old_avg * (n - 1) + bytes_created) / n
        self._save_stats()

    def get_available(self, min_success_rate: float = 0.0) -> list[dict]:
        """获取可用工具列表 (按成功率排序)"""
        tools = []
        for name, meta in self._index.items():
            file_path = self.tools_dir / name
            if not file_path.exists():
                continue
            rate = meta["success_count"] / max(meta["use_count"], 1)
            if rate >= min_success_rate:
                tools.append({
                    **meta,
                    "success_rate": rate,
                    "path": str(file_path),
                })
        return sorted(tools, key=lambda t: -t["success_rate"])

    def get_best_tool(self, preferred_type: Optional[str] = None) -> Optional[dict]:
        """获取最适合的工具"""
        tools = self.get_available()
        if preferred_type:
            typed = [t for t in tools if t["tool_type"] == preferred_type]
            if typed:
                return typed[0]
        return tools[0] if tools else None

    def run_tool(self, name: str, env: dict) -> dict:
        """
        执行一个工具

        Args:
            name: 工具文件名 (如 tool_scan_net.py)
            env: 环境字典 {workbench, sandbox, state_text, ...}

        Returns:
            {"success": bool, "data": dict, "output": str, "summary": str}
        """
        file_path = self.tools_dir / name
        if not file_path.exists():
            return {"success": False, "data": {}, "output": "", "summary": f"tool {name} not found"}

        try:
            # 动态导入工具模块
            spec = importlib.util.spec_from_file_location(
                name.replace(".py", ""), str(file_path)
            )
            if spec is None or spec.loader is None:
                return {"success": False, "data": {}, "output": "",
                        "summary": f"cannot import {name}"}

            module = importlib.util.module_from_spec(spec)
            # 将 tools_dir 加入 sys.path 让工具可以 import
            sys.path.insert(0, str(self.tools_dir))
            try:
                spec.loader.exec_module(module)
            finally:
                if str(self.tools_dir) in sys.path:
                    sys.path.remove(str(self.tools_dir))

            if not hasattr(module, 'run'):
                return {"success": False, "data": {}, "output": "",
                        "summary": f"tool {name} has no run() function"}

            result = module.run(env)
            if isinstance(result, dict):
                return result
            return {"success": True, "data": {}, "output": str(result), "summary": str(result)}

        except Exception as e:
            return {"success": False, "data": {}, "output": "",
                    "summary": f"tool {name} error: {e}"}

    def get_stats(self) -> dict:
        """获取注册表统计"""
        tools = self.get_available()
        return {
            "n_tools": len(tools),
            "tools": tools[:10],
            "total_uses": sum(t.get("use_count", 0) for t in self._index.values()),
            "unique_successful": sum(
                1 for t in tools if t.get("success_count", 0) > 0
            ),
        }
