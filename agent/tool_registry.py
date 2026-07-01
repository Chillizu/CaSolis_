"""
ToolRegistry — 工具注册表 (支持 JSON + .py)

JSON 工具: {"cmd":["cmd","--help"], "desc":"...", "cat":"discovered"}
  → 直接用 sandbox.execute() 运行
.py 工具: 兼容老版 Python 脚本 (运行时被动态导入)
  → 保持 backward compatibility
"""

import json, os, sys, importlib.util
from pathlib import Path

class ToolRegistry:
    def __init__(self, tools_dir: str = "data/persistent/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}
        self._stats_file = self.tools_dir / "_registry.json"
        self._load_stats()
        self._scan()

    def _load_stats(self):
        if self._stats_file.exists():
            try:
                self._index = json.loads(self._stats_file.read_text())
            except Exception:
                self._index = {}

    def _save_stats(self):
        try:
            self._stats_file.write_text(json.dumps(self._index, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _scan(self):
        """扫描 tools/ 目录, 发现所有 .json 和 .py 工具"""
        for f in list(self.tools_dir.glob("tool_*.json")) + list(self.tools_dir.glob("tool_*.py")):
            if f.name not in self._index:
                meta = {
                    "name": f.name,
                    "tool_type": self._infer_type(f),
                    "description": "",
                    "use_count": 0,
                    "success_count": 0,
                    "last_used_step": 0,
                    "created_step": 0,
                    "avg_bytes": 0,
                    "format": "json" if f.suffix == ".json" else "py",
                }
                # 尝试从 JSON 读取描述
                if f.suffix == ".json":
                    try:
                        data = json.loads(f.read_text())
                        if "desc" in data:
                            meta["description"] = data["desc"]
                    except Exception:
                        pass
                self._index[f.name] = meta
        self._save_stats()

    def _infer_type(self, path: Path) -> str:
        name = path.name.lower()
        if "list" in name or "scan" in name or "gather" in name:
            return "data_gather"
        if "analyze" in name or "check" in name or "validate" in name:
            return "analysis"
        if "write" in name or "generate" in name or "create" in name:
            return "creative"
        return "utility"

    def register(self, name: str, description: str = "",
                 tool_type: str = "utility", created_step: int = 0):
        file_path = self.tools_dir / name
        if not file_path.exists():
            return False
        if name not in self._index:
            self._index[name] = {
                "name": name, "tool_type": tool_type, "description": description,
                "use_count": 0, "success_count": 0, "last_used_step": created_step,
                "created_step": created_step, "avg_bytes": 0,
                "format": "json" if file_path.suffix == ".json" else "py",
            }
        else:
            self._index[name].update({
                "description": description or self._index[name].get("description", ""),
                "tool_type": tool_type or self._index[name].get("tool_type", "utility"),
            })
        self._save_stats()
        return True

    def log_use(self, name: str, step: int, success: bool, bytes_created: int = 0):
        if name not in self._index:
            return
        meta = self._index[name]
        meta["use_count"] += 1
        if success:
            meta["success_count"] += 1
        meta["last_used_step"] = step
        old_avg = meta.get("avg_bytes", 0)
        n = meta["use_count"]
        meta["avg_bytes"] = (old_avg * (n - 1) + bytes_created) / n
        self._save_stats()

    def get_available(self, min_success_rate: float = 0.0) -> list[dict]:
        tools = []
        for name, meta in self._index.items():
            file_path = self.tools_dir / name
            if not file_path.exists():
                continue
            rate = meta["success_count"] / max(meta["use_count"], 1)
            if rate >= min_success_rate:
                tools.append({**meta, "success_rate": rate, "path": str(file_path)})
        return sorted(tools, key=lambda t: -t["success_rate"])

    def get_best_tool(self, preferred_type: str = "") -> dict | None:
        tools = self.get_available()
        if preferred_type:
            typed = [t for t in tools if t.get("tool_type") == preferred_type]
            if typed:
                return typed[0]
        return tools[0] if tools else None

    def run_tool(self, name: str, env: dict) -> dict:
        """执行工具: JSON 工具 → sandbox.execute, .py 工具 → import + run()"""
        file_path = self.tools_dir / name
        if not file_path.exists():
            return {"success": False, "data": {}, "output": "", "summary": f"not found: {name}"}

        # JSON 工具: 直接执行命令
        if file_path.suffix == ".json":
            try:
                data = json.loads(file_path.read_text())
                cmd = data.get("cmd", [])
                if not cmd:
                    return {"success": False, "data": {}, "output": "", "summary": "no cmd in json"}
                sandbox = env.get("sandbox")
                if not sandbox:
                    return {"success": False, "data": {}, "output": "", "summary": "no sandbox"}
                if isinstance(cmd, list):
                    result = sandbox.execute_list(cmd, timeout=5)
                else:
                    result = sandbox.execute(cmd, timeout=5)
                output = (result.stdout or result.stderr or "").strip()
                return {
                    "success": result.exit_code == 0,
                    "data": {"cmd": cmd, "output": output[:500]},
                    "output": output[:500],
                    "summary": f"{' '.join(cmd) if isinstance(cmd,list) else cmd}: {len(output)}B"
                }
            except Exception as e:
                return {"success": False, "data": {}, "output": "", "summary": str(e)}

        # .py 工具: 兼容老版
        try:
            spec = importlib.util.spec_from_file_location(name.replace(".py", ""), str(file_path))
            if spec is None or spec.loader is None:
                return {"success": False, "data": {}, "output": "", "summary": f"cannot import {name}"}
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(self.tools_dir))
            try:
                spec.loader.exec_module(module)
            finally:
                if str(self.tools_dir) in sys.path:
                    sys.path.remove(str(self.tools_dir))
            if not hasattr(module, 'run'):
                return {"success": False, "data": {}, "output": "", "summary": f"no run() in {name}"}
            result = module.run(env)
            return result if isinstance(result, dict) else {"success": True, "output": str(result)}
        except Exception as e:
            return {"success": False, "data": {}, "output": "", "summary": f"error: {e}"}

    def get_stats(self) -> dict:
        tools = self.get_available()
        return {
            "n_tools": len(tools),
            "tools": tools[:10],
            "total_uses": sum(t.get("use_count", 0) for t in self._index.values()),
            "unique_successful": sum(1 for t in tools if t.get("success_count", 0) > 0),
        }
