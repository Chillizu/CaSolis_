"""
ToolFactory — 工具工厂

生成可复用的 Python 数据采集/分析工具。
工具保存在 data/persistent/tools/ 下, 格式统一, 可被 ToolRegistry 发现和执行。

工具类型:
  - data_gather: 采集特定系统信息
  - analysis: 分析已采集的数据
  - creative: 生成内容

每个生成的工具:
  - 是独立的 .py 文件
  - 实现 run(env) → dict 接口
  - 包含内置文档字符串
  - 可被 ToolRegistry.run_tool() 调用
"""

import os
import re
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path("data/persistent/tools")

# ── 安全命令清单 (LLM 只从这个池子选) ──
SAFE_COMMANDS = {
    "system": [
        "uname -a", "hostname", "uptime", "whoami", "id",
        "lscpu", "free -h", "df -h", "lsblk", "mount",
        "cat /proc/cpuinfo | head -20", "cat /proc/meminfo | head -10",
        "cat /proc/uptime", "cat /proc/loadavg",
        "ps aux --forest | head -20", "top -bn1 | head -10",
    ],
    "network": [
        "ip addr", "ip route", "ss -tuln", "cat /proc/net/dev",
        "cat /proc/net/tcp | head -10", "hostname -I",
        "arp -n 2>/dev/null", "ifconfig 2>/dev/null",
    ],
    "package": [
        "dpkg -l", "dpkg --version", "apt list --installed 2>/dev/null | head -20",
    ],
    "file": [
        "ls -la /", "df -hT", "lsblk -o NAME,SIZE,TYPE,MOUNTPOINT",
        "cat /etc/fstab 2>/dev/null", "findmnt -D",
    ],
    "dev": [
        "python3 --version", "gcc --version 2>/dev/null | head -1",
        "make --version 2>/dev/null | head -1", "perl --version 2>/dev/null | head -2",
        "node --version 2>/dev/null",
    ],
    "archive": [
        "tar --version | head -1", "gzip --version 2>/dev/null | head -1",
        "bzip2 --version 2>/dev/null | head -1", "xz --version 2>/dev/null | head -1",
        "unzip --version 2>/dev/null | head -1",
    ],
    "text": [
        "echo hello", "head -5 /etc/passwd", "wc -l /etc/passwd",
    ],
    "capability": [
        "which python3", "which gcc", "which make", "which perl",
        "which node", "which git", "which docker",
    ],
    "command": [
        "compgen -c 2>/dev/null | head -20", "echo $SHELL", "which bash",
    ],
}

# ── 工具写入模板 ──

TOOL_TEMPLATE = '''"""
Tool: {name}
Category: {category}
Auto-generated from category facts
"""

def run(env: dict) -> dict:
    """
    Gather {category} information from sandbox
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {{"success": False, "{category}_data": [], "summary": "no sandbox"}}

        data = []
        commands = {commands}

        for cmd in commands:
            result = sandbox.execute(cmd, timeout=3)
            if result and result.stdout and result.stdout.strip():
                data.append({{
                    "cmd": cmd,
                    "output": result.stdout.strip()[:500],
                }})

        if not data:
            return {{"success": True, "{category}_data": [], "summary": "no data found"}}

        summary = f"{{len(data)}} commands executed"
        return {{
            "success": True,
            "{category}_data": data,
            "summary": summary,
        }}
    except Exception as e:
        return {{"success": False, "{category}_data": [], "summary": str(e)}}
'''


# ── 固定工具模板 (质量经过验证) ──

TOOL_TEMPLATES = {
    "gather_packages": {
        "type": "data_gather",
        "description": "列出所有已安装的 dpkg 包",
        "code": TOOL_TEMPLATE.format(
            name="Package Lister", category="package",
            commands=SAFE_COMMANDS["package"],
        ),
    },
    "gather_network": {
        "type": "data_gather",
        "description": "采集网络接口和路由信息",
        "code": TOOL_TEMPLATE.format(
            name="Network Scanner", category="network",
            commands=SAFE_COMMANDS["network"],
        ),
    },
    "gather_processes": {
        "type": "data_gather",
        "description": "采集当前进程列表",
        "code": TOOL_TEMPLATE.format(
            name="Process Lister", category="process",
            commands=["ps aux --forest | head -30", "top -bn1 | head -15"],
        ),
    },
    "gather_storage": {
        "type": "data_gather",
        "description": "采集磁盘挂载和文件系统信息",
        "code": TOOL_TEMPLATE.format(
            name="Storage Scanner", category="storage",
            commands=SAFE_COMMANDS["file"],
        ),
    },
    "gather_hardware": {
        "type": "data_gather",
        "description": "采集 CPU/内存/系统架构硬件信息",
        "code": TOOL_TEMPLATE.format(
            name="Hardware Info", category="hardware",
            commands=SAFE_COMMANDS["system"][:5],
        ),
    },
    "analyze_system_profile": {
        "type": "analysis",
        "description": "综合系统画像",
        "code": TOOL_TEMPLATE.format(
            name="System Profiler", category="profile",
            commands=SAFE_COMMANDS["system"][:8],
        ),
    },
}


class ToolFactory:
    """工具工厂: 从安全命令池生成 Python 工具"""

    def __init__(self, tools_dir: str = "data/persistent/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def get_available_templates(self) -> list[dict]:
        return [
            {"name": name, "type": info["type"], "description": info["description"]}
            for name, info in TOOL_TEMPLATES.items()
        ]

    def generate(self, template_name: str, created_step: int = 0) -> Optional[str]:
        if template_name not in TOOL_TEMPLATES:
            return None
        info = TOOL_TEMPLATES[template_name]
        filename = f"tool_{template_name}.py"
        filepath = self.tools_dir / filename
        if filepath.exists():
            return filename
        try:
            filepath.write_text(info["code"])
            return filename
        except Exception as e:
            print(f"  ⚠️ ToolFactory: 生成 {filename} 失败: {e}")
            return None

    def generate_all(self, created_step: int = 0) -> list[str]:
        generated = []
        for name in TOOL_TEMPLATES:
            fname = self.generate(name, created_step)
            if fname:
                generated.append(fname)
        return generated

    def get_tool_info(self, template_name: str) -> Optional[dict]:
        info = TOOL_TEMPLATES.get(template_name)
        if not info:
            return None
        return {
            "name": f"tool_{template_name}.py",
            "type": info["type"],
            "description": info["description"],
        }

    # ── 观察驱动的工具自生成 ──

    def auto_generate_from_facts(self, fact_graph) -> list[str]:
        """
        根据 FactGraph 中积累的事实类别, 自动生成缺失的工具
        新类别从 SAFE_COMMANDS 池取命令, 无需 LLM
        """
        generated = []
        if not hasattr(fact_graph, 'nodes'):
            return generated

        cat_count = {}
        for node in fact_graph.nodes.values():
            cat = node.category
            cat_count[cat] = cat_count.get(cat, 0) + 1

        # 模板覆盖的类别
        template_cats = {"network", "package", "file", "system", "dev"}

        for cat, tool_name in [
            ("network", "gather_network"), ("package", "gather_packages"),
            ("file", "gather_storage"), ("system", "gather_hardware"),
            ("dev", "analyze_system_profile"),
        ]:
            if cat_count.get(cat, 0) < 3:
                continue
            fname = f"tool_{tool_name}.py"
            if (self.tools_dir / fname).exists():
                continue
            result = self.generate(tool_name)
            if result:
                generated.append(result)
                print(f"    [TOOL_AUTO] 模板→{result} (from {cat})")

        # 新类别: 从 SAFE_COMMANDS 池取命令
        new_cats = [
            c for c in cat_count
            if c not in template_cats
            and c not in ("general", "command", "explore", "script")
            and cat_count[c] >= 2
        ]
        for cat in new_cats:
            # 有这个类别的安全命令吗? 没有就用通用命令
            commands = SAFE_COMMANDS.get(cat, SAFE_COMMANDS.get("system"))
            fname = f"tool_gather_{cat}.py"
            if (self.tools_dir / fname).exists():
                continue

            commands = SAFE_COMMANDS[cat]
            code = TOOL_TEMPLATE.format(
                name=f"{cat.title()} Gatherer",
                category=cat,
                commands=commands,
            )
            try:
                (self.tools_dir / fname).write_text(code)
                generated.append(fname)
                print(f"    [TOOL_DYN] {fname} ({len(commands)} cmds, from {cat}: {cat_count[cat]} facts)")
            except Exception as e:
                print(f"    ⚠️ [TOOL_DYN] 写入失败: {e}")

        return generated
