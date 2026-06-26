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
from pathlib import Path

TOOLS_DIR = Path("data/persistent/tools")

# ── 工具模板 ──

TOOL_TEMPLATES = {

    # ─── 数据采集类 ───

    "gather_packages": {
        "type": "data_gather",
        "description": "列出所有已安装的 dpkg 包",
        "code": '''"""
Tool: 采集已安装包列表
用法: run(env) → {"packages": [...], "count": int}
"""

def run(env: dict) -> dict:
    """
    从沙箱采集所有已安装的 dpkg 包
    """
    import subprocess

    try:
        sandbox = env.get("sandbox")
        if sandbox:
            result = sandbox.execute("dpkg -l 2>/dev/null | grep '^ii' | awk '{print $2, $3}'")
            if result and result.stdout:
                lines = result.stdout.strip().split("\\n")
                packages = []
                for line in lines:
                    parts = line.strip().split(None, 1)
                    if len(parts) >= 1:
                        packages.append({"name": parts[0], "version": parts[1] if len(parts) > 1 else ""})
                return {
                    "success": True,
                    "packages": packages,
                    "count": len(packages),
                    "summary": f"found {len(packages)} installed packages"
                }

        return {"success": False, "packages": [], "count": 0, "summary": "no sandbox"}
    except Exception as e:
        return {"success": False, "packages": [], "count": 0, "summary": str(e)}
''',
    },

    "gather_network": {
        "type": "data_gather",
        "description": "采集网络接口和路由信息",
        "code": '''"""
Tool: 采集网络信息
用法: run(env) → {"interfaces": [...], "routes": [...]}
"""

def run(env: dict) -> dict:
    """
    从沙箱采集网络接口、路由、ARP 信息
    """
    result = {"interfaces": [], "routes": [], "success": False}
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "summary": "no sandbox"}

        # IP 地址
        r = sandbox.execute("ip addr 2>/dev/null | grep -E '^[0-9]+:|inet '")
        if r and r.stdout:
            lines = r.stdout.strip().split("\\n")
            current_iface = None
            for line in lines:
                if ":" in line and not line.strip().startswith("inet"):
                    import re
                    m = re.match(r"\\d+:\\s+(\\w+)", line)
                    if m:
                        current_iface = {"name": m.group(1), "ips": []}
                        result["interfaces"].append(current_iface)
                elif "inet " in line and current_iface is not None:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        current_iface["ips"].append(parts[1])

        # 路由
        r2 = sandbox.execute("ip route 2>/dev/null")
        if r2 and r2.stdout:
            for line in r2.stdout.strip().split("\\n"):
                if line.strip():
                    result["routes"].append(line.strip())

        result["success"] = True
        n_ips = sum(len(i.get("ips", [])) for i in result["interfaces"])
        result["summary"] = f"{len(result['interfaces'])} interfaces, {n_ips} IPs"
        return result
    except Exception as e:
        return {"success": False, "summary": str(e)}
''',
    },

    "gather_processes": {
        "type": "data_gather",
        "description": "采集当前进程列表",
        "code": '''"""
Tool: 采集进程列表
用法: run(env) → {"processes": [...], "count": int}
"""

def run(env: dict) -> dict:
    """
    从沙箱采集 ps 进程列表
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "processes": [], "count": 0, "summary": "no sandbox"}

        r = sandbox.execute("ps aux 2>/dev/null | head -50")
        if r and r.stdout:
            lines = r.stdout.strip().split("\\n")
            processes = []
            for line in lines[1:]:  # skip header
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "cmd": parts[10][:60],
                    })
            return {
                "success": True,
                "processes": processes,
                "count": len(processes),
                "summary": f"{len(processes)} processes"
            }

        return {"success": False, "processes": [], "count": 0, "summary": "empty output"}
    except Exception as e:
        return {"success": False, "processes": [], "count": 0, "summary": str(e)}
''',
    },

    "gather_storage": {
        "type": "data_gather",
        "description": "采集磁盘挂载和文件系统信息",
        "code": '''"""
Tool: 采集磁盘/文件系统信息
用法: run(env) → {"mounts": [...], "disks": [...]}
"""

def run(env: dict) -> dict:
    """
    采集磁盘挂载、文件系统类型、使用量
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "mounts": [], "disks": [], "summary": "no sandbox"}

        result = {"mounts": [], "disks": [], "success": False}

        r = sandbox.execute("df -hT 2>/dev/null | tail -n +2")
        if r and r.stdout:
            for line in r.stdout.strip().split("\\n"):
                parts = line.split()
                if len(parts) >= 7:
                    result["mounts"].append({
                        "filesystem": parts[0],
                        "type": parts[1],
                        "size": parts[2],
                        "used": parts[3],
                        "avail": parts[4],
                        "use_pct": parts[5],
                        "mount": parts[6],
                    })

        r2 = sandbox.execute("lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null | tail -n +2")
        if r2 and r2.stdout:
            for line in r2.stdout.strip().split("\\n"):
                parts = line.split()
                if len(parts) >= 3:
                    result["disks"].append({
                        "name": parts[0],
                        "size": parts[1],
                        "type": parts[2],
                        "mount": parts[3] if len(parts) > 3 else "",
                    })

        result["success"] = True
        result["summary"] = f"{len(result['mounts'])} mounts, {len(result['disks'])} disks"
        return result
    except Exception as e:
        return {"success": False, "mounts": [], "disks": [], "summary": str(e)}
''',
    },

    "gather_hardware": {
        "type": "data_gather",
        "description": "采集 CPU/内存/系统架构硬件信息",
        "code": '''"""
Tool: 采集硬件信息
用法: run(env) → {"cpu": {...}, "memory": {...}, "arch": "..."}
"""

def run(env: dict) -> dict:
    """
    采集 CPU 型号、核心数、内存总量、架构等硬件信息
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "summary": "no sandbox"}

        result = {"cpu": {}, "memory": {}, "arch": "", "success": False}

        # CPU
        r = sandbox.execute("cat /proc/cpuinfo 2>/dev/null | grep -E '^(processor|model name|cpu cores|vendor_id|cpu family)' | head -10")
        if r and r.stdout:
            for line in r.stdout.strip().split("\\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    result["cpu"][k.strip()] = v.strip()

        # Memory
        r2 = sandbox.execute("cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|SwapTotal|SwapFree)'")
        if r2 and r2.stdout:
            for line in r2.stdout.strip().split("\\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    result["memory"][k.strip()] = v.strip()

        # Architecture
        r3 = sandbox.execute("uname -m 2>/dev/null")
        if r3 and r3.stdout:
            result["arch"] = r3.stdout.strip()

        result["success"] = True
        result["summary"] = f"{result['cpu'].get('model name', '?')[:30]}, {result['memory'].get('MemTotal', '?')}"
        return result
    except Exception as e:
        return {"success": False, "summary": str(e)}
''',
    },

    # ─── 分析类 ───

    "analyze_system_profile": {
        "type": "analysis",
        "description": "综合系统画像: 合并硬件/网络/进程/存储数据",
        "code": '''"""
Tool: 综合系统画像
用法: run(env) → {"profile": str}
依赖: 先运行 gather_* 工具
"""

def run(env: dict) -> dict:
    """
    从 env 中读取已有的事实, 生成综合系统画像
    """
    wb = env.get("workbench")
    if not wb:
        return {"success": False, "profile": "", "summary": "no workbench"}

    # 从 FactGraph 收集关键事实
    facts = {}
    if hasattr(wb, 'graph'):
        for key, node in wb.graph.nodes.items():
            facts[key] = str(node.value)

    lines = []
    lines.append("# System Profile (Auto-generated)")
    lines.append("")

    # 系统信息
    for key in ["os_name", "os_version", "kernel_version", "arch"]:
        val = next((v for k, v in facts.items() if key in k), "")
        if val:
            lines.append(f"- {key}: {val}")

    # CPU
    cpu_info = {k: v for k, v in facts.items() if "cpu_" in k}
    if cpu_info:
        lines.append(f"- CPU: {cpu_info.get('cpu_model_name', '?')}")
        pc = cpu_info.get("cpu_processor", "")
        if pc:
            lines.append(f"- CPU cores: {int(pc) + 1 if pc.isdigit() else pc}")

    # Memory
    mem_total = facts.get("mem_memtotal", "")
    if mem_total:
        lines.append(f"- Memory: {mem_total}")

    # Network
    ifaces = {k: v for k, v in facts.items() if "net_iface" in k}
    if ifaces:
        lines.append(f"- Interfaces: {len(ifaces)}")
    ip_addr = facts.get("ip_address", "")
    if ip_addr:
        lines.append(f"- IP: {ip_addr}")

    # Capabilities
    caps = {k: v for k, v in facts.items() if "capability" in k}
    if caps:
        lines.append(f"- Capabilities: {', '.join(caps.keys())}")

    summary = "\\n".join(lines)
    return {"success": True, "profile": summary, "summary": f"profile with {len(lines)} lines"}
''',
    },
}


class ToolFactory:
    """工具工厂: 从模板生成 Python 工具文件"""

    def __init__(self, tools_dir: str = "data/persistent/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def get_available_templates(self) -> list[dict]:
        """返回所有可用模板的信息"""
        return [
            {
                "name": name,
                "type": info["type"],
                "description": info["description"],
            }
            for name, info in TOOL_TEMPLATES.items()
        ]

    def generate(self, template_name: str, created_step: int = 0) -> Optional[str]:
        """
        根据模板名生成工具文件

        Returns:
            文件名 (如 tool_gather_packages.py), 失败返回 None
        """
        if template_name not in TOOL_TEMPLATES:
            return None

        info = TOOL_TEMPLATES[template_name]
        filename = f"tool_{template_name}.py"
        filepath = self.tools_dir / filename

        if filepath.exists():
            return filename  # 已存在, 直接返回

        try:
            filepath.write_text(info["code"])
            return filename
        except Exception as e:
            print(f"  ⚠️ ToolFactory: 生成 {filename} 失败: {e}")
            return None

    def generate_all(self, created_step: int = 0) -> list[str]:
        """生成所有可用的工具"""
        generated = []
        for name in TOOL_TEMPLATES:
            fname = self.generate(name, created_step)
            if fname:
                generated.append(fname)
        return generated

    def get_tool_info(self, template_name: str) -> Optional[dict]:
        """获取工具模板信息"""
        info = TOOL_TEMPLATES.get(template_name)
        if not info:
            return None
        return {
            "name": f"tool_{template_name}.py",
            "type": info["type"],
            "description": info["description"],
        }

    # ── P13++: 观察驱动的工具自生成 ──

    def auto_generate_from_facts(self, fact_graph) -> list[str]:
        """
        根据 FactGraph 中积累的事实类别, 自动生成缺失的工具

        Args:
            fact_graph: FactGraph 实例

        Returns:
            新生成的文件名列表
        """
        generated = []
        if not hasattr(fact_graph, 'nodes'):
            return generated

        # 统计每个类别的事实数
        cat_count = {}
        for node in fact_graph.nodes.values():
            cat = node.category
            cat_count[cat] = cat_count.get(cat, 0) + 1

        # 类别→工具的映射: 当某个类别事实够多但没有对应工具时生成
        cat_to_tool = {
            "network": "gather_network",
            "package": "gather_packages",
            "file": "gather_storage",
            "system": "gather_hardware",
            "dev": "analyze_system_profile",
        }

        for cat, tool_name in cat_to_tool.items():
            if cat_count.get(cat, 0) < 3:
                continue  # 不够多, 不生成
            # 检查工具是否已存在
            fname = f"tool_{tool_name}.py"
            if (self.tools_dir / fname).exists():
                continue
            # 生成
            result = self.generate(tool_name)
            if result:
                generated.append(result)
                print(f"    [TOOL_AUTO] 从类别'{cat}'({cat_count[cat]}事实) → {result}")

        return generated
