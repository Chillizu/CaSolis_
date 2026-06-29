"""
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

    summary = "\n".join(lines)
    return {"success": True, "profile": summary, "summary": f"profile with {len(lines)} lines"}
