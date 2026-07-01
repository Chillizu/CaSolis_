"""
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
            lines = r.stdout.strip().split("\n")
            current_iface = None
            for line in lines:
                if ":" in line and not line.strip().startswith("inet"):
                    import re
                    m = re.match(r"\d+:\s+(\w+)", line)
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
            for line in r2.stdout.strip().split("\n"):
                if line.strip():
                    result["routes"].append(line.strip())

        result["success"] = True
        n_ips = sum(len(i.get("ips", [])) for i in result["interfaces"])
        result["summary"] = f"{len(result['interfaces'])} interfaces, {n_ips} IPs"
        return result
    except Exception as e:
        return {"success": False, "summary": str(e)}
