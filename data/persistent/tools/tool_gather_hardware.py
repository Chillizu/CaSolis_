"""
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
            for line in r.stdout.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    result["cpu"][k.strip()] = v.strip()

        # Memory
        r2 = sandbox.execute("cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|SwapTotal|SwapFree)'")
        if r2 and r2.stdout:
            for line in r2.stdout.strip().split("\n"):
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
