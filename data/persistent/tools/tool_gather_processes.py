"""
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
            lines = r.stdout.strip().split("\n")
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
