"""
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
                lines = result.stdout.strip().split("\n")
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
