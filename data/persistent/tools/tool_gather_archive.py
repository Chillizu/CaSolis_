"""
Tool: Archive Gatherer
Category: archive
Auto-generated from category facts
"""

def run(env: dict) -> dict:
    """
    Gather archive information from sandbox
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "archive_data": [], "summary": "no sandbox"}

        data = []
        commands = ['tar --version | head -1', 'gzip --version 2>/dev/null | head -1', 'bzip2 --version 2>/dev/null | head -1', 'xz --version 2>/dev/null | head -1', 'unzip --version 2>/dev/null | head -1']

        for cmd in commands:
            result = sandbox.execute(cmd, timeout=3)
            if result and result.stdout and result.stdout.strip():
                data.append({
                    "cmd": cmd,
                    "output": result.stdout.strip()[:500],
                })

        if not data:
            return {"success": True, "archive_data": [], "summary": "no data found"}

        summary = f"{len(data)} commands executed"
        return {
            "success": True,
            "archive_data": data,
            "summary": summary,
        }
    except Exception as e:
        return {"success": False, "archive_data": [], "summary": str(e)}
