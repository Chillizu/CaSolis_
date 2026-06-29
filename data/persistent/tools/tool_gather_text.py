"""
Tool: Text Gatherer
Category: text
Auto-generated from category facts
"""

def run(env: dict) -> dict:
    """
    Gather text information from sandbox
    """
    try:
        sandbox = env.get("sandbox")
        if not sandbox:
            return {"success": False, "text_data": [], "summary": "no sandbox"}

        data = []
        commands = ['echo hello', 'head -5 /etc/passwd', 'wc -l /etc/passwd']

        for cmd in commands:
            result = sandbox.execute(cmd, timeout=3)
            if result and result.stdout and result.stdout.strip():
                data.append({
                    "cmd": cmd,
                    "output": result.stdout.strip()[:500],
                })

        if not data:
            return {"success": True, "text_data": [], "summary": "no data found"}

        summary = f"{len(data)} commands executed"
        return {
            "success": True,
            "text_data": data,
            "summary": summary,
        }
    except Exception as e:
        return {"success": False, "text_data": [], "summary": str(e)}
