"""
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
            for line in r.stdout.strip().split("\n"):
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
            for line in r2.stdout.strip().split("\n"):
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
