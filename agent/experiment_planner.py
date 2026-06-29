"""
ExperimentPlanner — P16 R3: 实验规划

将假设转为可在沙箱执行的实验:
  1. Passive observation: 读能同时看 A/B 的命令
  2. Active intervention: 改 A 看 B
  3. Counterfactual: WM 模拟预测

用法:
  planner = ExperimentPlanner()
  plan = planner.plan(hypothesis)
  result = sandbox.execute(plan["cmd"])
"""
import random
from typing import Optional


# 已知事实 → 读取命令的映射
READ_CMDS = {
    "cpu_cores": ["nproc", "grep -c ^processor /proc/cpuinfo", "lscpu | grep '^CPU(s)'"],
    "cpu_model": ["lscpu | grep 'Model name'", "cat /proc/cpuinfo | grep 'model name' | head -1"],
    "mem_total": ["free -h | grep Mem", "cat /proc/meminfo | grep MemTotal"],
    "mem_free": ["free -h | grep Mem | awk '{print $4}'"],
    "swap_total": ["free -h | grep Swap", "cat /proc/meminfo | grep SwapTotal"],
    "load": ["cat /proc/loadavg", "uptime"],
    "kernel": ["uname -r", "uname -a"],
    "kernel_release": ["uname -r"],
    "os_name": ["cat /etc/os-release 2>/dev/null | head -1", "lsb_release -d 2>/dev/null"],
    "os_version_id": ["cat /etc/os-release | grep VERSION_ID", "lsb_release -r 2>/dev/null"],
    "architecture": ["uname -m", "arch"],
    "hostname": ["hostname", "cat /etc/hostname"],
    "current_user": ["whoami", "id -un"],
    "users": ["who", "cat /etc/passwd | cut -d: -f1 | sort -u"],
    "uid_info": ["id"],
    "disk_root": ["df -h / | tail -1", "df -h / | awk 'NR==2{print $2,$3,$4}'"],
    "disk_persistent": ["df -h /workspace 2>/dev/null | tail -1", "df -h /data 2>/dev/null | tail -1"],
    "ip_addr": ["ip addr show 2>/dev/null | grep 'inet '", "ifconfig 2>/dev/null | grep 'inet '"],
    "mac_addr": ["ip link show 2>/dev/null | grep 'ether '", "ifconfig 2>/dev/null | grep ether"],
    "gateway": ["ip route | grep default", "route -n | grep '^0.0.0.0'"],
    "passwd_line_count": ["wc -l /etc/passwd", "cat /etc/passwd | wc -l"],
    "uptime_seconds": ["cat /proc/uptime | awk '{print $1}'"],
    "process_count": ["ps aux | wc -l", "ps -e --no-headers | wc -l"],
    "python_version": ["python3 --version 2>/dev/null", "python --version 2>/dev/null"],
    "n_packages": ["dpkg -l 2>/dev/null | wc -l", "pacman -Q 2>/dev/null | wc -l"],
}


class ExperimentPlanner:
    """将假设转为可执行的实验计划"""

    def __init__(self, sandbox=None, workbench=None):
        self.sandbox = sandbox
        self.workbench = workbench
        self._last_experiment_step = -10  # 频率控制

    def plan(self, hypothesis: dict) -> Optional[dict]:
        """
        从假设生成实验计划。

        返回:
          {cmd, timeout, type, hypothesis_key, predicted_exit, predicted_output_len, ...}
          或 None (无法生成)
        """
        src = hypothesis.get("if_node", "")
        dst = hypothesis.get("then_node", "")
        rel = hypothesis.get("rel", "")
        key = hypothesis.get("_key", f"{src}:{rel}:{dst}")

        # 1. 被动观察: 找能同时读 src 和 dst 的命令
        plan = self._passive_observation(src, dst)
        if plan:
            plan.update({"hypothesis_key": key, "type": "observe"})
            return plan

        # 2. 主动干预: 改 src 看 dst
        plan = self._active_intervention(src, dst)
        if plan:
            plan.update({"hypothesis_key": key, "type": "intervene"})
            return plan

        # 3. 回退: 读 dst 本身 (至少验证 dst 状态)
        dst_cmds = READ_CMDS.get(dst, [])
        if dst_cmds:
            cmd = random.choice(dst_cmds)
            return {
                "cmd": cmd,
                "timeout": 10,
                "type": "observe",
                "hypothesis_key": key,
                "predicted_exit": 0,
                "predicted_output_len": 50,
                "description": f"Observe {dst} (baseline for {src}→{dst})",
            }

        return None

    def _passive_observation(self, src: str, dst: str) -> Optional[dict]:
        """找能同时观察 src 和 dst 的命令"""
        src_cmds = READ_CMDS.get(src, [])
        dst_cmds = READ_CMDS.get(dst, [])

        # 如果 src 和 dst 有相同命令, 用那个
        common = [c for c in src_cmds if c in dst_cmds]
        if common:
            cmd = common[0]
        elif src_cmds and dst_cmds:
            # 组合: 先读 src 再读 dst
            cmd = f"{src_cmds[0]}; {dst_cmds[0]}"
        elif src_cmds:
            cmd = src_cmds[0]
        elif dst_cmds:
            cmd = dst_cmds[0]
        else:
            return None

        return {
            "cmd": cmd,
            "timeout": 10,
            "predicted_exit": 0,
            "predicted_output_len": 100,
            "description": f"Observe {src} and {dst}",
        }

    def _active_intervention(self, src: str, dst: str) -> Optional[dict]:
        """通过写 /workspace/ 来修改 src 状态"""
        # 只对可写的「类文件」事实做干预
        writable_keys = {"current_user": "echo 'test_user' > /workspace/test_user"}
        if src in writable_keys:
            write_cmd = writable_keys[src]
            dst_cmds = READ_CMDS.get(dst, [])
            if dst_cmds:
                cmd = f"{write_cmd} && {dst_cmds[0]}"
                return {
                    "cmd": cmd,
                    "timeout": 10,
                    "predicted_exit": 0,
                    "predicted_output_len": 80,
                    "description": f"Intervene on {src}, observe {dst}",
                }

        # 回退: 写一个标记文件然后读 dst
        dst_cmds = READ_CMDS.get(dst, [])
        if dst_cmds:
            cmd = f"echo 'test_{src}' > /workspace/exp_{src} && {dst_cmds[0]}"
            return {
                "cmd": cmd,
                "timeout": 10,
                "predicted_exit": 0,
                "predicted_output_len": 80,
                "description": f"Write marker for {src}, read {dst}",
            }
        return None

    def execute_plan(self, plan: dict, sandbox) -> dict:
        """执行实验计划, 返回结果"""
        if sandbox is None:
            return {"success": False, "error": "no sandbox"}

        cmd = plan.get("cmd", "")
        timeout = plan.get("timeout", 10)

        # 安全约束: 只允许在 /workspace/ 内写操作
        if ">" in cmd or ">>" in cmd:
            # 检查写入路径
            parts = cmd.split(">")[1] if ">" in cmd else cmd.split(">>")[1]
            if not any(safe in parts for safe in ["/workspace/", "/tmp/"]):
                return {"success": False, "error": "unsafe write path"}

        try:
            result = sandbox.execute(cmd, timeout=timeout)
            return {
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": (result.stdout or ""),
                "stderr": (result.stderr or ""),
                "output_len": len((result.stdout or "") + (result.stderr or "")),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
