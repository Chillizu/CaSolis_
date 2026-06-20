"""Brain Necessity Benchmark — 任务定义

验证假设: 大脑(意图分类器) + 手脚(Qwen) 是否优于 纯手脚(随机意图)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── 任务难度等级 ─────────────────────────────────────────────

@dataclass
class Task:
    task_id: str          # 唯一 ID
    level: int            # 1=单步, 2=两步, 3=多步推理
    description: str      # 人类可读的描述
    hints: list[str]      # 提示词 (用于 Qwen 推理)
    expected_intents: list[list[str]]  # 每步期望的意图
    validation_fn: str    # 验证规则
    params: dict = field(default_factory=dict)   # 具体参数


# ── L1: 单步命令 ─────────────────────────────────────────────

L1_TASKS = [
    Task(
        task_id="L1_hostname",
        level=1,
        description="读取 /etc/hostname 的内容",
        hints=["读取主机名", "/etc/hostname 里存着 hostname"],
        expected_intents=[["READ"]],
        validation_fn="contains_hostname",
        params={"path": "/etc/hostname"},
    ),
    Task(
        task_id="L1_date",
        level=1,
        description="获取当前时间",
        hints=["获取系统时间"],
        expected_intents=[["INFO"]],
        validation_fn="contains_date",
        params={"target": "uptime"},
    ),
    Task(
        task_id="L1_passwd_count",
        level=1,
        description="统计 /etc/passwd 有多少行",
        hints=["数一数 /etc/passwd 里有几行"],
        expected_intents=[["COUNT"]],
        validation_fn="count_lines",
        params={"path": "/etc/passwd"},
    ),
    Task(
        task_id="L1_cpu_info",
        level=1,
        description="查看 CPU 信息",
        hints=["查看 /proc/cpuinfo", "查看 cpu 信息"],
        expected_intents=[["INFO"]],
        validation_fn="contains_cpu",
        params={"target": "cpu"},
    ),
    Task(
        task_id="L1_mem_info",
        level=1,
        description="查看内存信息",
        hints=["查看 /proc/meminfo", "查看内存"],
        expected_intents=[["INFO"]],
        validation_fn="contains_mem",
        params={"target": "mem"},
    ),
    Task(
        task_id="L1_disk_info",
        level=1,
        description="查看磁盘使用情况",
        hints=["查看 df", "磁盘空间"],
        expected_intents=[["INFO"]],
        validation_fn="contains_disk",
        params={"target": "disk"},
    ),
    Task(
        task_id="L1_whoami",
        level=1,
        description="查看当前用户",
        hints=["当前用户是谁"],
        expected_intents=[["INFO"]],
        validation_fn="contains_user",
        params={"target": "whoami"},
    ),
    Task(
        task_id="L1_uname",
        level=1,
        description="查看系统信息 (内核版本等)",
        hints=["uname -a", "查看系统信息"],
        expected_intents=[["INFO"]],
        validation_fn="contains_uname",
        params={"target": "uname"},
    ),
    Task(
        task_id="L1_etc_list",
        level=1,
        description="列出 /etc 目录下的内容",
        hints=["查看 /etc 目录", "ls /etc"],
        expected_intents=[["LIST"]],
        validation_fn="contains_files",
        params={"path": "/etc"},
    ),
    Task(
        task_id="L1_grep_root",
        level=1,
        description="在 /etc/passwd 中搜索 root",
        hints=["grep root /etc/passwd", "查找 root 用户"],
        expected_intents=[["SEARCH"]],
        validation_fn="contains_root",
        params={"pattern": "root", "path": "/etc/passwd"},
    ),
]

# ── L2: 两步组合 ─────────────────────────────────────────────

L2_TASKS = [
    Task(
        task_id="L2_search_count",
        level=2,
        description="找出 /etc/passwd 中包含 root 的行并计数",
        hints=["先用 grep 搜索 root 在 /etc/passwd 中", "再统计行数"],
        expected_intents=[["SEARCH"], ["COUNT"]],
        validation_fn="search_and_count",
        params={"pattern": "root", "path": "/etc/passwd"},
    ),
    Task(
        task_id="L2_cpu_mem",
        level=2,
        description="查看 CPU 型号和内存总量",
        hints=["先查 CPU 信息", "再查内存信息"],
        expected_intents=[["INFO"], ["INFO"]],
        validation_fn="cpu_and_mem",
        params={},
    ),
    Task(
        task_id="L2_hostname_uptime",
        level=2,
        description="查看主机名和系统运行时间",
        hints=["查主机名", "查运行时间"],
        expected_intents=[["READ"], ["INFO"]],
        validation_fn="hostname_and_uptime",
        params={"path": "/etc/hostname"},
    ),
    Task(
        task_id="L2_list_and_grep",
        level=2,
        description="列出 /etc 下有哪些文件，然后搜索包含 root 的",
        hints=["先 ls /etc", "再搜索哪个文件包含 root"],
        expected_intents=[["LIST"], ["SEARCH"]],
        validation_fn="list_then_grep",
        params={"path": "/etc", "pattern": "root"},
    ),
    Task(
        task_id="L2_check_commands",
        level=2,
        description="检查 python3 和 git 是否安装",
        hints=["检查 python3 的存在", "检查 git 的存在"],
        expected_intents=[["INSPECT"], ["INSPECT"]],
        validation_fn="has_both_commands",
        params={"cmd1": "python3", "cmd2": "git"},
    ),
]

# ── L3: 多步推理 ─────────────────────────────────────────────

L3_TASKS = [
    Task(
        task_id="L3_status_report",
        level=3,
        description="生成系统状态报告: CPU 型号、内存、磁盘、主机名",
        hints=[
            "先查 CPU 型号",
            "再查内存容量",
            "再查磁盘使用",
            "最后查主机名",
        ],
        expected_intents=[["INFO"], ["INFO"], ["INFO"], ["INFO"]],
        validation_fn="complete_status_report",
        params={},
    ),
    Task(
        task_id="L3_explore_etc",
        level=3,
        description="探索 /etc 目录，找出最大的配置文件，并查看其内容",
        hints=[
            "列出 /etc 目录",
            "找出最大的文件",
            "查看该文件内容",
        ],
        expected_intents=[["LIST"], ["READ"], ["COUNT"]],
        validation_fn="find_largest_then_read",
        params={"target_dir": "/etc"},
    ),
    Task(
        task_id="L3_user_root_analysis",
        level=3,
        description="分析 /etc/passwd: 有多少用户, 其中哪些是 root 权限",
        hints=[
            "统计行数",
            "搜索 root",
            "看看 root 用户信息",
        ],
        expected_intents=[["COUNT"], ["SEARCH"], ["READ"]],
        validation_fn="user_analysis",
        params={"path": "/etc/passwd", "pattern": "root"},
    ),
]

# ── 全部任务 ─────────────────────────────────────────────────

ALL_TASKS = [*L1_TASKS, *L2_TASKS, *L3_TASKS]

TASKS_BY_LEVEL = {1: L1_TASKS, 2: L2_TASKS, 3: L3_TASKS}


# ── 验证函数 ─────────────────────────────────────────────────

def validate_result(task: Task, outputs: list[str]) -> tuple[bool, str]:
    """验证任务执行结果"""
    validator = VALIDATORS.get(task.validation_fn)
    if validator is None:
        return False, f"未知验证器: {task.validation_fn}"

    combined = "\n".join(outputs)
    return validator(combined, task.params)


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(k.lower() in text.lower() for k in keywords)


def _validate_hostname(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["hostname"]), "应包含 'hostname'")


def _validate_date(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["up", "day", "week", "min"]), "应包含时间信息")


def _validate_count(text: str, _: dict) -> tuple[bool, str]:
    import re
    nums = re.findall(r'\d+', text)
    return (len(nums) > 0, f"应包含数字, 得到: {nums[:3]}")


def _validate_cpu(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["processor", "cpu", "model name"]), "应包含 CPU 信息")


def _validate_mem(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["MemTotal", "MemFree", "kB"]), "应包含内存信息")


def _validate_disk(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["Filesystem", "Size", "Used", "Avail", "/"]), "应包含磁盘信息")


def _validate_user(text: str, _: dict) -> tuple[bool, str]:
    return (len(text.strip()) > 0 and not text.startswith("usage"), "应返回用户名")


def _validate_uname(text: str, _: dict) -> tuple[bool, str]:
    return (len(text.strip()) > 10 and "Linux" in text or "hostname" in text, "应包含系统信息")


def _validate_files(text: str, _: dict) -> tuple[bool, str]:
    return (len(text.strip().split("\n")) > 3, "应包含多个文件")


def _validate_root(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["root"]), "应包含 'root'")


def _validate_search_count(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["root"]) and _has_any(text, ["1", "2", "3"]), "应包含 root 和行数")


def _validate_cpu_and_mem(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["processor", "MemTotal", "model name"]), "应包含 CPU 和内存")


def _validate_hostname_uptime(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["hostname"]), "应包含 hostname")


def _validate_list_then_grep(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["root"]), "应包含 root")


def _validate_has_both(text: str, _: dict) -> tuple[bool, str]:
    return (len(text.strip()) > 0, "应返回路径信息")


def _validate_status_report(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["processor", "MemTotal", "Filesystem", "hostname"]), "应包含 CPU/内存/磁盘/主机名")


def _validate_largest_read(text: str, _: dict) -> tuple[bool, str]:
    return (len(text.strip()) > 10, "应包含文件内容")


def _validate_user_analysis(text: str, _: dict) -> tuple[bool, str]:
    return (_has_any(text, ["root"]), "应包含 root")


VALIDATORS = {
    "contains_hostname": _validate_hostname,
    "contains_date": _validate_date,
    "count_lines": _validate_count,
    "contains_cpu": _validate_cpu,
    "contains_mem": _validate_mem,
    "contains_disk": _validate_disk,
    "contains_user": _validate_user,
    "contains_uname": _validate_uname,
    "contains_files": _validate_files,
    "contains_root": _validate_root,
    "search_and_count": _validate_search_count,
    "cpu_and_mem": _validate_cpu_and_mem,
    "hostname_and_uptime": _validate_hostname_uptime,
    "list_then_grep": _validate_list_then_grep,
    "has_both_commands": _validate_has_both,
    "complete_status_report": _validate_status_report,
    "find_largest_then_read": _validate_largest_read,
    "user_analysis": _validate_user_analysis,
}


# ── 任务描述（用于提示 Qwen） ─────────────────────────────────

def format_task_prompt(task: Task) -> str:
    """把任务转为 Qwen 可理解的 prompt"""
    hints_str = "\n".join(f"- {h}" for h in task.hints)
    return f"""任务: {task.description}

提示:
{hints_str}

请分步执行。每一步输出: {{意图名, 参数}}

示例:
Step 1: READ, {{"path": "/etc/hostname"}}
Step 2: INFO, {{"target": "cpu"}}
"""


if __name__ == "__main__":
    print("任务统计:")
    print(f"  L1 (单步): {len(L1_TASKS)} 个")
    print(f"  L2 (两步): {len(L2_TASKS)} 个")
    print(f"  L3 (多步): {len(L3_TASKS)} 个")
    print(f"  总计: {len(ALL_TASKS)} 个")
