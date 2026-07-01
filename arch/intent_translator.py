"""
意图翻译器 — 将模型输出的「意图」转为可执行 shell 命令

核心思路:
  模型不生成命令字符串，生成「意图」
  翻译器将意图转为精确的 bash 命令
  创造力 = 意图的组合，不是字符串的排列
"""

# 意图定义
INTENTS = {
    # 基本操作
    "READ_FILE":     "cat {path} 2>/dev/null",           # 读文件
    "LIST_DIR":      "ls {flags} {path} 2>/dev/null",    # 列目录
    "PEEK_FILE":     "head -{n} {path} 2>/dev/null",     # 看文件头部
    "TAIL_FILE":     "tail -{n} {path} 2>/dev/null",     # 看文件尾部
    
    # 系统信息
    "CPU_INFO":      "cat /proc/cpuinfo 2>/dev/null | head -{n}",
    "MEM_INFO":      "cat /proc/meminfo 2>/dev/null | head -{n}",
    "DISK_INFO":     "df -h / 2>/dev/null",
    "USER_INFO":     "whoami 2>/dev/null; id 2>/dev/null",
    "PROCESS_INFO":  "ps aux 2>/dev/null | head -{n}",
    "SYSTEM_INFO":   "uname -a 2>/dev/null",
    "UPTIME":        "uptime 2>/dev/null",
    "DATE_INFO":     "date 2>/dev/null",
    
    # 管道操作
    "GREP":          "grep '{pat}' {path} 2>/dev/null | head -{n}",
    "COUNT":         "wc -l {path} 2>/dev/null",
    "SORT":          "sort {path} 2>/dev/null | head -{n}",
    "FILTER_COL":    "cut -d{delim} -f{fld} {path} 2>/dev/null | head -{n}",
    
    # 探索
    "WHICH":         "which {cmd} 2>/dev/null",
    "HELP":          "{cmd} --help 2>/dev/null | head -{n}",
    "VERSION":       "{cmd} --version 2>/dev/null | head -{n}",
    
    # 组合
    "PIPE":          "{cmd1} | {cmd2}",                  # 手动管道组合
}

# 意图 → 特殊 token ID 映射
# 这些 token 在 BPE 词表之外（从 TV = 2022+19 开始）
INTENT_NAMES = list(INTENTS.keys())
N_INTENTS = len(INTENT_NAMES)  # 20
INTENT_TOKEN_START = 1500  # 在 BPE 词表内留空间（< V = 2022）

# 为每个意图分配一个 token ID
intent_token_map = {}
for i, name in enumerate(INTENT_NAMES):
    intent_token_map[name] = INTENT_TOKEN_START + i


def parse_intent(text):
    """
    从文本中解析意图（训练时用）
    
    输入: "[THOUGHT] check cpu\n[CMD] cat /proc/cpuinfo | head -5\n[OBS] processor: 0"
    输出: ("READ_PROC", {"path": "cpuinfo", "n": "5"})
    """
    # 提取 CMD 部分
    cmd_start = text.find("[CMD]")
    obs_start = text.find("[OBS]")

    if cmd_start < 0:
        return None

    if obs_start >= 0:
        cmd = text[cmd_start+5:obs_start].strip()
    else:
        cmd = text[cmd_start+5:].strip()

    # 尝试匹配意图
    cmd_lower = cmd.lower()

    match = _match_intent(cmd_lower, cmd)
    if match:
        return match

    return ("UNKNOWN", {"cmd": cmd})


def _match_intent(cmd_lower, cmd_orig):
    """匹配命令到意图"""

    # READ_FILE
    if cmd_lower.startswith("cat "):
        path = cmd_orig[4:].strip()
        parts = path.split("|")
        if len(parts) > 1:
            path = parts[0].strip()
            rest = "|".join(parts[1:])
            if "grep" in rest or "filter" in rest:
                return ("FILTER_COL", {"path": path, "delim": ":", "fld": "1", "n": "10"})
        if "/proc/" in path:
            return ("PEEK_FILE", {"path": path, "n": "10"})
        return ("READ_FILE", {"path": path})

    # LIST_DIR
    if cmd_lower.startswith("ls "):
        parts = cmd_orig[3:].strip().split()
        flags = ""
        path = ""
        for p in parts:
            if p.startswith("-"):
                flags += p
            else:
                path = p
        if not path:
            path = "/"
        return ("LIST_DIR", {"flags": flags or "", "path": path})

    # PEEK_FILE (head)
    if cmd_lower.startswith("head "):
        params = cmd_orig[5:].strip()
        if params.startswith("-"):
            n = params[1:].split()[0]
            path = params[len("-"+n):].strip()
        else:
            n = "10"
            path = params
        return ("PEEK_FILE", {"path": path, "n": n})

    # WHICH
    if cmd_lower.startswith("which "):
        return ("WHICH", {"cmd": cmd_orig[6:].strip()})

    # SYSTEM_INFO
    if cmd_lower.startswith("uname"):
        return ("SYSTEM_INFO", {})
    if cmd_lower.startswith("whoami"):
        return ("USER_INFO", {})
    if cmd_lower.startswith("id"):
        return ("USER_INFO", {})
    if cmd_lower.startswith("uptime"):
        return ("UPTIME", {})
    if cmd_lower.startswith("date"):
        return ("DATE_INFO", {})
    if cmd_lower.startswith("df "):
        return ("DISK_INFO", {})
    if cmd_lower.startswith("free") or cmd_lower.startswith("cat /proc/meminfo"):
        return ("MEM_INFO", {"n": "10"})
    if "cpuinfo" in cmd_lower:
        return ("CPU_INFO", {"n": "10"})

    # PROCESS
    if cmd_lower.startswith("ps ") or "ps aux" in cmd_lower:
        return ("PROCESS_INFO", {"n": "10"})

    # COUNT
    if cmd_lower.startswith("wc "):
        path = cmd_orig[3:].strip()
        for flag in ["-l", "-w", "-c"]:
            path = path.replace(flag, "").strip()
        return ("COUNT", {"path": path or "/tmp"})

    # GREP
    if cmd_lower.startswith("grep "):
        rest = cmd_orig[5:].strip()
        if "|" in rest:
            # pipe: grep pattern
            return ("GREP", {"pat": rest.split("|")[0].strip(), "path": "/tmp", "n": "10"})
        parts = rest.split()
        if len(parts) >= 2:
            return ("GREP", {"pat": parts[0], "path": parts[1], "n": "10"})

    # HELP / VERSION
    if cmd_lower.endswith("--help"):
        cmd_name = cmd_lower[:-7].strip()
        return ("HELP", {"cmd": cmd_name, "n": "20"})
    if cmd_lower.endswith("--version"):
        cmd_name = cmd_lower[:-10].strip()
        return ("VERSION", {"cmd": cmd_name, "n": "10"})

    return None


def intent_to_command(intent_name, params):
    """把意图转回 shell 命令"""
    template = INTENTS.get(intent_name)
    if template is None:
        return params.get("cmd", "")

    try:
        return template.format(**params)
    except KeyError:
        return params.get("cmd", "")


def format_intent_text(text):
    """
    把标准训练数据转成意图格式
    
    "[THOUGHT] ...\n[CMD] cat /proc/cpuinfo\n[OBS] ..."
    →
    "[THOUGHT] ...\n[INTENT] PEEK_FILE path=/proc/cpuinfo n=10\n[OBS] ..."
    """
    result = parse_intent(text)
    if result is None:
        return None

    intent_name, params = result
    param_str = " ".join(f"{k}={v}" for k, v in params.items())
    new_text = text.replace("[CMD]", f"[INTENT] {intent_name}")
    cmd_start = new_text.find("[INTENT]")
    obs_start = new_text.find("[OBS]")
    if cmd_start >= 0 and obs_start >= 0:
        new = new_text[:cmd_start]
        new += f"[INTENT] {intent_name} {param_str}"
        new += new_text[obs_start:]
        return new

    return new_text


def test():
    """测试"""
    test_cmds = [
        "cat /proc/cpuinfo",
        "ls -la /tmp",
        "head -5 /etc/passwd",
        "which python3",
        "whoami",
        "grep root /etc/passwd",
        "uname -a",
        "wc -l /etc/passwd",
        "cat /proc/meminfo | head -10",
    ]

    for cmd in test_cmds:
        text = f"[THOUGHT] test\n[CMD] {cmd}\n[OBS] output here"
        result = parse_intent(text)
        if result:
            intent_name, params = result
            back = intent_to_command(intent_name, params)
            print(f"  {cmd:35s} → {intent_name:15s} → {back}")
        else:
            print(f"  {cmd:35s} → UNMATCHED")

    # Test full format
    print()
    full = "[THOUGHT] checking\n[CMD] cat /proc/cpuinfo | head -5\n[OBS] processor: 0"
    conv = format_intent_text(full)
    print(f"原格式: {full}")
    print(f"转意图: {conv}")


if __name__ == "__main__":
    test()
