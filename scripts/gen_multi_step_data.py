"""
生成多步训练样本 (扩充版)

用排列组合生成大量合理的两步序列:
  Step 1: intent A, state_text with history='无'
  Step 2: intent B, state_text with history='[前一步结果]'
"""

import json
import random

random.seed(42)

INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP"]
INTENT_MAP = {n: i for i, n in enumerate(INTENTS)}

# 参数池
DIRS = ["/", "/etc", "/var/log", "/proc", "/tmp", "/home", "/opt", "/usr/bin"]
FILES = ["passwd", "hostname", "hosts", "syslog", "cpuinfo", "meminfo", "shadow", "group", "fstab", "services", "shells", "environment"]

# 每类意图的 "上步描述" 模板
INTENT_DESCS = {
    "READ": [
        "查看 /etc/{file} 的内容", "读取 /etc/{file}", "打开 /etc/{file} 看看",
        "阅读 /etc/{file} 文件", "显示 /etc/{file} 的内容",
    ],
    "LIST": [
        "列出 {dir} 目录", "看看 {dir} 里有什么", "查看 {dir} 的内容",
        "浏览 {dir} 中的所有文件", "列举 {dir} 的子目录",
    ],
    "SEARCH": [
        "在 /etc/{file} 中搜索 {kw}", "找出包含 {kw} 的行",
        "过滤 /etc/{file} 中的 {kw}", "从 {file} 中查找 {kw}",
        "在 {dir}/{file} 里搜索 {kw}",
    ],
    "INFO": [
        "查看 CPU 信息", "查看内存信息", "查看磁盘使用情况",
        "获取当前时间", "查看当前用户", "查看系统信息",
        "查看主机名", "查看运行时间", "查看 CPU 核心数",
    ],
    "INSPECT": [
        "检查 {cmd} 是否安装", "确认 {cmd} 是否存在",
        "查看 {cmd} 命令能不能用", "检查 {cmd} 的安装状态",
    ],
    "COUNT": [
        "统计 /etc/{file} 有多少行", "数一数 /etc/{file} 有几行",
        "计算 {file} 的行数", "统计文件的行数",
    ],
    "EXPLORE": [
        "探索 {dir} 目录", "看看 {dir} 目录里有什么",
        "浏览 {dir} 看看有没有有趣的内容",
    ],
    "HELP": [
        "查看 {cmd} 的帮助文档", "了解 {cmd} 的用法",
        "查看 {cmd} 有哪些选项", "查阅 {cmd} 的手册",
    ],
}

KW = ["root", "admin", "error", "failed", "nobody", "warning"]
CMDS = ["python3", "git", "docker", "curl", "node", "gcc", "java", "nginx", "ssh", "vim", "ruby", "perl", "make", "ffmpeg"]

# 前一步结果摘要 (按意图区分)
STEP1_RESULTS = {
    "READ": "前一步输出了文件内容, 共 {n} 行",
    "LIST": "前一步列出了目录列表, 共 {n} 项",
    "SEARCH": "前一步搜索到了 {n} 行匹配结果",
    "INFO": "前一步输出了系统信息",
    "INSPECT": "前一步确认了命令存在",
    "COUNT": "前一步统计了行数: {n} 行",
    "EXPLORE": "前一步浏览了目录结构",
    "HELP": "前一步显示了帮助信息",
}

# 合理意图转移矩阵 (从 intent A 到 intent B 是否合理)
# 0=不合理, 1=可行, 2=推荐
TRANSITION_MATRIX = {
    "READ":    {"READ": 0, "LIST": 0, "SEARCH": 1, "INFO": 0, "INSPECT": 0, "COUNT": 2, "EXPLORE": 0, "HELP": 0},
    "LIST":    {"READ": 2, "LIST": 0, "SEARCH": 2, "INFO": 1, "INSPECT": 0, "COUNT": 2, "EXPLORE": 0, "HELP": 0},
    "SEARCH":  {"READ": 0, "LIST": 0, "SEARCH": 2, "INFO": 0, "INSPECT": 0, "COUNT": 2, "EXPLORE": 0, "HELP": 0},
    "INFO":    {"READ": 0, "LIST": 1, "SEARCH": 0, "INFO": 2, "INSPECT": 0, "COUNT": 0, "EXPLORE": 1, "HELP": 0},
    "INSPECT": {"READ": 0, "LIST": 0, "SEARCH": 0, "INFO": 0, "INSPECT": 0, "COUNT": 0, "EXPLORE": 0, "HELP": 2},
    "COUNT":   {"READ": 0, "LIST": 0, "SEARCH": 1, "INFO": 0, "INSPECT": 0, "COUNT": 0, "EXPLORE": 0, "HELP": 0},
    "EXPLORE": {"READ": 1, "LIST": 2, "SEARCH": 1, "INFO": 2, "INSPECT": 0, "COUNT": 0, "EXPLORE": 0, "HELP": 0},
    "HELP":    {"READ": 0, "LIST": 2, "SEARCH": 0, "INFO": 0, "INSPECT": 0, "COUNT": 0, "EXPLORE": 0, "HELP": 0},
}


def fill_template(template: str, **kwargs) -> str:
    """填充模板变量"""
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def generate_single_step(intent: str, desc_templates: list[str]) -> tuple[str, str, str, str]:
    """生成一条单步状态文本, 返回 (intent, state_text, dir, file)"""
    dir_path = random.choice(DIRS)
    known_file = random.choice(FILES)
    desc = fill_template(random.choice(desc_templates), file=known_file, dir=dir_path, cmd=random.choice(CMDS), kw=random.choice(KW))
    return intent, desc, dir_path, known_file


def generate_multi_step(n_per_class: int = 50):
    samples = []

    for intent_a in INTENTS:
        for intent_b in INTENTS:
            weight = TRANSITION_MATRIX.get(intent_a, {}).get(intent_b, 0)
            if weight == 0:
                continue
            # 每对生成 (weight * 3) 条
            count = weight * 3 + random.randint(0, 2)
            for _ in range(count):
                dir_path = random.choice(DIRS)
                known_file = random.choice(FILES)
                
                # Step 1
                desc_a = fill_template(random.choice(INTENT_DESCS[intent_a]), file=known_file, dir=dir_path, cmd=random.choice(CMDS), kw=random.choice(KW))
                state_1 = f"当前目录: {dir_path} 已知文件: {known_file} 上步: {desc_a} 历史: 无"
                samples.append({"source": "multistep", "state_text": state_1, "intent": intent_a, "intent_id": INTENT_MAP[intent_a], "step": 1})

                # Step 2
                desc_b = fill_template(random.choice(INTENT_DESCS[intent_b]), file=known_file, dir=dir_path, cmd=random.choice(CMDS), kw=random.choice(KW))
                n_lines = random.randint(1, 100)
                hist = fill_template(STEP1_RESULTS.get(intent_a, "前一步操作完成"), n=n_lines)
                state_2 = f"当前目录: {dir_path} 已知文件: {known_file} 上步: {desc_b} 历史: {hist}"
                samples.append({"source": "multistep", "state_text": state_2, "intent": intent_b, "intent_id": INTENT_MAP[intent_b], "step": 2})

    random.shuffle(samples)

    # 每类截取 n_per_class 条
    final = []
    per_class = {n: [] for n in INTENTS}
    for s in samples:
        per_class[s["intent"]].append(s)
    for n in INTENTS:
        sampled = per_class[n][:n_per_class]
        final.extend(sampled)
        print(f"  {n:10s}: {len(per_class[n]):4d} → {len(sampled):4d} 条")

    random.shuffle(final)
    return final


def main():
    print("生成多步训练数据...")
    samples = generate_multi_step(n_per_class=50)
    
    from collections import Counter
    dist = Counter(s['intent'] for s in samples)
    print(f"\n总计: {len(samples)} 条")
    step1 = sum(1 for s in samples if s['step']==1)
    step2 = sum(1 for s in samples if s['step']==2)
    print(f"  步1: {step1} 条  步2: {step2} 条")

    output_path = "data/intent_multistep.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n输出: {output_path}")
    print(f"\n前 4 条:")
    for s in samples[:4]:
        print(f"  step={s['step']} [{s['intent']}] {s['state_text'][:80]}")

    return samples


if __name__ == "__main__":
    main()
