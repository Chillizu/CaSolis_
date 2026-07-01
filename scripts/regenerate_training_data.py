"""
P3: 训练数据再生 — 将旧格式 state_text 升级为 P3 格式

旧格式: 当前目录: /etc 已知文件: passwd 上步: 读取 /etc/passwd 历史: 无
新格式: 当前目录: /etc 已知文件: passwd 上步: 读取 /etc/passwd 输出: ... 工作区: 无 历史: ...

并为新意图生成训练数据。
"""

import json, os, sys, copy, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

# 意图 → 典型输出摘要
INTENT_OUTPUT_MAP = {
    "READ":      "文件内容",
    "LIST":      "目录列表",
    "SEARCH":    "搜索结果: 匹配",
    "INFO":      "系统信息",
    "COUNT":     "行数统计",
    "INSPECT":   "/bin/命令",
    "EXPLORE":   "目录内容",
    "HELP":      "帮助信息",
    "READ_ETC":  "配置文件内容",
    "USB_DEVICES":"设备列表", 
    "DISK_USAGE":"磁盘用量",
    "LS_TMP":    "临时文件列表",
    "ARCH_INFO": "x86_64",
    "CUSTOM":    "命令输出",
}

def upgrade_state_text(old_text: str, intent: str) -> str:
    """向旧 state_text 添加 P3 字段"""
    output_summary = INTENT_OUTPUT_MAP.get(intent, "命令输出")
    # 如果已有 P3 字段, 不重复添加
    if "输出:" in old_text:
        return old_text
    # 在 历史: 前面插入 输出: 和 工作区:
    if " 历史: " in old_text:
        before, after = old_text.split(" 历史: ", 1)
        return f"{before} 输出: {output_summary} 工作区: 无 历史: {after}"
    else:
        return f"{old_text} 输出: {output_summary} 工作区: 无 历史: 无"


def generate_variant(state_text: str, intent: str) -> list[str]:
    """生成同一意图的语义变体 (不改变意图含义)"""
    variants = []
    # 只改变 输出: 字段, 保持其他不变
    output_variants = {
        "READ":      ["文件内容", "读取结果", "文本内容", "文件数据"],
        "LIST":      ["目录列表", "文件列表", "目录内容"],
        "SEARCH":    ["匹配结果: root", "0 匹配行", "找到 1 行"],
        "INFO":      ["CPU 信息", "内存信息", "系统版本", "运行时间"],
        "INSPECT":   ["/bin/bash", "not found", "命令已安装"],
        "CUSTOM":    ["命令输出", "cmd 结果", "stdout 返回"],
    }
    outs = output_variants.get(intent, ["命令输出"])
    current_out = INTENT_OUTPUT_MAP.get(intent, "命令输出")
    for out in outs:
        if out != current_out:
            variants.append(state_text.replace(f"输出: {current_out}", f"输出: {out}"))
    return variants


def main():
    src = "data/intent_train_v3.jsonl"
    dst = "data/intent_train_p3.jsonl"
    
    with open(src) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    print(f"原数据: {len(rows)} 条")
    
    # 升级到 P3 格式 + 扩展变体
    new_rows = []
    seen_texts = set()
    
    # 先确定有效的意图集
    from agent.conductor import INTENTS
    valid_intents = set(INTENTS[:13])  # 不含 CUSTOM
    
    for r in rows:
        intent = r.get("intent", "")
        text = r.get("state_text", "")
        
        if intent not in valid_intents and intent != "CUSTOM":
            continue
        
        # 升级主版本
        upgraded = upgrade_state_text(text, intent)
        if upgraded not in seen_texts:
            seen_texts.add(upgraded)
            r2 = copy.deepcopy(r)
            r2["state_text"] = upgraded
            new_rows.append(r2)
        
        # 添加输出变体 (数据增强)
        for variant in generate_variant(text, intent):
            if variant not in seen_texts:
                seen_texts.add(variant)
                r3 = copy.deepcopy(r)
                r3["state_text"] = variant
                new_rows.append(r3)
    
    # 额外生成新意图数据 (LS_TMP, ARCH_INFO)
    base_ls = [r for r in new_rows if r["intent"] == "LIST"]
    base_info = [r for r in new_rows if r["intent"] == "INFO"]
    
    extra_count = 0
    for r in base_ls[:150]:
        r2 = copy.deepcopy(r)
        r2["intent"] = "LS_TMP"
        r2["state_text"] = r["state_text"].replace(
            "输出: 目录列表", "输出: 临时文件列表"
        ).replace("输出: 文件列表", "输出: 临时文件列表")
        if r2["state_text"] not in seen_texts:
            seen_texts.add(r2["state_text"])
            new_rows.append(r2)
            extra_count += 1
    
    for r in base_info[:150]:
        r2 = copy.deepcopy(r)
        r2["intent"] = "ARCH_INFO"
        r2["state_text"] = r["state_text"].replace(
            "输出: 系统信息", "输出: x86_64"
        )
        if r2["state_text"] not in seen_texts:
            seen_texts.add(r2["state_text"])
            new_rows.append(r2)
            extra_count += 1
    
    # 保存
    with open(dst, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"P3 格式: {len(new_rows)} 条")
    print(f"  新意图数据: {extra_count}")
    for intent, cnt in Counter(r["intent"] for r in new_rows).most_common():
        print(f"  {intent:15s} {cnt}")


if __name__ == "__main__":
    main()
