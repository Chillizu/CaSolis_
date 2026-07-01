# Folunar_ 创意能力升级方案评估

> 评估时间: 2026-06-24  
> 评估范围: INTENTS 扩展、奖励函数、content 参数、低风险先行方案

---

## 1. 结论速览

| 问题 | 结论 |
|------|------|
| Q1 加入 WRITE/APPEND/GENERATE | **合理, 但要分阶段**。先不扩展意图维度, 用 CUSTOM + 工作栏目标让系统学会"创造"; 稳定后再把 WRITE/GENERATE 提升为一级意图。 |
| Q2 创造奖励 | **必须加意图专属奖励**, 但不能只给固定值。建议: 成功写(+0.3) + 字节奖励(+0.1/100B, 上限+0.5) + 新文件加成(+0.2) - 重复覆盖惩罚。 |
| Q3 content 来源 | 从 **Workbench.facts** 自动生成。最佳方案: 用 python3 把事实 JSON 化后写文件, 避免 shell 注入。 |
| Q4 更简单启动创造 | **先做 3 件小事**: (1) 脚本输出改到 `/workspace/scripts/`; (2) CUSTOM 池加入 `python3 -c` 写文件命令; (3) 工作栏自生成目标增加"写摘要"。 |

---

## 2. Q1: 把 WRITE/APPEND 加入 INTENTS 是否合理?

### 2.1 当前状态

当前 14 个意图全部为"读"类:

```python
# agent/nanny.py:18, agent/conductor.py:16, agent/online_agent.py:59
INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP",
           "READ_ETC", "USB_DEVICES", "DISK_USAGE", "LS_TMP", "ARCH_INFO", "CUSTOM"]
```

`command_registry.json` 里已经有 `WRITE`/`APPEND`/`CAT` 模板, 但 `param_rules.json` 没有对应规则, 所以它们永远不会被分类器选中。

### 2.2 加入后会触发的改动链

如果直接扩展到 17 个意图, 影响面如下:

| 组件 | 当前 | 改动后 | 是否已经支持扩展 |
|------|------|--------|------------------|
| `INTENTS` 列表 | 14 | 17 | 需要改 3 处 |
| `IntentClassifier` 输出层 | `nn.Linear(128, 14)` | `nn.Linear(128, 17)` | 有 `expand_intents()` 方法 |
| `ConductorHead.class_proj` | `nn.Linear(64, 14)` | `nn.Linear(64, 17)` | 有 `expand_intents()` 方法 |
| `WorldModelNetV3` 输入 | `384 + 16 + 14` | `384 + 16 + 17` | 有 `expand_intents()` 方法 |
| 训练数据 | `data/intent_train_v3.jsonl` 没有 WRITE 样本 | 需要新样本 | 无样本 |
| 奖励函数 | 14 类的 base reward | 需要新增 3 类 | 需改 `_compute_reward()` |
| 参数提取 | `param_rules.json` 无规则 | 需新增规则 | 无 |
| 命令执行 | 模板已存在 | 需要 content 来源 | 部分有 |

### 2.3 评估结论

**合理, 但不建议一步到位。**

- **合理性**: 系统已经卡在"反复 hostname"的局部最优, 引入写操作是打破只读循环的自然下一步。
- **风险**: 意图维度扩展后, 训练数据里 WRITE/APPEND/GENERATE 的样本为 0, 分类器会随机输出这些类, 而奖励函数目前对写操作不友好, 新维度会被迅速压制。
- **建议策略**: 先让系统在 `CUSTOM` 路径下学会写文件(P4 方案), 产生真实 WRITE 轨迹后, 再把这些意图提升为一级意图。

---

## 3. Q2: 创造奖励怎么设计?

### 3.1 当前奖励函数的问题

当前 `_compute_reward()` 在 `agent/online_agent.py` 约 383 行:

```python
reward = base_reward[intent] + (0.3 if new_fact else 0) + novelty * 0.3
```

- `new_fact` 由 `Workbench.extract_facts()` 判断。
- `WRITE` 写文件后, `extract_facts()` 不会把文件内容识别为事实(除非文件是 `/etc/hostname` 这种已读文件, 但写操作不会触发读取)。
- 因此 WRITE 几乎永远拿不到 `new_fact` 加成, 基础分又低, 会被系统完全忽略。

### 3.2 推荐奖励设计

在 `_compute_reward()` 中增加意图专属分支:

```python
# 写操作专属奖励 (在基础奖励之后, 重复惩罚之前计算)
written_bytes = 0
if intent in ("WRITE", "APPEND", "GENERATE") and success:
    written_bytes = self._get_written_file_size(result, params)
    # 1. 成功创作奖励
    reward += 0.3
    # 2. 内容长度奖励: 每 100 字节 +0.1, 上限 +0.5
    reward += min(0.5, written_bytes / 100.0 * 0.1)
    # 3. 新文件加成: 如果文件之前不存在
    if getattr(result, "was_created", False):
        reward += 0.2
    # 4. 内容质量: 写入了 python/json/脚本等结构化内容额外 +0.1
    if self._is_structured_content(params):
        reward += 0.1
```

同时需要在奖励表中加入基础分:

```python
self.INTENT_REWARD_BASE.update({
    "WRITE":    0.4,   # 低于 READ/INFO, 但为正
    "APPEND":   0.35,
    "GENERATE": 0.5,   # 脚本生成价值更高
})
```

### 3.3 防止奖励失衡的控制阀

| 控制手段 | 说明 |
|----------|------|
| 字节奖励上限 | 固定上限 0.5, 避免系统写大文件刷分 |
| 重复覆盖惩罚 | 同一文件 30 步内重复写, 奖励 ×0.3 |
| 路径白名单 | 只允许 `/tmp/`, `/workspace/`, `/persistent/scripts/` |
| 创作冷却 | 每步最多一次写操作, 防止全写循环 |
| 结构化加成 | 只有写 JSON/脚本/报告才给额外分, 避免无意义填充 |

### 3.4 风险

- **奖励过高** → 系统放弃读操作, 专写垃圾文件。
- **奖励过低** → WRITE 继续不被选中。
- 建议把 WRITE 成功奖励控制在 **0.4~0.8** 区间, 低于 INFO/SEARCH(1.0~1.2), 但高于 LS_TMP(0.05)。

---

## 4. Q3: content 参数从哪来?

### 4.1 当前模板的问题

`command_registry.json` 中 WRITE 模板:

```json
"WRITE": { "base": ["sh", "-c"], "args": ["printf '%s\\n' {content} > {path}"] }
```

这是**危险的 shell 注入**: 如果 `content` 包含单引号, 会截断命令并执行任意 shell。

### 4.2 推荐方案: Workbench 事实 + Python 安全写入

在 `Workbench` 中新增 `build_write_content()`:

```python
def build_write_content(self, style: str = "report") -> tuple[str, str]:
    """
    从已有事实生成安全的内容字符串
    返回: (content, suggested_filename)
    """
    import json, random
    selected = []
    system_facts = self.get_facts_by_category("system")
    if not system_facts:
        system_facts = list(self.facts.keys())
    n = min(len(system_facts), 2 + random.randint(0, 2))
    selected = random.sample(system_facts, n)

    records = {k: self.facts[k]["value"] for k in selected}

    if style == "json":
        content = json.dumps(records, ensure_ascii=False, indent=2)
        filename = "/tmp/fact_report.json"
    elif style == "summary":
        lines = ["# Folunar System Summary", ""]
        for k, v in records.items():
            lines.append(f"{k}: {v}")
        content = "\n".join(lines)
        filename = "/tmp/summary.txt"
    else:
        # 默认: key=value
        content = "\n".join(f"{k}={v}" for k, v in records.items())
        filename = "/tmp/facts.txt"

    return content, filename
```

执行时改为通过 python3 写文件, 彻底避免 shell 注入:

```python
# TemplateEngine 中的安全写法
import base64
encoded = base64.b64encode(content.encode()).decode()
cmd = f"echo '{encoded}' | base64 -d > {path}"
```

或者更直接:

```python
# 在 TemplateEngine 中, 对 WRITE/APPEND 特殊处理
if intent in ("WRITE", "APPEND"):
    op = ">" if intent == "WRITE" else ">>"
    # 用 base64 传递 content, 再用 python3 解码写入
    encoded = base64.b64encode(content.encode()).decode()
    shell_cmd = (
        f"python3 -c \"import base64; "
        f"data=base64.b64decode('{encoded}'); "
        f"open('{path}', 'wb').write(data)\""
    )
    args = ["sh", "-c", shell_cmd]
```

### 4.3 content 来源优先级

| 优先级 | 来源 | 使用场景 |
|--------|------|----------|
| 1 | Workbench facts JSON | 写 `/tmp/fact_report.json` |
| 2 | 上一步命令输出 | 把 `INFO`/`CUSTOM` 输出重定向 |
| 3 | 固定模板字符串 | 兜底, 如 `# generated by Folunar` |
| 4 | 脚本内容 | `GENERATE` 意图专用 |

---

## 5. Q4: 有没有更简单的方式让系统开始"创造"?

**有, 而且应该先做这个。**

### 5.1 方案一: 调整脚本保存路径(10 分钟)

当前 `agent/online_agent.py::_create_and_run_script()` 把脚本存到 `/persistent/scripts/`。这个路径是跨会话保留的, 但系统可能觉得"写了也没用"。

改为 `/workspace/scripts/`:

```python
# agent/online_agent.py: 在 _create_and_run_script 中
self.sandbox.execute("mkdir -p /workspace/scripts", timeout=5)
script_name = f"discover_{self.step_count}.sh"
# ... 后续路径全部替换 /persistent/scripts/ -> /workspace/scripts/
```

### 5.2 方案二: CUSTOM 命令池加入写文件命令(20 分钟)

在 `config/command_registry.json` 的 `custom_commands` 里增加:

```json
"write_tmp_summary": {
  "args": ["python3", "-c", "import os, json; data={k:v for k,v in os.environ.items()}; print(json.dumps(data, indent=2))"],
  "desc": "环境变量摘要"
},
"write_hello": {
  "args": ["sh", "-c", "echo 'hello from Folunar' > /tmp/hello.txt && cat /tmp/hello.txt"],
  "desc": "写测试文件"
},
"write_timestamp": {
  "args": ["sh", "-c", "date > /tmp/last_run.txt && cat /tmp/last_run.txt"],
  "desc": "写时间戳文件"
}
```

### 5.3 方案三: 工作栏自生成目标(30 分钟)

在 `Workbench.generate_self_goal()` 中增加写摘要目标:

```python
# 在 generate_self_goal 的末尾增加
if len(self.facts) >= 4:
    return ("CUSTOM", {
        "custom_args": ["python3", "-c",
            "import json, os; "
            "facts={\"host\":\"$(cat /etc/hostname)\",\"time\":str(os.times())}; "
            "open('/tmp/auto_summary.json','w').write(json.dumps(facts, indent=2)); "
            "print('written')"],
        "cluster": "CREATIVE"
    })
```

更优雅的做法是新增一个 `_build_summary_goal()` 方法, 让工作栏在事实足够多时主动提出写摘要。

### 5.4 这三件事为什么安全

- 不涉及分类器维度变化, 不需要重训。
- 不修改奖励函数核心逻辑。
- 路径限定在 `/tmp`, 不会破坏系统文件。
- 可以让系统先产生"写文件成功并获得奖励"的真实轨迹, 为后续 WRITE 意图扩展积累数据。

---

## 6. 风险评估

| 风险类型 | 等级 | 说明 |
|----------|------|------|
| 破坏性 | 中 | Docker 沙箱内写入 `/tmp`/`/workspace` 安全, 但当前模板未限制路径, 可能覆盖 `/tmp/discoveries.md` 或 `/etc/hostname`。必须加白名单。 |
| 训练稳定性 | 高 | 14→17 维扩展后, 新维度样本为 0, 如果立即全量重训, 分类器可能坍塌。建议零权重初始化新维度 + 渐进式训练。 |
| 奖励平衡 | 高 | WRITE 奖励一旦过高, 系统会放弃读操作。建议从低奖励开始, 根据成功率动态调整。 |
| Shell 注入 | 高 | 当前 `{content}` 直接拼入 shell 命令, 必须改为 base64/python3 安全写入。 |
| 意图混淆 | 中 | WRITE/APPEND/CAT 语义相近, 分类器可能混淆。建议先合并为 WRITE, 后续再拆分。 |

---

## 7. 推荐实施顺序

### P0: 安全基线(必须先做)

1. 在 `TemplateEngine` 中对 WRITE/APPEND 做 base64/python3 安全写入。
2. 在 `_rescue_params()` 中增加写操作路径白名单, 只允许 `/tmp/`, `/workspace/`, `/persistent/scripts/`。
3. 在 `ExecResult` 中增加 `was_created` 和 `output_size` 字段(或新增 `_get_written_file_size()`)。

### P1: 无意图扩展的"创造"热身

1. 把 `generate_script()` 输出路径改为 `/workspace/scripts/`。
2. 在 `custom_commands` 中加入 `write_hello`, `write_timestamp`, `write_tmp_summary`。
3. 在 `generate_self_goal()` 中加入"写摘要到 /tmp"目标。

### P2: 奖励函数改造

1. 在 `_compute_reward()` 中加入 WRITE/APPEND/GENERATE 专属奖励分支。
2. 更新 `INTENT_REWARD_BASE` 和 `INTENT_REWARD_CREATIVE`。
3. 增加重复写文件惩罚(同一文件 30 步内 ×0.3)。

### P3: 参数规则与内容生成

1. 在 `param_rules.json` 中增加 WRITE/APPEND/GENERATE 规则。
2. 在 `Workbench` 中实现 `build_write_content()`。
3. 在 `TemplateEngine` 中支持 `content` 参数的安全注入。

### P4: 意图维度扩展

1. 修改 3 处 `INTENTS` 列表, 加入 `WRITE`, `APPEND`, `GENERATE`。
2. 修改 `N_INTENTS = 17`(`conductor.py` 和 `online_agent.py`)。
3. 修改 `WorldModel` 默认 `n_intents=17`。
4. 运行 `expand_intents()` 自动扩展旧 checkpoint。
5. 收集 P1~P3 阶段的真实 WRITE 轨迹, 生成 `data/intent_train_v3.jsonl` 增量样本。
6. 增量训练分类器/Conductor/WorldModel。

---

## 8. 具体代码改动建议

### 8.1 路径白名单(必须)

文件: `agent/online_agent.py`, 在 `_rescue_params()` 末尾增加:

```python
if intent in ("WRITE", "APPEND", "GENERATE"):
    p = fixed.get("path", "")
    allowed_prefixes = ("/tmp/", "/workspace/", "/persistent/scripts/")
    if not any(p.startswith(prefix) for prefix in allowed_prefixes):
        fixed["path"] = "/tmp/folunar_output.txt"
```

### 8.2 安全写入(必须)

文件: `benchmark/template_engine.py`, 修改 `build_args()`:

```python
if intent in ("WRITE", "APPEND"):
    import base64
    content = params.get("content", "")
    path = params.get("path", "/tmp/output.txt")
    op = ">" if intent == "WRITE" else ">>"
    encoded = base64.b64encode(content.encode()).decode()
    shell_cmd = (
        f"python3 -c \"import base64; "
        f"data=base64.b64decode('{encoded}'); "
        f"f=open('{path}','wb'); f.write(data); f.close(); "
        f"print(len(data))\""
    )
    return ["sh", "-c", shell_cmd]
```

### 8.3 写文件大小反馈

文件: `agent/online_agent.py`, 新增方法:

```python
def _get_written_file_size(self, result: ExecResult, params: dict) -> int:
    """从结果输出或沙箱 stat 获取写入字节数"""
    try:
        if result.stdout and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    except Exception:
        pass
    if self.sandbox and params.get("path"):
        try:
            r = self.sandbox.execute(f"stat -c %s {params['path']}", timeout=5)
            if r.exit_code == 0 and r.stdout.strip().isdigit():
                return int(r.stdout.strip())
        except Exception:
            pass
    return 0
```

### 8.4 奖励函数分支

文件: `agent/online_agent.py`, 在 `_compute_reward()` 中重复惩罚之前插入:

```python
# 创作奖励
if intent in ("WRITE", "APPEND", "GENERATE") and success:
    written_bytes = self._get_written_file_size(result, params)
    reward += 0.3  # 成功创作基础奖励
    reward += min(0.5, written_bytes / 100.0 * 0.1)  # 字节奖励上限 0.5
    if getattr(result, "was_created", False):
        reward += 0.2
    # 重复写文件惩罚
    write_key = f"write:{params.get('path','')}"
    if getattr(self, "_recent_writes", {}).get(write_key, 0) > self.step_count - 30:
        reward *= 0.3
    self._recent_writes[write_key] = self.step_count
```

### 8.5 内容生成器

文件: `agent/workbench.py`, 新增:

```python
def build_write_content(self, style: str = "report") -> tuple[str, str]:
    import json, random
    keys = self.get_facts_by_category("system") or list(self.facts.keys())
    if len(keys) < 2:
        content = "# Generated by Folunar\n"
        return content, "/tmp/folunar_note.txt"
    n = min(len(keys), 2 + random.randint(0, 2))
    selected = random.sample(keys, n)
    records = {k: self.facts[k]["value"] for k in selected}
    if style == "json":
        content = json.dumps(records, ensure_ascii=False, indent=2)
        return content, "/tmp/fact_report.json"
    lines = ["# Folunar System Summary", ""] + [f"{k}: {v}" for k, v in records.items()]
    return "\n".join(lines), "/tmp/summary.txt"
```

### 8.6 意图维度扩展(最后做)

文件: `agent/nanny.py:18`, `agent/conductor.py:16`, `agent/online_agent.py:59`:

```python
INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP",
           "READ_ETC", "USB_DEVICES", "DISK_USAGE", "LS_TMP", "ARCH_INFO", "CUSTOM",
           "WRITE", "APPEND", "GENERATE"]
```

文件: `agent/conductor.py:19`:

```python
N_INTENTS = 17
```

文件: `agent/online_agent.py:60`:

```python
N_INTENTS = 17
```

文件: `agent/world_model.py:34`:

```python
def __init__(self, embed_dim: int = 384, n_intents: int = 17, ...):
```

### 8.7 参数规则

文件: `config/param_rules.json`, 在 `intents` 对象末尾增加:

```json
"WRITE": {
  "description": "写入文件",
  "params": ["path", "content"],
  "rules": [
    {
      "param": "path",
      "priority": 10,
      "patterns": ["(/(?:tmp|workspace|persistent/scripts)/[\\w./-]+)"],
      "note": "只允许写到安全目录"
    },
    {
      "param": "path",
      "priority": 1,
      "default": "/tmp/folunar_output.txt",
      "note": "默认安全路径"
    }
  ],
  "keyword_map": {}
},
"APPEND": {
  "description": "追加到文件",
  "params": ["path", "content"],
  "rules": [
    {
      "param": "path",
      "priority": 10,
      "patterns": ["(/(?:tmp|workspace|persistent/scripts)/[\\w./-]+)"],
      "note": "只允许写到安全目录"
    },
    {
      "param": "path",
      "priority": 1,
      "default": "/tmp/folunar_output.txt",
      "note": "默认安全路径"
    }
  ],
  "keyword_map": {}
},
"GENERATE": {
  "description": "生成脚本/报告",
  "params": ["path", "content"],
  "rules": [
    {
      "param": "path",
      "priority": 1,
      "default": "/workspace/scripts/generated.sh",
      "note": "默认生成脚本路径"
    }
  ],
  "keyword_map": {}
}
```

---

## 9. 训练与验证建议

1. **不要冷启动**: 先跑 P1~P3 至少 500 步, 收集 50+ 条 WRITE 轨迹, 再扩展意图维度。
2. **新维度零初始化**: 扩展分类器/Conductor/WorldModel 时, 新维度权重用 `std=0.01` 零附近初始化(当前代码已经这样做)。
3. **冻结旧层前 N 轮**: 增量训练时, 前 5 个 epoch 冻结旧意图输出权重, 只训练新维度。
4. **监控指标**:
   - WRITE 被选中频率: 目标 5~10%, 超过 30% 说明奖励过高。
   - 平均文件大小: 目标 50~500 字节, 过大说明在刷字节奖励。
   - 成功率: 目标 >80%, 过低检查路径/content 生成。

---

## 10. 最终建议

- **立即做**: P0 安全写入 + P1 三件小事(改脚本路径、加 CUSTOM 写命令、加自生成摘要目标)。
- **下周做**: P2 奖励函数改造 + P3 参数规则与内容生成。
- **有数据后再做**: P4 意图维度扩展到 17。

这样可以用最小改动让系统先" accidentally "学会创造, 再把这些行为提升为一级意图, 避免架构大改后的训练崩溃。
