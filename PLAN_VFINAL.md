# Folunar_ 项目 — 总体规划书 vFinal

> 整合 ChatGPT · Claude · Gemini · Kimi 各两轮评审
> 写作日期: 2026-06-18

---

## 一、项目背景与终极目标

从零构建一个在 Docker 沙箱中自主探索、持续学习、且每次行为都不同的 AI agent。

**哲学假设**: 「思考 = 概率 + 最优选择」。人类认知与机器预测本质相同，区别仅在于规模和时间尺度。

**物理约束**:
- CPU only (Intel Core Ultra 9 185H, 16C/22T, 105°C 热限)
- 30GB RAM, ~24GB 可用
- Docker 可用
- 单人业余时间

**三年路线**: Qwen/DeepSeek SFT → GRU 7.6M → Mamba 7.3M~19.7M → MTP 18.8M → SlotMind 6.9M → ModelCluster 35~46M → **大脑+手脚分离 (当前)**

---

## 二、当前架构设计（v2，基于四大模型评审修正）

```
┌────────────────────────────────────────────────────────────┐
│         环境 (Docker 沙箱)                                  │
│         │ 执行命令        ↑ 观察输出                         │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
┌─────────▼────────────────┼──────────────────────────────────┐
│   大脑 (意图分类器)        │                                 │
│                          │                                  │
│  backbone: Sentence-     │                                  │
│   Transformer (22M 冻结)  │                                  │
│  分类头: Linear(384→10)  │                                  │
│  (~4K 可训参数)           │                                  │
│                          │                                  │
│  输入: 结构化状态 →       │                                  │
│        文本序列化 → MiniLM │                                  │
│                          │                                  │
│  输出: 10 类意图分布       │                                  │
└──────────────────┬───────┘                                  │
                   │ 意图 + 上下文                              │
┌──────────────────▼──────────────────────────────────────────┐
│   手脚 (Qwen 2.5 1.5B, 参数推理器)                          │
│                                                             │
│  输入: "意图=SEARCH, 上下文=最近在 /etc 下"                  │
│  输出: {"pattern": "root", "path": "/etc/passwd"}           │
│  约束: JSON schema 校验 + 参数验证层                        │
│  回退: 参数无效 → 不执行 → 返回错误信号                      │
└──────────────────┬──────────────────────────────────────────┘
                   │ 参数 JSON
┌──────────────────▼──────────────────────────────────────────┐
│   确定性模板引擎 (0 幻觉, 0 延迟)                            │
│                                                             │
│  SEARCH {pattern, path} → ["grep", "'root'", "/etc/passwd"] │
│  安全: subprocess.run(args=[...]), 无 shell=True            │
│  白名单: /proc/, /etc/, /tmp/, /usr/ 等                     │
└─────────────────────────────────────────────────────────────┘

好奇心系统 (独立, 并行):
  RND target: 固定随机网络, 编码状态
  RND predictor: 可训练小 MLP, 预测 target 输出
  好奇心 = ||predictor(状态') - target(状态')||^2
  解决 Noisy TV: date/ps/uptime 不改变状态 → 不触发好奇
```

---

## 三、全部评审摘要

### 第一轮 (对 v1 计划书的原始批评)

| 模型 | 最尖锐的批评 | 最有价值的建议 |
|------|-------------|---------------|
| **Kimi** | "46M 在 2000 条上从零训 = 参数/数据比 23,000:1，必然过拟合" | 24h MVP: Sentence Transformer + 硬编码模板 |
| **Claude** | "参数从哪里来？计划书最大的空洞" | RND 代替 ICM; TF-IDF baseline 先行; shell 注入警告 |
| **Gemini** | "20 个按钮的遥控器，不是思考" | Ensemble World Models 分离认知/环境不确定性 |
| **ChatGPT** | "Novelty ≠ Utility，新发现不等于能力增长" | AB 测试框架、KPI 体系、安全合规评估 |

### 第二轮 (对 v2 修正版的二次评审)

| 模型 | 最尖锐的批评 | 最有价值的建议 |
|------|-------------|---------------|
| **Kimi** | "状态编码有三个矛盾的定义，可能导致系统完全失败" | 完整的单状态编码设计方案 + 在线经验回放 |
| **Claude** | (未回复) | — |
| **Gemini** | "Qwen 在篡位——真正的思考在 Qwen 里，大脑只是按按钮" | 向量意图库 (连续向量匹配替代离散分类) |
| **ChatGPT** | "RND 误差 ≠ 能力增长，agent 会陷入新颖性陷阱" | Skill Injection Test; Brain Necessity Benchmark |

---

## 四、当前识别的所有问题

### 🔴 级（可能导致系统完全失败）

| # | 问题 | 来自 | 修复方案 |
|---|------|------|---------|
| 1 | 状态编码定义模糊 | Kimi v2 | 统一为结构化元组 → 文本序列化 → MiniLM 嵌入 |
| 2 | 分类器离线训练，无法在线适应 | Kimi v2 | 加经验回放池 + 定期微调 |
| 3 | RND 误差 ≠ 能力增长 (Novelty Trap) | ChatGPT v2 | 加 Utility metric；区分新颖 vs 有用 |
| 4 | 没有写操作 | Gemini v2 | 加 WRITE_FILE / RUN_SCRIPT 意图 |

### 🟡 级（会限制能力但不会让系统崩溃）

| # | 问题 | 来自 | 修复方案 |
|---|------|------|---------|
| 5 | 参数推理没有验证层 | Kimi v2 | Qwen 输出 → JSON schema → 路径/命令存在检查 |
| 6 | PIPE 在扁平分类下无法表达 | Kimi v2 | v2 先移除 PIPE，留到 v3 |
| 7 | 意图边界仍有重叠 | Kimi v2 | 精简到 7 类：READ, LIST, SEARCH, INFO, INSPECT, EXPLORE, HELP |
| 8 | Qwen 推理延迟被低估 | Claude v1 | 0.5-2s/步，不是 200ms；安排异步流水线 |
| 9 | 状态静态化后 RND 永久归零 | Gemini v2 | 状态必须包含输出语义摘要，不只是路径 |

### 🟢 级（改进空间）

| # | 问题 | 来自 | 修复方案 |
|---|------|------|---------|
| 10 | 分类器用硬标签而不是软标签 | ChatGPT v2 | 未来改多标签分类 (READ 0.7 + SEARCH 0.3) |
| 11 | 没有长期目标维护 | ChatGPT v2 | 未来加 PLAN 意图 |
| 12 | 意图不能自扩展 | Gemini v2 | 未来向量意图库 (度量学习 + 检索) |
| 13 | MiniLM 对系统底层字符串可能编码失真 | Gemini v2 | 验证集包含非自然语言样本 |

---

## 五、收到的全部建议汇总

### 架构建议
- ✅ **已采纳**: 46M 从零训 → Sentence Transformer backbone + 4K 头 (Kimi v1)
- ✅ **已采纳**: Qwen 不做 bash 翻译 → 只做参数推理 JSON (Claude v1)
- ✅ **已采纳**: ICM 好奇心 → RND 状态编码预测 (Claude v1, Gemini v1)
- ✅ **已采纳**: 20 类意图 → 10 类 (所有模型)
- ✅ **已采纳**: subprocess.run(args=[...]) 避免 shell 注入 (Claude v1)
- ⏳ **待决策**: 向量意图库代替 Softmax 分类 (Gemini v2)
- ⏳ **待决策**: 加 WRITE_FILE / EXECUTE 意图 (Gemini v2)
- ⏳ **待决策**: 多标签 Soft Intent (ChatGPT v2)

### 实验建议
- ⏳ **待决策**: Skill Injection Test — 人工塞所有技能，测瓶颈是技能还是规划 (ChatGPT v2)
- ⏳ **待决策**: Brain Necessity Benchmark — L1/L2/L3 任务测 Brain Gain (ChatGPT v2)
- ⏳ **待决策**: 先跑 TF-IDF baseline 再上 4K 头 (Claude v1)

### 指标建议
- ⏳ **待考虑**: New Capability Discovery Rate (ChatGPT v2)
- ⏳ **待考虑**: 探索阶段 60% 成功率 / 生产阶段 90% 成功率 (ChatGPT v2)
- ⏳ **待考虑**: 用 Multi-head RND 代替多个独立 predictor (Gemini v2)

### 安全建议
- ✅ **已采纳**: Docker 只读 root、无网络、限制系统调用 (Claude v1, ChatGPT v1)
- ⏳ **待实现**: Qwen 参数 JSON schema + 验证层 (Kimi v2)
- ⏳ **待实现**: 命令白名单二次校验 (Claude v1)

---

## 六、统一的状态编码设计（根据 Kimi v2 建议）

```
状态 = {
    "cwd": "/home/user",                           # 当前目录
    "visited_dirs": {"/", "/etc", "/proc"},         # 已访问目录
    "known_files": {"/etc/passwd", "/proc/cpuinfo"},# 已知文件
    "recent_steps": [                               # 最近 5 步历史
        ("READ", "/etc/passwd"),
        ("SEARCH", "root"),
    ],
    "last_summary": "文件30行, 包含 'root'",        # 上步输出语义摘要（确定性规则生成，不是 LLM）
}
```

序列化为文本（喂给 MiniLM）:
```
"当前目录: /home/user. 已访问: /, /etc, /home, /proc. 
 已知文件: /etc/passwd, /proc/cpuinfo. 
 历史: READ /etc/passwd → SEARCH root. 
 上步摘要: 文件30行,包含'root'."
```

这个文本 → MiniLM → 384 维嵌入 → 同时用于意图分类和 RND 输入。

---

## 七、MVP 实验计划

### 第一步：Brain Necessity Benchmark (48h)

验证核心假设：「大脑+手脚」是否优于「纯手脚」。

```
实验组: 大脑(RND 分类器) + 手脚(Qwen + 模板引擎)
对照组: 手脚(Qwen + 模板引擎) + 随机意图选择

任务:
  L1: "读取 /etc/hostname"
  L2: "找出 /etc/passwd 中包含 root 的行并计数"
  L3: "找出 CPU 型号 → 检查内存大小 → 写一个状态报告"

指标:
  Brain Gain = (实验组成功率 - 对照组成功率) / 对照组成功率

通过条件: L1/Brain Gain < 5%（大脑对简单任务无意义, 符合预期）
          L2/Brain Gain > 10%（大脑开始产生价值）
          L3/Brain Gain > 30%（大脑在长链决策中价值显著）
```

### 第二步：Skill Injection Test (延着上一步)

```
给 agent 人工注入所有 10 个意图的全部参数能力
测成功率提升 → 如果提升小 = 瓶颈是规划(大脑)
             如果提升大 = 瓶颈是技能(需要加意图)
```

### 第三步：闭环系统搭建

```
如果 Brain Necessity Benchmark 证明大脑有价值:

  Phase 1 - 状态编码 (1天)
    实现上述统一状态编码
    实现输出摘要器（规则驱动）

  Phase 2 - 意图分类 (1天)
    Sentence Transformer + Linear(384→10)
    2000 条数据训 10 epoch
    验证准确率 > 80%

  Phase 3 - 手脚集成 (1天)
    Ollama + Qwen 2.5 1.5B
    参数 JSON schema + 验证层
    确定性模板引擎 + 白名单

  Phase 4 - RND 好奇心 (1天)
    Multi-head RND predictor
    状态编码转移预测
    在线更新 predictor

  Phase 5 - 闭环 (1天)
    Docker 沙箱中运行
    持续 N 步自主探索
    记录指标
```

---

## 八、成功标准与失败条件

### 如果 3 个月内:

| 指标 | 目标 | 失败条件 |
|------|------|---------|
| 意图分类准确率 | >80% | <60% |
| 参数推理成功率 | >90% | <70% |
| 有效探索步数/总步数 | >50% | <30% |
| L3 Brain Gain | >30% | <10% |
| 连续 100 步不重复 | 达成 | 无法达成 |

**失败 → 转向路线**：
- 如果大脑没有 Brain Gain → 说明瓶颈不在决策，放弃多模型架构，改用纯 Qwen ReAct Agent
- 如果 Qwen 参数推理不可控 → 把参数也用模板写死，放弃 LLM，退化为规则系统
- 如果 RND 好奇心不工作 → 改用 ε-greedy 随机探索，放弃好奇心模块

---

## 九、参与评审的模型

| 模型 | v1 轮 | v2 轮 |
|------|-------|-------|
| ChatGPT | ✅ 完整评审 | ✅ 二次评审 |
| Claude  | ✅ 完整评审 | ❌ 未回复 |
| Gemini  | ✅ 完整评审 | ✅ 二次评审 |
| Kimi    | ✅ 完整评审 | ✅ 二次评审 |

所有原始评审和回复文件位于 `Response/` 目录。
