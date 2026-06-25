# Folunar_ P11-P15: 从稳定到自由

> 最终更新: 2026-06-25
> 上一阶段: P10 层级架构完成 — 300步/100%/16意图
> 下一阶段: P11 持久化 → P12 知识拓展 → P13 自造工具 → P14 LLM 双模型 → P15 脑启发动态层

---

## 一、现状: P10 已完成的

| 模块 | 做什么 | 状态 |
|------|--------|------|
| **FactGraph** (`agent/fact_graph.py`) | 图结构知识库: 节点=事实, 边=关系, schema 缺口检测 | 生产 |
| **MetaCognitiveSelector** (`agent/meta_selector.py`) | 3 MODE 选择: EXPLORE/CREATE/LEARN, 规则 + MLP 偏置 | 生产 |
| **GoalGenerator** (`agent/goal_generator.py`) | 模式驱动目标生成: 缺口/好奇心/创作; utility gate; CUSTOM 过滤 | 生产 |
| **GrowingWorldModel V4** (`agent/world_model_v4.py`) | 核+叶架构: 共享隐层 + 逐意图预测头; 自动扩展 | 生产 |
| **EpisodicMemory** (`agent/episodic_memory.py`) | JSONL 环形缓冲: 余弦相似度 + 时间衰减召回; 惊喜计算 | 生产 |
| **CreativeWriter P0-P2** (`agent/creative_writer.py`) | Ollama LLM 插件: 异步生成 + 自适应频率; gemma4:e4b | 异步 |
| **IntentClassifier** (140KB) | 66K 参数意图分类器 | 生产 |
| **Conductor + Nanny** (140KB) | 67K 参数 thought 向量 + 翻译 | 生产 |
| **WorldModel V3** (0.5M) | 全局预测器 | 生产 |
| **RND** (0.3M) | 好奇心模块 | 生产 |
| **ErrorRecovery** | 错误恢复 + 回退 | 生产 |

### 当前指标 (300步)

| 指标 | 值 |
|------|-----|
| 成功率 | 100% (300/300) |
| 意图覆盖 | 16/16 种 (各 ~6%) |
| 总奖励 | 512 |
| FactGraph 节点 | 86 |
| Schema 覆盖 | 44% |
| 沙箱内文件 | 80 个, ~192KB |
| 创造力 | 系统画像/discovery/验证脚本/实验脚本 |
| LLM 异步 | 成功1次, gemma4:e4b 9GB 太慢 |

---

## 二、发现的 8 个根本局限

| # | 局限 | 影响 | 优先级 |
|---|------|------|--------|
| 1 | **无持久化** — docker rm -f 后一切归零; checkpoint 散落 | 知识无法积累, 每次从零开始 | P0 |
| 2 | **知识面太窄** — 只读 /proc/cpuinfo, /etc/os-release, free -h | 不理解系统全貌, 事实 86 节点后停滞 | P0 |
| 3 | **无自造工具** — 只能写 shell 模板, 不会生成 Python 工具并复用 | 创造力被格式束缚, 无法自举 | P1 |
| 4 | **LLM 太慢** — gemma4:e4b 9GB CPU 推理 60s | 深度内容生成不稳定 | P1 |
| 5 | **无跨 session 学习** — checkpoint 存但不统一恢复 | 每次训练从头开始 | P0 |
| 6 | **模板束缚** — 内容格式固定 (profile/discovery/check) | 表达不自由, 无新颖格式 | P1 |
| 7 | **无解释能力** — 报告事实但不推理 "为什么" | 没有真正的理解 | P2 |
| 8 | **无网络** — --network none 隔离 | 无法获取外部知识 | 底层约束 |

---

## 三、目标架构: 三层 + 两图 + 一记忆 (P15 最终形态)

```
┌─────────────────────────────────────────────────────────────┐
│  顶层: 元认知层 (MetaCognitive) — 前额叶                    │
│  MetaCognitiveSelector: 3 MODE + 置信度门控 + 自我反思      │
│  输入: FactGraph 覆盖度 / RND / WM loss / 历史 / 置信度     │
│  输出: MODE + intent_bias + 自适应学习率                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  中层: 目标层 (GoalGenerator) — 海马 / 计划                 │
│  输入: MODE + FactGraph.gaps() + RND + WM 分歧 + ToolReg    │
│  输出: Goal 队列 (fact_fill / curiosity / create / verify)  │
│  新: TOOL 目标类型 — "发现缺失能力 -> 自造工具 -> 使用"     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  底层: 动作层 (Action) — 运动皮层                           │
│  ActionExecutor + KnowledgeMapper                           │
│  输入: Goal + state_text                                    │
│  输出: 命令执行 + 结果回写 FactGraph + 新工具注册           │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Dynamic      │    │ Growing      │    │ Episodic     │
  │ FactGraph    │    │ World        │    │ Memory       │
  │              │    │ Model V5     │    │              │
  │ 节点=事实     │◄──►│ core+叶      │◄──►│ 惊喜事件     │
  │ 边=关系       │    │ +图注意力    │    │ 相似召回     │
  │ schema自动扩展│    │ +自我反思    │    │ 长期存储     │
  │ 持久SQLite    │    │ 持久checkpoint│   │ 持久SQLite   │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌──────────────────┐
                    │  持久化层         │
                    │  PersistentStore  │
                    │  SQLite + 文件挂载 │
                    │  version控制      │
                    │  docker volume    │
                    └──────────────────┘
```

---

## 四、P11: 统一持久化 (第一步, 收益最大)

### 做什么
Agent 跑的所有状态 — FactGraph / WM V4 / Experience / MetaSelector / EpisodicMemory / ToolRegistry — 统一存到 `data/persistent/`, 通过 Docker volume 挂载, 即使是 `docker rm -f` 也不丢失。

### 文件清单

| 文件 | 做什么 |
|------|--------|
| `agent/persistent_store.py` | 统一 save/load 入口, SQLite + PyTorch checkpoint |
| `data/persistent/` | Docker volume 挂载路径 |

### 接口设计

```python
class PersistentStore:
    def save_factgraph(self, graph)         → data/persistent/fact_graph.json
    def load_factgraph(self)                → FactGraph | None
    
    def save_world_model_v4(self, wm_model) → data/persistent/models/wm_v4.pt
    def load_world_model_v4(self, n_intents)→ GrowingWorldModelV4 | None
    
    def save_classifier(self, classifier)   → data/persistent/models/classifier.pt
    def load_classifier(self)               → IntentClassifier | None
    
    def save_conductor(self, conductor)     → data/persistent/models/conductor.pt
    def load_conductor(self)                → ConductorHead | None
    
    def save_experience(self, buffer)       → data/persistent/experience.jsonl
    def load_experience(self)               → ExperienceBuffer | None
    
    def save_workbench(self, wb, graph)     → data/persistent/workbench.json (含FactGraph)
    def load_workbench(self)                → WorkbenchState | None
    
    def save_episodic_memory(self, mem)     → data/persistent/episodic_memory.jsonl
    def load_episodic_memory(self)          → EpisodicMemory | None
    
    def save_tool_registry(self, tools)     → data/persistent/tools/registry.json
    def load_tool_registry(self)            → dict | None
    
    def save_meta_selector(self, meta)      → data/persistent/meta_selector.json
    def load_meta_selector(self)            → MetaCognitiveState | None
    
    def version(self) -> int                → 当前 schema 版本, 用于迁移
    def migrate(self, old_v, new_v)         → 版本迁移
```

### 实现要点
- SQLite 存结构数据 (事实图、工具注册、元选择器状态、统计)
- PyTorch 模型用 `torch.save/load` 到 `data/persistent/models/`
- 每个 checkpoint 带 `version` 字段, load 时校验
- 版本不一致 → 软重置 (保留旧数据 + 新 schema)
- Docker 启动挂载: `-v $(pwd)/data/persistent:/app/data/persistent`

### 成功标准
```
1. 跑 100 步 → docker rm -f → 跑 100 步
2. 第二次运行: FactGraph 节点数 >= 第一次的一半
3. WM V4 恢复后继续训练 (loss 不跳变)
4. Experience buffer 从 100 继续增长, 不清零
```

---

## 五、P12: 知识拓展 (系统探索)

### 做什么
在 `--network none` 的限制下, 系统性地探索沙箱内部所有可到达的信息源: `/proc/*`, `/sys/*`, `/usr/bin/*`, `dpkg`, `apt`, `man`, 以及其他可发现的工具。

### 新增模块

| 文件 | 做什么 |
|------|--------|
| `agent/knowledge_mapper.py` | 双广度优先探索引擎 |
| `config/discovery_phases.json` | 探索阶段配置 |

### 探索阶段

```
Phase A (静态清单)
  ├─ ls /usr/bin /bin /sbin → 所有可用命令
  ├─ dpkg -l → 所有已安装包
  ├─ ls /proc /sys /etc /dev → 文件系统全貌
  ├─ ls /sys/class/* → 设备类别
  └─ ls /proc/net /proc/sys → 网络/内核参数

Phase B (命令自描述)
  ├─ cmd --help (对每个新命令)
  ├─ man cmd (关键命令)
  └─ whatis cmd (快速分类)

Phase C (子系统 BFS)
  ├─ cat /proc/cpuinfo /proc/meminfo /proc/uptime
  ├─ cat /proc/net/dev /proc/net/tcp /proc/net/route
  ├─ cat /proc/1/cgroup → 容器检测
  ├─ cat /sys/class/net/*/address → MAC
  └─ find /sys/devices → 硬件拓扑

Phase D (只读执行探测)
  ├─ uname -a; lscpu; lshw (如果有)
  ├─ ip addr; ip route; ss -tuln
  ├─ df -h; mount; lsblk
  ├─ ps aux; env
  └─ timedatectl; locale

Phase E (能力推断)
  ├─ python3 --version; which python3
  ├─ gcc --version; which gcc
  ├─ curl --version → 但 network none
  └─ 推断: can_run_python / can_compile / has_network
```

### 安全约束
- 只读白名单: 禁止 `rm/dd/mkfs/iptables/passwd/reboot/shutdown`
- 每命令 timeout 2s, 输出截断 4KB
- --network none 天然防外连
- Phase A-B 在 50 步内完成, C-E 逐步分散

### 自动 Schema 扩展
发现大量新事实 → 自动推断新类别:
- `sys_class_*` 节点 → 创建 `hardware` schema
- `proc_net_*` 节点 → 创建 `network` schema
- `dpkg_*` 节点 → 创建 `package` schema
- `cmd_*` 节点 → 创建 `command` schema

### 成功标准
```
1. FactGraph 节点 > 200 (86→200+)
2. Schema 覆盖 > 60%
3. 能回答 "这台机有什么 python 包?" "有什么网络设备?"
4. 自动推断出至少 3 个新 schema 类别
```

---

## 六、P13: 自造工具 (工具工厂)

### 做什么
Agent 能自己生成 Python/shell 工具 → 保存 → 自动注册 → 在后续步骤中使用 → 记录效果 → 持续优化。

### 新增模块

| 文件 | 做什么 |
|------|--------|
| `agent/tool_factory.py` | 工具生成引擎 (规则+LLM) |
| `agent/tool_registry.py` | 工具发现+注册+统计 |
| `data/persistent/tools/` | 工具存储目录 |

### 工具生命周期

```
发现需求 (CREATE mode / GoalGenerator 检测缺口)
    │
    ▼
ToolFactory.generate(需求描述, 工具类型)
    │  ├─ Python 采集工具
    │  ├─ Shell 探索脚本
    │  └─ 复合分析工具
    │
    ▼
保存到 data/persistent/tools/tool_<name>.py
    │
    ▼
ToolRegistry.autodiscover() → 扫描 data/persistent/tools/
    │
    ▼
注册到 TOOL 意图 (新增 intent #17: TOOL)
    │
    ▼
Agent 选择 TOOL 意图 + tool_name 参数
    │
    ▼
执行 → 结果回写 FactGraph + 记录使用次数
    │
    ▼
低效工具 → 废弃标记; 高效工具 → 优先使用
```

### 工具类型

| 类型 | 生成方式 | 示例 |
|------|---------|------|
| `data_gather` | 规则模板 | `list_installed_packages.py`, `scan_network_state.py` |
| `analysis` | 规则+LLM | `analyze_log.py`, `correlate_facts.py` |
| `creative` | LLM | `write_poem.py`, `generate_story.py` |
| `utility` | 规则 | `backup_facts.py`, `validate_data.py` |

### 工具接口契约

```python
# 每个工具有统一接口
def run(env: dict) -> dict:
    """
    env: {"workbench": ..., "state_text": ..., "tools_dir": ...}
    returns: {"success": bool, "data": dict, "summary": str}
    """
    pass
```

### 成功标准
```
1. 100 步内生成至少 3 个工具
2. 至少 1 个工具被 TOOL 意图使用
3. 工具生成的 FactGraph 节点被其他意图利用
4. 高效工具获得更高优先级
```

---

## 七、P14: LLM 双模型路由

### 做什么
qwen3.5:0.8b (988MB) 做快思考 — 每步路由、意图粗分、目标草拟; gemma4:e4b (9GB) 做慢思考 — 深度报告、代码生成、复杂分析。

### 修改

| 文件 | 改动 |
|------|------|
| `agent/creative_writer.py` | 双模型路由, qwen 做 gate |
| 新增 config | 路由策略配置 |

### 路由逻辑

```
generate_content() 被调用
    │
    ├─ 如果 task == report/code/analysis → 走 gemma (异步, 不限时)
    ├─ 如果 task == quick/summary/fallback → 走 qwen (同步, 5s)
    └─ 如果 task 未知 → qwen 先判断, 决定是否走 gemma
```

### 双模型分工

| 特征 | qwen3.5:0.8b | gemma4:e4b |
|------|-------------|-------------|
| 大小 | 988MB | ~9GB |
| 推理速度 | ~2s | ~10-60s |
| 角色 | 每步路由 + 快思 | 深度生成 + 创作 |
| 调用频率 | 每步 (可能) | 每 40-80 步 |
| 输出格式 | 简单文本, 命令选择 | 长报告, 代码, 分析 |
| 超时策略 | 5s 超时 → fallback | 不限时 + 异步 |

### 成功标准
```
1. LLM 生成成功率 > 10% (从 ~0.3%)
2. qwen 路由准: 90% 正确判断需要/不需要 gemma
3. gemma 深度报告 300 步内至少生成 1 份
```

---

## 八、P15: 脑启发动态层

### 做什么
WM V4 → V5: 图注意力、递归自我反思、元认知门控。让模型根据置信度动态决定哪些"认知层"参与决策。

### 新增/修改

| 文件 | 改动 |
|------|------|
| `agent/world_model_v5.py` | 核+叶+图注意力+自我反思 |
| `agent/meta_selector.py` | 置信度门控 + 动态层激活 |
| `agent/online_agent.py` | 元认知反馈循环 |

### 三个升级

**A. 图注意力 (Graph Attention)**
WM forward 前, 根据当前意图从 FactGraph 选择 top-k 最相关节点, 聚合到状态表示:

```python
# 当前: state_emb = encoder(state_text)
# V5: state_emb = encoder(state_text) + attention(fact_graph, query=intent_emb)
```

**B. 递归自我反思 (Self-Reflection)**
每步执行后, WM 预测 vs 实际结果 → `model_confidence`:

```python
# 持续高误差 → 强制切 LEARN mode
if avg_prediction_error > threshold:
    meta.force_mode("LEARN")
    # WM 增加训练步数/学习率
```

**C. 元认知门控 (Mode Gate)**
根据置信度 + 事实覆盖度, 动态选择参与层:

```
高置信度 + 事实足      → CREATE (自由创作)
低置信度 + 新刺激       → EXPLORE (探索新知识)
预测持续错              → LEARN (修正模型)
工具可用                → TOOL layer 优先
```

### 成功标准
```
1. WM V5 预测误差 < V4 同条件下的 80%
2. 自我反思正确触发 MODE 切换 (100% 召回的必要模式切换)
3. 图注意力使 FactGraph 利用率提高 2x
4. 动态层不增加每步推理时间 > 20%
```

---

## 九、完整实施路线图

```
P11: 统一持久化 (NOW)
  ├── agent/persistent_store.py        ← 新文件
  ├── 修改 online_agent.py             ← save/load 入口
  └── READ: Docker volume 挂载逻辑

P12: 知识拓展 (NEXT)
  ├── agent/knowledge_mapper.py        ← 新文件
  ├── config/discovery_phases.json     ← 新文件
  ├── 修改 online_agent.py             ← 集成 KnowledgeMapper
  ├── 修改 workbench.py                ← 自动 schema 扩展
  └── 修改 fact_graph.py               ← schema 动态推断

P13: 自造工具 (AFTER)
  ├── agent/tool_factory.py            ← 新文件
  ├── agent/tool_registry.py           ← 新文件
  ├── 修改 goal_generator.py           ← 工具需求检测
  ├── 修改 online_agent.py             ← TOOL 意图集成
  └── 修改 GoalGenerator               ← TOOL 目标生成

P14: LLM 双模型 (NEXT-NEXT)
  ├── 修改 agent/creative_writer.py    ← 双模型路由
  ├── 测试: qwen 路由准确率
  └── 调优: gemma 异步 + thermal gate

P15: 脑启发动态层 (FINAL)
  ├── agent/world_model_v5.py          ← 新文件
  ├── 修改 agent/meta_selector.py      ← 置信度门控
  └── 修改 agent/online_agent.py       ← 元认知循环
```

### 每步验证

```
P11: 跑100步 → docker rm -f → 跑100步 → 检查状态恢复
P12: 跑200步 → FactGraph > 200节点 → Schema > 60%
P13: 跑200步 → 工具≥5个 → TOOL意图使用≥3次
P14: 跑200步 → LLM成功≥3次 → qwen路由准确≥80%
P15: 跑200步 → WM误差下降 → MODE切换智能
```

---

## 十、风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 持久化状态不一致 (version mismatch) | 中 | 高 | version 字段 + auto-migrate + 软重置 |
| KnowledgeMapper 触发危险命令 | 低 | 高 | 只读白名单 + timeout + 沙箱隔离 |
| 自造工具生成恶意代码 | 低 | 中 | 沙箱隔离 + 静态 import 检查 |
| LLM 幻觉污染 FactGraph | 中 | 中 | LLM 输出只进 opinion 节点, 不覆盖 command 事实 |
| 同时改多处导致 bug 难定位 | 中 | 高 | 每 Phase A/B 测试, 保留 step_legacy() |
| WM V4→V5 破坏训练 | 中 | 高 | 新 checkpoint 不覆盖旧; 可回退 |
| --network none 知识上限 | 高 | 中 | 接受; 沙箱内知识已足够 P11-P15 验证 |
| doken rm -f 前未保存 | 中 | 高 | 每 100 步自动保存; exit 时自动保存 |

---

## 十一、与已有模块的关系

| 已有模块 | P11 影响 | P12 影响 | P13 影响 | P14 影响 | P15 影响 |
|----------|---------|---------|---------|---------|---------|
| FactGraph | 持久化到 SQLite | schema 自动扩展 | 工具产出入图 | LLM 产出入 opinion 节点 | 图注意力 |
| WM V4 | 持久化 checkpoint | — | — | — | → V5 |
| MetaSelector | 持久化历史 | — | — | — | 置信度门控 |
| GoalGenerator | — | 新知识缺口 | TOOL 目标 | — | — |
| CreativeWriter | — | — | 工具写作 | 双模型路由 | — |
| EpisodicMemory | 持久化 | — | — | — | — |
| Classifier/Conductor/RND | 持久化 | — | — | — | — |
| ErrorRecovery | — | 新命令失败恢复 | 工具执行错误 | — | — |
| Nanny | — | — | TOOL intent 翻译 | — | — |
| Workbench | 持久化 snapshot | 新提取规则 | 工具结果提取 | — | — |

---

## 十二、不被改变的核心哲学

| 原则 | 保持 |
|------|------|
| **小模型决策, 大模型创作** | 分类器/Conductor/WM 决策; LLM 只碰文本 |
| **--network none 隔离** | 知识只在沙箱内, 外部通过 checkpoint 传递 |
| **CPU 训练** | 所有模型 CPU, 4 线程 |
| **Docker 非特权** | 防副作用 |
| **A/B 自适应** | 保留; P15 置信度门控是 A/B 的增强 |
| **安全写入 base64+python3** | 不变 |
| **Kimi 仅设计(不在闭环)** | 不变 |
