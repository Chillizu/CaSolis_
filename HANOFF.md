# Folunar — Handoff 文档

> 生成于: 2026-06-29
> 项目: 自主进化 Linux Agent, CPU-only 训练, Docker `--network none` 隔离
> 哲学: 小模型(分类器/Conductor)指决策, 大模型(gemma4)做创作但非必须
> 当前阶段: P11-P15 完成 + P16 推理层设计完毕(待实施)
> 最后验证: 9小时马拉松 27,799 步, 92.2% 成功率, 418/418 命令自发现

---

## 一、一句话版

系统是一个在 Docker 沙箱内自主探索的 Linux agent。它通过小模型决策（什么模式、什么目标、什么意图）, 在沙箱里执行 Linux 命令, 从输出中提取事实, 存入图结构知识库(FactGraph), 基于已知事实发现缺口和假设, 再生成新目标。**不依赖 LLM 做决策**——LLM 仅用于报告风格的文本创作, 且太慢(76s/次, CPU)所以频率很低。

核心循环:
```
元认知选择 MODE → 目标生成器提目标 → 分类器/Conductor 选意图
  → 沙箱执行命令 → 提取事实 → 更新知识图
  → 推理/假设 → 新目标 → 继续
```

---

## 二、完整历程

| 阶段 | 做了什么 | 状态 |
|------|---------|------|
| **P0-P9.7** | 核心闭环: 17意图, 100% 稳定, 模板驱动内容创作 | 旧架构, 已归档 |
| **P10** | 层级架构: MODE选择器(3种), GoalGenerator, GrowingWorldModel V4, EpisodicMemory, CreativeWriter, FactGraph | 生产 |
| **P11** | 统一持久化: PersistentStore(SQLite+PyTorch+Docker volume) | 生产 |
| **P12** | 知识拓展: KnowledgeMapper, Phase A-E 探索, 418命令扫描 | 生产 |
| **P13** | 自造工具: ToolFactory(JSON模板), ToolRegistry, 自发现引擎 | 生产 |
| **P14** | LLM管道: qwen3.5+gemma4 双模型(但qwen不靠谱被回退到命令池) | 生产 |
| **P15** | 脑启发动态层: WM V4 图注意力, 自我反思, 置信度门控, 去人为化(自适应奖励/动态StateEncoder/动态GoalGenerator) | 生产 |
| **P16** | **推理层设计完成(待实施)**: BeliefGraph→TransitionMiner→HypothesisEngine→ExperimentPlanner→Verdict | **设计** |

---

## 三、当前架构 (P15 最终, 2600+ 行主循环)

### 核心循环

```
state_text (环境状态)
    │
    ├─ MetaCognitiveSelector → MODE (EXPLORE/CREATE/LEARN)
    ├─ GoalGenerator → Goal 队列 (缺口/好奇心/创作/假设)
    ├─ 想象力 (WorldModel 评分, 80%概率)
    ├─ Conductor (thought向量 → Nanny翻译, A/B自适应)
    ├─ 分类器 (IntentClassifier, 3类: OBSERVE/CREATE/TRY)
    │
    ▼
参数提取 → 模板引擎 → 沙箱命令
    │
    ▼
Docker沙箱 (folunar-sandbox, --network none)
    │
    ▼
ExecResult
    ├─ Workbench.extract_facts (FactGraph 集成)
    ├─ RND.compute_novelty (好奇心)
    ├─ CreativeWriter (异步LLM生成, 每100步)
    ├─ EpisodicMemory (惊喜检测)
    ├─ ExperienceBuffer 存储
    ├─ 在线训练 (分类器+WM+Conductor)
    ├─ 自适应概率触发 (替代所有 step%N)
    └─ InferenceEngine (规则式, 待替换)
```

### INTENTS (3 类, P15 重构后)

| 意图 | 做什么 |
|------|--------|
| **OBSERVE** | READ/LIST/INFO/SEARCH/INSPECT/COUNT — 观察环境 |
| **CREATE** | WRITE/CREATE — 创作内容到沙箱 |
| **TRY** | 执行命令/工具/实验 — 不确定结果的行为 |

### 模块文件一览

| 文件 | 行数 | 功能 |
|------|------|------|
| `agent/online_agent.py` | 2600+ | 主循环: step() + 所有辅助方法 |
| `agent/workbench.py` | ~750 | 事实提取 + FactGraph 集成 + 探针生成 |
| `agent/fact_graph.py` | ~450 | 图结构知识库: 节点/边/schema/缺口/交叉分析 |
| `agent/goal_generator.py` | ~600 | 动态目标生成: 自适应概率 + 实验生成 |
| `agent/meta_selector.py` | ~200 | MODE 选择: 规则+MLP, R8置信度门控 |
| `agent/world_model_v4.py` | ~300 | 增长型WM: 核+叶架构, 图注意力, 自我反思 |
| `agent/conductor.py` | ~250 | thought向量 384→128→64→16, 自动维度适配 |
| `agent/nanny.py` | ~150 | thought→intent 翻译, class_proj投影 |
| `agent/state_encoder.py` | ~130 | 动态状态编码, FactGraph驱动(去人为化) |
| `agent/knowledge_mapper.py` | ~500 | 知识拓展: 418命令发现 + 意图映射 |
| `agent/tool_factory.py` | ~200 | 工具工厂: 从418池生成JSON工具 |
| `agent/tool_registry.py` | ~200 | 工具注册表: 执行 + 统计 |
| `agent/creative_writer.py` | ~350 | LLM创作: 异步gemma4 + 同步回退 |
| `agent/inference_engine.py` | ~180 | 推理引擎: 规则式(待替换为P16) |
| `agent/persistent_store.py` | ~300 | 持久化: SQLite+PyTorch+版本控制 |
| `agent/episodic_memory.py` | ~150 | 情景记忆: 惊喜环形缓冲 |
| `agent/rnd.py` | ~150 | RND好奇心: 新颖度+兴趣重置 |
| `agent/error_recovery.py` | ~200 | 错误恢复: 分类器+回退 |
| `agent/experience.py` | ~150 | 经验回放缓冲区 |
| `agent/detailed_logger.py` | ~100 | JSONL日志 |
| `agent/sandbox_executor.py` | ~100 | Docker沙箱执行 |

### 模型指标

| 模型 | 参数量 | 作用 |
|------|--------|------|
| IntentClassifier | ~66K | 状态文本 → 3类意图 |
| ConductorHead | ~67K | 状态文本 → 16维 thought |
| WorldModelNet V3 | ~0.5M | 全局预测 |
| GrowingWorldModel V4 | 核+16叶 | 逐意图预测 + 自动扩展 |
| RND | ~0.3M | 新颖度检测 |
| MiniLM (冻结) | ~90MB | 状态文本嵌入 |

### 去人为化(已完成的)

| 之前 | 之后 |
|------|------|
| 32个手写奖励值 | 自适应奖励: 从经验中学习 |
| 270行手写StateEncoder | 130行: FactGraph驱动动态选择 |
| GoalGenerator 11个硬编码阈值 | 自适应概率 + 动态评分 |
| 17个硬编码意图 | 3个元意图(OBSERVE/CREATE/TRY) |
| 50+个 step%N 间隔 | 2个保留(持久化安全+LLM轮询) |
| 6个Python工具模板 | 418池JSON工具, 自发现 |
| 14个手写gap→命令映射 | 自动从意图映射表动态查找 |

---

## 四、当前指标 (马拉松 27,799 步)

| 指标 | 值 |
|------|-----|
| 步数 | 27,799 |
| 成功率 | 92.2% |
| 总奖励 | 40,810 |
| 自发现命令 | 418/418 (全部) |
| FactGraph 节点 | ~200 |
| LLM 成功生成 | 多次 (~76s/次) |
| 崩溃/死锁 | 0 |
| 意图覆盖 | 3/3 种 |
| 持久化 | 8个组件, SQLite+Docker volume |

---

## 五、系统能/不能做什么

### 能

| 能力 | 示例 |
|------|------|
| **探索沙箱** | 418个/usr/bin命令全发现, /proc//sys/etc 全扫描 |
| **提取事实** | 任何命令输出 → 图节点 |
| **推理(规则式)** | "no network → isolated", "22 cores → server-class" |
| **创作报告** | 模板报告(94-210B), LLM报告(300-600B) |
| **生成脚本** | 随机组合3-5个418命令成shell脚本 |
| **自造工具** | JSON工具文件(沙箱重启后丢失) |
| **异步LLM** | gemma4后台生成, 不阻塞主循环 |
| **持久化** | SQLite+checkpoint, docker rm -f 不丢 |
| **自适应频率** | 所有操作概率随状态变化, 无硬编码步数 |

### 不能

| 缺失 | 原因 |
|------|------|
| **真正的因果推理** | InferenceEngine 只有4条硬编码规则, 从数据学不到新关系 |
| **假设→实验→验证** | GoalGenerator._self_generate_experiment只是随机挑2个事实+模板查表 |
| **原创代码生成** | LLM代码管道搭好了但gemma4 CPU 76s, 频率极低 |
| **跨session学习** | PersistentStore存了但KnowledgeMapper每轮重扫, 不真正积累理解 |
| **输出格式自由** | 模板固定, 不是涌现 |

---

## 六、P16: 推理层设计 (待实施)

**核心转变**: 从「我写规则告诉系统怎么推理」变成「系统自己从数据中发现因果」。

彻底替换:

| 旧代码 | 问题 | 替换为 |
|--------|------|--------|
| `InferenceEngine` (4条硬编码规则) | `if cpu>8: server-class` | TransitionMiner 从数据学习 |
| `build_cross_analysis()` (手写if/then) | 规则不能自发现 | 因果边自动挖掘 |
| `_self_generate_experiment()` (模板查表) | 随机组合无意义 | HypothesisEngine 按不确定性选 |

5个新模块(全部不依赖LLM):

| 模块 | 做什么 | 之前存在？ |
|------|--------|-----------|
| **BeliefGraph** | FactGraph升级: 边带置信度/证据/支持/反驳 | 从未 |
| **TransitionMiner** | 记录三态因果表 + 离线挖掘 | 从未 |
| **HypothesisEngine** | 生成可验证假设, 按 uncertainty×testability 排序 | 从未 |
| **ExperimentPlanner** | 把假设转为沙箱命令 + WM预测 | 从未 |
| **Verdict & Reflection** | 比较预测vs实际, 更新信念, 反馈WM | 从未 |

实施顺序:

```
R1 (今天): 只记录, 不改行为
  FactGraph 加 confidence/n_evidence
  每步记录 (pre_state, action, post_state)
  
R2 (今天-明天): 离线因果挖掘
  TransitionMiner 算共现/增益/时序
  HypothesisEngine 生成 top-5 假设
  
R3 (后天): 在线实验闭环
  ExperimentPlanner + Verdict
  GoalGenerator 新增 hypothesis_test 类型
  
R4 (后续): 自我改进
  验证结果喂WM → MODE自动切LEARN模式
```

详见 `PLAN_P16_REASONING.md` 和 `REASONING_LAYER_DESIGN.md` (Kimi 设计)。

---

## 七、非重复造轮子证明

| 之前试过的 | 为什么不是同样的东西 |
|-----------|-------------------|
| **InferenceEngine** (agent/inference_engine.py) | 4条硬编码 if/then: `cpu>8→server-class`。P16 的 TransitionMiner 从数据中自动发现因果, 不需要任何人写规则 |
| **build_cross_analysis()** (agent/fact_graph.py) | 手写 if/elif 查字典。P16 的因果边是统计驱动的, 系统自己学什么因导致什么果 |
| **_self_generate_experiment()** (agent/goal_generator.py) | 随机挑2个类别 + 模板字典查命令。P16 的 HypothesisEngine 按不确定性×可测试性评分选最高价值的假设 |
| **LLM 代码生成** (agent/online_agent.py) | 依赖 gemma4 CPU 76s/次。P16 全部用小模型 + 统计, 不需要LLM |
| **自适应实验** (goal_generator 的 self_goal) | 只是调高某类目标的概率。P16 是系统化的假设→实验→验证闭环 |

---

## 八、文件索引

| 文件 | 内容 |
|------|------|
| `agent/` | 所有核心模块 (20个 .py 文件) |
| `benchmark/` | 模板引擎 + 参数提取器 |
| `config/` | 命令注册, 参数规则, 创作提示 |
| `data/persistent/` | 持久化数据 (SQLite, PyTorch models, JSON) |
| `checkpoints/` | 旧版模型 |
| `run_logs/` | 运行日志 (.jsonl) |
| `scripts/` | 测试脚本 |
| `PLAN_P11_P15.md` | P11-P15 实施计划 (已完成) |
| `PLAN_P16_REASONING.md` | P16 推理层计划 (待实施) |
| `REASONING_LAYER_DESIGN.md` | Kimi 设计的推理层详细方案 (12KB) |
| `ARCH_REVIEW_P10_EVOLUTION.md` | Kimi 架构评审 |
| `AGENTS.md` | 项目级规则 + 当前架构说明 |
| `CHANGELOG.md` | 变更日志 |

---

## 九、运行手册

### 快速测试
```bash
cd /home/chillizu/Projects/Folunar_
docker rm -f folunar-sandbox 2>/dev/null
source .venv/bin/activate
timeout 120 python3 -u doc/script.py  # or inline python
```

### 长程运行
```bash
cd /home/chillizu/Projects/Folunar_
source .venv/bin/activate
nohup python3 -u scripts/marathon.py > marathon.log 2>&1 &
```

### 查看持久化状态
```bash
cd /home/chillizu/Projects/Folunar_
source .venv/bin/activate
python3 -c "
from agent.persistent_store import PersistentStore
import json, sqlite3
db = sqlite3.connect('data/persistent/folunar.db')
c = db.cursor()
c.execute('SELECT run_id, n_steps, success_rate FROM run_stats')
for r in c.fetchall(): print(f'{r[0]}: {r[1]} steps, {r[2]:.0%} success')
"
```

### 清理沙箱
```bash
docker rm -f folunar-sandbox
```

---

## 十、关键设计决策 (别改)

| 决策 | 原因 |
|------|------|
| 3个元意图(OBSERVE/CREATE/TRY)而非17具体 | 意图空间固定, 新行为不要求重训分类器 |
| Conductor 输出16维 tanh (非sigmoid) | 对比损失需要负值, 避免坍缩 |
| Nanny class_proj 投影 (非原型匹配) | 原型向量坍缩, 投影法更有判别力 |
| CPU训练, 4线程 | 热限制 105°C |
| Docker 非特权+无网络 | 隔离最大化 |
| 418命令池自发现 (非手写列表) | 去人为化 |
| 自适应概率(非 step%N) | 无需手工调参, 系统自动平衡 |
| Kimi 仅设计(不在闭环) | API不稳定 + 用户要求 |
| FactGraph 200节点自然饱和 | 沙箱信息有限, 不强制增长 |
| 推理引擎不依赖LLM | CPU太慢 (76s), 统计学因果就够了 |

---

## 十一、下一步做 P16 R1

R1 是最低风险的改动:
1. 给 FactGraph 节点加 `confidence`/`n_evidence` 字段
2. 在 step() 中记录 `(pre_state, action, post_state)` 到 transition 表
3. 跑 100 步验证记录正常工作
4. 不改变任何现有行为

开始实施前记得读 `PLAN_P16_REASONING.md`。
