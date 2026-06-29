# Folunar_ — 项目规则与架构

> 最后更新: 2026-06-29
> 当前状态: P11-P15 全部完成, P16 推理层 R1-R4 已实施, P17 自我意识层已实施
> 最后验证: 200 步 91% 成功率, LLM 自由格式输出 3 次, 自省意图产生
> 哲学: 小模型(分类器/Conductor)指决策, LLM(qwen3.5:0.8b)做自省+创作, CPU异步不堵塞

---

## 工作规则

| 规则 | 说明 |
|------|------|
| Kimi 咨询决策 | **技术决策优先问 Kimi 而不是用户**。用 subagent 调用 kimi-coding.kimi-consultant，Kimi 的回复可直接执行，除非它要求用户确认。报告时向用户总结做了什么和为什么 |
| 自主研究 | 需要查资料/论文时用 web_search / fetch_content / librarian 工具搜索，不要每步都问用户 |
| 汇报风格 | 做完事直接汇报结果，不需要每一步都征求许可。只有改变用户偏好的决策才需要问 |
| 分步实施 | 大改动拆成 P0→P1→P1.5→P2→... 的渐进步骤，每步完成后再规划下一步 |
| 先读后写 | 改代码前先读完整文件，理解上下文再动手 |
| 任务跟踪 | 用 todo 工具跟踪每一步进度 |
| 禁止 EMOJI | **任何时候都不使用 emoji**。AGENTS.md 本身也不应包含 emoji |
| 密钥管理 | HF token 等密钥存放在 AGENTS.md 的「密钥存储」章节，不写进代码 |

---

## 项目概述

系统是一个在 Docker 沙箱内自主探索的 Linux agent。通过小模型决策（什么模式、什么目标、什么意图），在沙箱里执行 Linux 命令，从输出中提取事实，存入图结构知识库(FactGraph)，基于已知事实发现缺口和假设，再生成新目标。

**不依赖 LLM 做决策**——LLM 仅用于报告风格的文本创作，且太慢(76s/次, CPU)所以频率很低。

核心循环:
```
元认知选择 MODE → 目标生成器提目标 → 分类器/Conductor 选意图
  → 沙箱执行命令 → 提取事实 → 更新知识图
  → 推理/假设 → 新目标 → 继续
```

---

## 当前架构 (P15 最终, 2600+ 行主循环)

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
    └─ InferenceEngine (规则式, 待替换为 P16)
```

### 完整历程

| 阶段 | 做了什么 | 状态 |
|------|---------|------|
| **P0-P9.7** | 核心闭环: 17意图, 100% 稳定, 模板驱动内容创作 | 旧架构, 已归档 |
| **P10** | 层级架构: MODE选择器(3种), GoalGenerator, GrowingWorldModel V4, EpisodicMemory, CreativeWriter, FactGraph | 生产 |
| **P11** | 统一持久化: PersistentStore(SQLite+PyTorch+Docker volume) | 已完成 |
| **P12** | 知识拓展: KnowledgeMapper, Phase A-E 探索, 418命令扫描 | 已完成 |
| **P13** | 自造工具: ToolFactory(JSON模板), ToolRegistry, 自发现引擎 | 已完成 |
| **P14** | LLM管道: qwen3.5+gemma4 双模型(但qwen不靠谱被回退到命令池) | 已完成 |
| **P15** | 脑启发动态层: WM V4 图注意力, 自我反思, 置信度门控, 去人为化(自适应奖励/动态StateEncoder/动态GoalGenerator) | 已完成 |
| **P16** | **推理层(R1-R4全部完成)**: TransitionMiner因果挖掘→HypothesisEngine假设生成→ExperimentPlanner实验→Verdict验证→WM反馈+自动schema+MODE自适应LEARN | **已实施** |
| **P17** | **自我意识层**: SelfModel(意图成功率/高光/自描述), LLM自省每隔20步问"你想做什么", qwen3.5:0.8b替代gemma4(2s vs 76s), 自由格式输出异步管道 | **已实施** |

### INTENTS (P15 重构后: 3 类元意图)

| 意图 | 做什么 |
|------|--------|
| **OBSERVE** | READ/LIST/INFO/SEARCH/INSPECT/COUNT — 观察环境 |
| **CREATE** | WRITE/CREATE — 创作内容到沙箱 |
| **TRY** | 执行命令/工具/实验 — 不确定结果的行为 |

**为什么从 17 个具体意图减到 3 个**: 意图空间固定, 新行为不要求重训分类器。具体命令由模板引擎+418命令池动态决定。

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

### 去人为化 (P15 已完成)

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

## 当前指标 (马拉松 27,799 步)

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

## 系统能/不能做什么

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

### 不能 (仍有差距)

| 缺失 | 原因 |
|------|------|
| **原创代码生成** | qwen3.5:0.8b 可以写简单脚本但 gemma4 太慢, LLM代码非核心 |
| **跨session学习** | PersistentStore存了但KnowledgeMapper每轮重扫, 不真正积累理解 |
| **稳定输出质量** | LLM 有幻觉(编日期/节点名), quality check 只验数值 |

## P16: 推理层 (已实施 R1-R4)

**核心转变**: 从「写规则告诉系统怎么推理」变成「系统自己从数据中发现因果」。

已替换的旧代码:

| 旧代码 | 问题 | 替换为 (已实施) |
|--------|------|----------------|
| `InferenceEngine` (4条硬编码规则) | `if cpu>8: server-class` | TransitionMiner 从 transition 数据自动学习 |
| `build_cross_analysis()` (手写if/then) | 规则不能自发现 | 因果边从共现中自动挖掘 |
| `_self_generate_experiment()` (模板查表) | 随机组合无意义 | HypothesisEngine 按 uncertainty×testability 选假设 |

5个新模块(全部不依赖LLM):

| 模块 | 做什么 | 状态 |
|------|--------|------|
| **FactGraph 升级** | 边带 n_support/n_against/hypothesis_key, 新边类型 CORRELATES/CAUSES/PREDICTS/INHIBITS | 已实施 (R1) |
| **TransitionMiner** | 从 transition 数据挖因果边 (P(B|A)-P(B) 增益) | 已实施 (R2) |
| **HypothesisEngine** | 生成假设, 按 uncertainty×testability×(1+\|corr\|) 排序+去重 | 已实施 (R2) |
| **ExperimentPlanner** | 把假设转为沙箱命令 (被动观察/主动干预), 安全约束 | 已实施 (R3) |
| **Verdict** | 比较预测vs实际, 更新边权重, 移除弱边, 反馈WM | 已实施 (R3) |

R4 自我改进: Verdict→WM 反馈训练, 自动 schema 扩展, MetaCognitiveSelector R9/R10 自适应 LEARN 模式。

## P17: 自我意识层 (已实施)

**核心转变**: 从「分析环境」到「认识自己」。

系统不再是只响应 gap/mode 驱动:

| 之前 | 之后 |
|------|------|
| FactGraph facts → gap → goal | Self stats + intent history → LLM自省 → "我想做X" |
| 写报告 = 模板拼合 | LLM 异步自由格式输出, 模板只兜底 |
| LLM (gemma4) 76s/次, 频率极低 | qwen3.5:0.8b, 2s/次, 每20步自省 |

| 模块 | 做什么 | 文件 |
|------|--------|------|
| **SelfModel** | 追踪每类意图的成功率、高光时刻、创作记录、生成自描述 | `agent/self_model.py` |
| **SelfReflect** | LLM 自省: "根据你对自己的了解, 想做什么?" → CREATE goal | 内嵌于 `creative_writer.py` + `online_agent.py` |
| **Free-form pipeline** | 每5步触发 async LLM 生成, 结果自动消费写沙箱 | `online_agent.py` step() |

自省示例 (qwen3.5:0.8b 实际输出):
> "I want to explore the file system and write a script that summarizes all my discoveries using Python."
详见 `PLAN_P16_REASONING.md` 和 `REASONING_LAYER_DESIGN.md`。


## 关键目录

| 目录 | 用途 |
|------|------|
| `agent/` | 所有核心模块 (20个 .py 文件) |
| `benchmark/` | 模板引擎 + 参数提取器 |
| `config/` | 命令注册, 参数规则, 创作提示 |
| `data/persistent/` | 持久化数据 (SQLite, PyTorch models, JSON) |
| `checkpoints/` | 旧版模型 checkpoint |
| `run_logs/` | 运行日志 (.jsonl) |
| `scripts/` | 测试和运行脚本 |
| `doc/` | 文档和计划文件 |

---

## 开发命令

### 快速测试
```bash
cd /home/chillizu/Projects/Folunar_
docker rm -f folunar-sandbox 2>/dev/null
source .venv/bin/activate
timeout 120 python3 -u doc/script.py
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
import sqlite3
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

## 关键设计决策 (别改)

| 决策 | 原因 |
|------|------|
| 3个元意图(OBSERVE/CREATE/TRY)而非17具体 | 意图空间固定, 新行为不要求重训分类器 |
| Conductor 输出16维 tanh (非sigmoid) | 对比损失需要负值, 避免坍缩 |
| Nanny class_proj 投影 (非原型匹配) | 原型向量坍缩, 投影法更有判别力 |
| A/B 门控 (非直接替换) | 可回退, 无风险 |
| Docker 非特权+无网络 | 隔离最大化, 防副作用 |
| CPU训练, 限制4线程 | 热限制 105°C, torch.set_num_threads(4) |
| Kimi 仅设计用(不在闭环里) | 用户要求, 且 API 不稳定 |
| 安全写入 base64+python3 (非 shell 模板) | 防 shell 注入 |
| 418命令池自发现 (非手写列表) | 去人为化 |
| 自适应概率(非 step%N) | 无需手工调参, 系统自动平衡 |
| 推理引擎不依赖LLM | CPU太慢 (76s), 统计学因果就够了 |
| FactGraph 200节点自然饱和 | 沙箱信息有限, 不强制增长 |
| RND interest_reset() 每 20 步 | 防新颖度永久归零 |
| 想象力 80% 概率每步触发 | 最大化想象力使用率 |
| 奖励自适应学习 | 替代手写 INTENT_REWARD |
| SAFE_COMMANDS 可被自发现覆盖 | 逐步去人为化 |

---

## 对话技巧与模式 (持续积累)

| 模式 | 应对策略 |
|------|---------|
| **"直接做, 别问"** | 用户不喜欢每步被问意见。做完直接汇报结果, 除非改变偏好的决策才问 |
| **渐进式改进** | 用户偏好"先把1,2做了"的逐步推进。大方向拆成子任务, 每完成一个汇报一次 |
| **先做再说** | 用户倾向先实施再讨论。方案不需要预先审批, 做完看效果再调整 |
| **提问即探索** | "能创造什么"这类问题不是要即时答案, 而是要求盘点真实产出。需要实际检查沙箱/日志给出量化答案 |
| **挑战即确认** | "嗯哼？" 表示用户在确认是否已经理解了当前的方案，而不是质疑。应当直接,自信总结 |
| **全部都要** | 当用户说"全部都要解决"时, 不是要一次搞定; 而是要在方案里全部覆盖, 分步实施 |
| **底层优先** | 持久化 → 知识 → 工具 → LLM → 脑启发; 每一步为下一步打基础 |
| **问 Kimi, 别问用户** | 遇到技术问题或需要做决策时, 优先用 subagent 问 kimi-for-coding, 而不是问用户 |

---

## 密钥存储

| 密钥 | 值 | 用途 |
|------|-----|------|
| HF_TOKEN | `YOUR_HF_TOKEN` | HuggingFace 访问 token |
| DEEPSEEK_KEY | `YOUR_DEEPSEEK_KEY` | DeepSeek API (临时代码生成) |

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `agent/` | 所有核心模块 (20个 .py 文件) |
| `PLAN_P11_P15.md` | P11-P15 实施计划 (已完成) |
| `PLAN_P16_REASONING.md` | P16 推理层计划 (待实施) |
| `REASONING_LAYER_DESIGN.md` | Kimi 设计的推理层详细方案 |
| `HANOFF.md` | 完整交接文档 (P15 最终状态) |
| `ARCH_REVIEW_P10_EVOLUTION.md` | Kimi 架构评审 |
| `CHANGELOG.md` | 变更日志 |
| `data/persistent/` | 持久化数据目录 |
| `run_logs/` | 运行日志 |
| `checkpoints/` | 模型 checkpoint |
| `main_session_*.jsonl` | PI 平台历史对话 (上下文继承用) |
