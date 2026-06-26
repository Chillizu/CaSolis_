# Folunar_ — 项目规则与架构

> 最后更新: 2026-06-26
> 当前阶段: P11-P15 全部完成
> 状态: 持久化/知识拓展/自造工具/LLM/脑启发层 均已集成

---

## 工作规则

| 规则 | 说明 |
|------|------|
| Kimi 咨询决策 | **技术决策优先问 Kimi 而不是用户**。用 subagent 调用 kimi-coding.kimi-consultant, Kimi 的回复可直接执行, 除非它要求用户确认。报告时向用户总结做了什么和为什么 |
| 自主研究 | 需要查资料/论文时用 web_search / fetch_content / librarian 工具搜索, 不要每步都问用户 |
| 汇报风格 | 做完事直接汇报结果, 不需要每一步都征求许可。只有改变用户偏好的决策才需要问 |
| 分步实施 | 大改动拆成 P0→P1→P1.5→P2→... 的渐进步骤, 每步完成后再规划下一步 |
| 先读后写 | 改代码前先读完整文件, 理解上下文再动手 |
| 任务跟踪 | 用 todo 工具跟踪每一步进度 |

---

## 当前架构 (P11-P15 自主架构)

### 核心循环

```
state_text (环境状态)
    │
    ├─ MetaCognitiveSelector → MODE (EXPLORE/CREATE/LEARN)
    ├─ GoalGenerator → Goal 队列 (缺口/好奇心/创作)
    ├─ 想象力 (WorldModel 评分, 80%概率)
    ├─ Conductor (thought向量 → Nanny翻译, A/B)
    ├─ 分类器 (IntentClassifier, 16类, UCB偏差)
    │
    ▼
参数提取 (ParameterExtractor + _rescue_params)
    │
    ▼
模板引擎 (TemplateEngine → build_args → shell命令)
    │
    ▼
Docker沙箱 (folunar-sandbox, --network none)
    │
    ▼
ExecResult
    │
    ├─ Workbench.extract_facts (含 FactGraph)
    ├─ RND.compute_novelty (好奇心)
    ├─ CreativeWriter (异步LLM生成)
    ├─ EpisodicMemory (惊喜检测)
    ├─ ExperienceBuffer 存储
    ├─ 在线训练 (分类器+WM+Conductor)
    └─ 详细日志 (run_logs/*.jsonl)
```

### 模块清单

| 文件 | 功能 | 状态 |
|------|------|------|
| `agent/online_agent.py` | 主循环, 2600+ 行 | 生产 |
| `agent/workbench.py` | 事实提取 + 内容生成 (含 FactGraph 集成) | 生产 |
| `agent/fact_graph.py` | 图结构知识库: 节点+边+schema+缺口检测 | 生产 |
| `agent/meta_selector.py` | 3 MODE 选择器 (EXPLORE/CREATE/LEARN) + R8 置信度门控 | 生产 |
| `agent/goal_generator.py` | 模式驱动目标生成 + utility gate | 生产 |
| `agent/world_model_v4.py` | 增长型世界模型: 核+叶+图注意力+自我反思 | 生产 |
| `agent/episodic_memory.py` | 情景记忆: 惊喜环形缓冲 | 生产 |
| `agent/creative_writer.py` | Ollama LLM 插件: 异步生成 | 生产 |
| `agent/persistent_store.py` | 统一持久化: SQLite+PyTorch+version控制 | 生产 |
| `agent/knowledge_mapper.py` | 知识拓展+自发现引擎: 418命令 BFS + RND驱动 | 生产 |
| `agent/tool_factory.py` | 工具工厂: 安全命令池+新类别自动生成 | 生产 |
| `agent/tool_registry.py` | 工具注册表: 扫描+注册+统计 | 生产 |
| `agent/conductor.py` | thought 向量: 384→128→64→[16+11] | 生产 |
| `agent/nanny.py` | thought→intent 翻译 | 生产 |
| `agent/world_model.py` | 全局世界模型 (V3, 训练中) | 生产 |
| `agent/rnd.py` | RND 好奇心: 新颖度检测+兴趣重置 | 生产 |
| `agent/state_encoder.py` | 状态编码: 环境→state_text | 生产 |
| `agent/error_recovery.py` | 错误恢复: 分类器+回退 | 生产 |
| `agent/sandbox_executor.py` | Docker 沙箱: 持久容器+隔离 | 生产 |
| `agent/experience.py` | 经验回放缓冲区 | 生产 |
| `agent/detailed_logger.py` | JSONL 日志: 12+字段/步 | 生产 |

### 模型指标

| 模型 | 参数量 | 做什么 |
|------|--------|--------|
| IntentClassifier | ~66K | 状态文本 → 意图分类 |
| ConductorHead | ~67K | 状态文本 → 16维 thought 向量 |
| WorldModelNet V3 | ~0.5M | 全局预测 (exit/value/length/error) |
| GrowingWorldModel V4 | 核+16叶 | 逐意图预测 + 自动扩展 |
| RND | ~0.3M | 新颖度检测 + 好奇心奖励 |
| MiniLM (冻结) | ~90MB | 状态文本嵌入 |

### INTENTS (16 有效)

| 索引 | 意图 | 索引 | 意图 |
|------|------|------|------|
| 0 | READ | 8 | READ_ETC |
| 1 | LIST | 9 | USB_DEVICES |
| 2 | SEARCH | 10 | DISK_USAGE |
| 3 | INFO | 11 | LS_TMP |
| 4 | INSPECT | 12 | ARCH_INFO |
| 5 | COUNT | 13 | CUSTOM |
| 6 | EXPLORE | 14 | WRITE |
| 7 | (HELP, 无效) | 15 | APPEND |
| — | — | 16 | GENERATE |

---

## 当前指标 (P10 最终, 300步)

| 指标 | 值 |
|------|-----|
| 成功率 | 100% (300/300) |
| 意图覆盖 | 16/16 种 (各 ~6%) |
| 总奖励 | 512 |
| FactGraph 节点 | 86 |
| Schema 覆盖 | 44% |
| 沙箱内文件 | 80 个, ~192KB |
| LLM 异步 | 成功1次 (gemma4:e4b 9GB CPU 60s) |

---

## 实施路线图 (P11-P15)

见 `PLAN_P11_P15.md`

| 阶段 | 做什么 | 状态 |
|------|--------|------|
| P11 | 统一持久化 (PersistentStore + SQLite + Docker volume) | 待开始 |
| P12 | 知识拓展 (KnowledgeMapper: 双BFS探索沙箱) | 待开始 |
| P13 | 自造工具 (ToolFactory + ToolRegistry + TOOL 意图) | 待开始 |
| P14 | LLM 双模型 (qwen 快思 + gemma 深度) | 待开始 |
| P15 | 脑启发动态层 (图注意力 + 自我反思 + 置信度门控) | 待开始 |

---

## 关键设计决策 (别改)

| 决策 | 原因 |
|------|------|
| Conductor 输出 16 维 tanh (非 sigmoid) | 对比损失需要负值, 避免坍缩 |
| Nanny 使用 class_proj 投影 (非原型匹配) | 原型向量坍缩到 INFO, 投影法更有判别力 |
| A/B 门控 (非直接替换) | 可回退, 无风险 |
| Docker 非特权+无网络 | 隔离最大化, 防副作用 |
| CPU 训练, 限制 4 线程 | 热限制 105°C, torch.set_num_threads(4) |
| Kimi 仅设计用(不在闭环里) | 用户要求, 且 API 不稳定 |
| 安全写入 base64+python3 (非 shell 模板) | 防 shell 注入 |
| RND interest_reset() 每 20 步 | 防新颖度永久归零 |
| 想象力 80% 概率每步触发 | 最大化想象力使用率 |
| P10 不改变已训练模型角色 | 分类器/Conductor/WM 继续决策, LLM 只碰文本 |
| LLM 输出只进 opinion 节点 | 防幻觉污染 command-derived 事实 |

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

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `PLAN_P11_P15.md` | P11-P15 完整实施路线图 |
| `HANOFF.md` | P10 完成时的交接文档 |
| `ARCH_REVIEW_P10_EVOLUTION.md` | Kimi 对 P10 后进化的完整评审 |
| `data/persistent/` | 持久化数据目录 |
| `run_logs/` | 运行日志 |
| `checkpoints/` | 模型 checkpoint |
