# Changelog — Folunar_

> 项目级操作日志, 记录每次里程碑变更。
> 格式: `[日期] 阶段 — 做了什么`

---

## [2026-06-25] P10 — 层级架构完成

### 新增模块
- `agent/fact_graph.py` — FactGraph: dict+邻接表图结构知识库, schema 缺口检测, 交叉推理
- `agent/meta_selector.py` — MetaCognitiveSelector: 3 MODE (EXPLORE/CREATE/LEARN)
- `agent/goal_generator.py` — GoalGenerator: 模式驱动目标生成, utility gate, CUSTOM 过滤
- `agent/world_model_v4.py` — GrowingWorldModel V4: 核+叶架构, 逐意图预测, 自动扩展
- `agent/episodic_memory.py` — EpisodicMemory: 惊喜环形缓冲, 余弦相似度+衰减召回
- `agent/creative_writer.py` — CreativeWriter P0-P2: Ollama LLM 异步生成, 4风格, 自适应频率

### 修改
- `agent/online_agent.py` — step() 重写为 MODE→GOAL→ACTION 流程
- `agent/workbench.py` — FactGraph 集成, cross_analysis/change_report/fact_history 委派
- `agent/conductor.py` — checkpoint 自适应收缩 (19→17)
- `agent/goal_generator.py` — CUSTOM 30% 硬过滤, 条件 follow_up, force_create 每40步
- `agent/creative_writer.py` — prompt 缩减为 8 条 facts 加速推理

### 当前指标 (300 步)
- 100% 成功率, 16/16 意图均匀覆盖
- FactGraph 86 节点, Schema 44%
- 沙箱内 80 文件, ~192KB 自主内容
- LLM 异步成功 1 次

### 已知问题
- PersistentStore 不存在: docker rm -f 后一切归零
- 知识面窄: 只读 /proc/cpuinfo /etc/os-release /free -h
- LLM 慢: gemma4:e4b 9GB CPU ~60s
- 无自造工具: 只能写 shell 模板

---

## [2026-06-24] P9.7 — 层级架构规划与前期工作

### 新增
- `config/creative_prompts.yaml` (后改为 creative_writer.py 内联)
- `scripts/test_creative_writer.py`

### 指标
- 16 意图 100% 成功率 (3连)
- 沙箱内 15 文件, 28KB
- RND 新颖度 0.003→0.078

---

## [2026-06-24] P9.6 — GENERATE + 推理

- `build_generate_content()`: profile/experiment/discovery_log 三种生成
- `_build_cross_analysis()`: 跨事实推理
- `_build_change_report()`: 变化检测
- `_track_fact_history()`: 历史追踪

---

## [2026-06-24] P9.5 — 内容深度

- `_build_fact_analysis()`: 自然语言总结
- 脚本从 echo OK 升级为真实验证 (对比+WARN+退出码)
- 100% 3连

---

## [2026-06-21] P9.4 — 100% 突破

- 消灭最后 7% 失败 (DISK_USAGE/INFO/CUSTOM)
- `_extract_generic` 通用提取
- 事实 18→40
- 首次 100%

---

## [2026-06-18] P9.3 — 内容生成器

- `build_write_content()`: JSON/Markdown/Shell 三种风格
- SEARCH/USB_DEVICES 修复
- 93→99%

---

## [2026-06-18] P8.0-P8.5d — 多重改进

- 错误恢复 34%→87%
- 参数推断增强
- 多样性强化 (13/13)
- UCB+LR 调度
- 模板外化
- 动态探针
- 自由扩展命令
- 65→93%

---

## [2026-06-17] P7.0-P7.2 — 稳定化

- CUSTOM 维度统一
- 核心流修复
- 300 步稳定
- 70→75%

---

## [2026-06-11] P6.0-P6.4 — 探针+事实去噪

- 探针闭环
- 脚本模板
- 新颖度维护
- 新意图自动连接
- 65→70%

---

## [2026-06-10] P5.0-P5.6 — Conductor + 多样性

- Conductor (thought 向量)
- 多样性调度
- 逆想象
- 元学习
- 50→65%

---

## [2026-06-09] P0-P5 — 核心闭环

- 状态编码→分类器→模板引擎→Docker 沙箱
- 第一个可运行版本
- 30→50%
