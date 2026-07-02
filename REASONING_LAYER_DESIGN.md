# CaSolis_ 认知推理层设计方案 (P16)

> 目标: 在现有 MODE -> GOAL -> ACTION 层级 + FactGraph + GrowingWorldModel V4 之上,
> 增加一个不以 LLM 为核心的真正推理层, 使系统能自发提出假设、设计实验、验证/修正因果信念。
> 约束: CPU-only, 无 GPU, Docker 沙箱隔离, 小模型决策。

---

## 1. 现状与缺口

当前系统已经具备:
- `FactGraph`: 200+ 节点的事实图, 有节点、边、schema、缺口检测。
- `MetaCognitiveSelector`: EXPLORE / CREATE / LEARN 三层 MODE。
- `GoalGenerator`: 动态生成 gap_fill / try_command / content_create / verify 目标。
- `GrowingWorldModel V4`: 核+叶, 能按意图预测 exit/length/error/value/next_thought/agreement。
- `KnowledgeMapper` + `ToolRegistry`: 命令与工具的发现、注册、复用。

缺口:
- `build_cross_analysis()` 仍是手写 if/then 规则 (cpu>8 -> server-class 等)。
- 边 (`requires/extends/verifies/conflicts`) 是手工预设或 schema 自动生成, 没有从数据中学习。
- 没有"假设 -> 实验 -> 验证 -> 修正"的闭环。
- 世界模型预测误差没有被用来驱动主动学习。

---

## 2. 核心设计原则

1. **小模型决策, 符号 + 统计推理, LLM 仅做报告装饰。**
2. **把 FactGraph 升级成 BeliefGraph: 每个节点/边都有置信度和证据计数。**
3. **把世界模型当作生成模型: 假设是预测, 实验是采样, 误差是惊讶, 反馈修正信念。**
4. **因果发现靠观察 + 最小干预, 不做全量 PC/FGES (算不起)。**
5. **增量实现: 先记录, 再离线挖, 再在线验证, 最后自动改进。**

---

## 3. 总体架构

```
┌──────────────────────────────────────────────┐
│  元认知层 (MetaCognitiveSelector)            │
│  MODE = EXPLORE | CREATE | LEARN(REASON)     │
│  新增输入: belief_confidence, wm_error       │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│  目标层 (GoalGenerator)                      │
│  新增 goal_type = hypothesis_test            │
│  输入: FactGraph.gaps + candidate_hypotheses │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│  动作层 (ActionExecutor)                     │
│  执行 ExperimentPlanner 生成的沙箱命令       │
└──────────────────────┬───────────────────────┘
                       │
   ┌───────────────────┼───────────────────┐
   ▼                   ▼                   ▼
┌──────────┐    ┌──────────────┐    ┌─────────────┐
│ Belief   │◄──►│ Transition   │◄──►│ Growing     │
│ Graph    │    │ Miner        │    │ World Model │
│ (FactGraph│   │ (causal      │    │ V4          │
│  upgrade)│    │  discovery)  │    │             │
└────┬─────┘    └──────┬───────┘    └──────┬──────┘
     │                 │                   │
     ▼                 ▼                   ▼
┌──────────────────────────────────────────────┐
│  HypothesisEngine -> ExperimentPlanner       │
│  -> Verdict -> Reflection -> Feedback        │
└──────────────────────────────────────────────┘
```

---

## 4. 模块设计

### 4.1 BeliefGraph (FactGraph 升级)

在现有 `Node` 和 `Edge` 上增加不确定性字段:

```python
class BeliefNode(Node):
    confidence: float   # 0~1, 证据越充分越接近 1
    n_evidence: int     # 观察到该事实的次数
    last_updated: int   # step

class BeliefEdge:
    rel: str
    weight: float       # 因果/相关强度, 可正可负
    n_support: int      # 支持该边的观测数
    n_against: int      # 反驳该边的观测数
    hypothesis_key: str # 由哪个假设生成
```

新增边类型:
- `CORRELATES`: 统计共现
- `CAUSES`: 干预 A 后 B 改变
- `PREDICTS`: 世界模型学到的预测关系
- `INHIBITS`: 负相关

所有预设边 (`requires/extends/verifies`) 保留, 但权重从 1.0 改为可学习, 并允许被反驳。

### 4.2 Transition Miner (因果发现)

每一步记录三元组:

```python
Transition = {
    "step": int,
    "pre_state": {key: value},
    "action": (intent, params, cmd),
    "post_state": {key: value},
    "duration": float,
    "exit_code": int,
}
```

从 transition 表中学习局部因果:

1. **共现统计**: 对任意事实对 (A, B), 统计 `P(B_change | A_change)` 和 `P(B_change)`。
2. **增益分数**: `gain = P(B_change|A_change) - P(B_change)`。
3. **时序优先**: 只在 A 变化先于 B 变化时加分。
4. **干预检测**: 如果 action 明确写/改了 A, 且随后 B 改变, 给 `CAUSES` 边更高权重。
5. **最小假设**: 只保留 top-k 高分边, 避免组合爆炸。

算法复杂度 O(n_facts^2 * window), 用滑动窗口限制最近 500 步, CPU 可承受。

### 4.3 HypothesisEngine (假设生成)

把假设表示为可验证的预测:

```python
Hypothesis = {
    "key": str,
    "if_node": str,      # A
    "rel": str,          # causes / predicts / inhibits
    "then_node": str,    # B
    "predicted_effect": str,  # e.g. "B.value increases", "B appears"
    "confidence": float,
    "testability": float,
    "novelty": float,
    "priority": float,
}
```

生成策略:
- 从 BeliefGraph 的高不确定性节点中挑 A 和 B。
- 优先选已有 `CORRELATES` 边但尚未验证 `CAUSES` 的对。
- 优先选跨类别组合 (system + network, package + capability 等)。
- 用 RND novelty 防止重复同一假设。

评分:
```
priority = uncertainty(A) * uncertainty(B)
         * testability(A, B)
         * (1 + novelty)
         * (1 + |correlation_weight|)
```

每次只生成 top-5 候选, 避免假设爆炸。

### 4.4 ExperimentPlanner (实验设计)

输入: 一个 Hypothesis。
输出: 一条可在沙箱执行的命令 + 预测结果。

三类实验:

| 假设类型 | 实验方式 | 示例 |
|---------|---------|------|
| 被动观察型 | 读取能同时观测 A/B 的命令 | `cat /proc/net/dev` 验证 ip_addr 与网络流量 |
| 主动干预型 | 在 /workspace 创建/修改 A, 观察 B | 创建文件后 `ls` 验证 LIST 能发现它 |
| 反事实型 | 用 WM V4 `simulate()` 预测不执行 A 时的 B | 与真实执行对比, 估计因果效应 |

规划步骤:
1. 查询 `KnowledgeMapper` / `ToolRegistry`, 找能观测 A 和 B 的命令/工具。
2. 若 A 可写 (如文件), 生成一个最小干预脚本。
3. 用 `GrowingWorldModel.simulate()` 预测 exit_code 和关键输出。
4. 选择预期 **预测误差最大** 或 **信息增益最高** 的实验 (主动推理)。
5. 通过 `TemplateEngine` 打包成安全命令, 进入 Docker 沙箱。

### 4.5 Verdict & Reflection (验证与反馈)

执行后:
1. `Workbench.extract_facts()` 更新 post_state。
2. 比较预测 vs 实际:
   - exit_code 是否一致?
   - B 是否按假设变化?
   - 是否有新事实出现?
3. 更新 BeliefEdge:
   - 支持: `n_support += 1`, `weight += learning_rate * (1 - weight)`
   - 反驳: `n_against += 1`, `weight -= learning_rate * (1 + weight)`
4. 更新 `GrowingWorldModel` 训练样本, 降低未来预测误差。
5. 把验证结果写入 EpisodicMemory 作为"惊讶事件"。
6. 如果 WM 平均误差持续高, `MetaCognitiveSelector` 切到 LEARN 模式。

---

## 5. 与现有层级架构的集成

### 5.1 MODE 层

`MetaCognitiveSelector.select()` 增加两个输入:
- `belief_confidence`: BeliefGraph 平均边置信度。
- `wm_prediction_error`: 最近 20 步世界模型误差。

新增规则 (R9/R10):
- R9: 如果存在高优先级未验证假设且系统稳定 -> 进入 LEARN(REASON)。
- R10: 如果 WM 预测误差 > 阈值 -> 强制 LEARN, 优先做能修正模型的实验。

### 5.2 GOAL 层

`GoalGenerator.generate()` 增加候选类型:
```python
if mode == "LEARN":
    candidates += hypothesis_engine.propose_goals(belief_graph, top_k=3)
```

每个 hypothesis 生成一个 `Goal(type="hypothesis_test", intent="TRY", ...)`。

### 5.3 ACTION 层

在 `OnlineAgent.step()` 中, 当目标类型是 `hypothesis_test`:
1. 调用 `ExperimentPlanner.plan(goal.hypothesis)`。
2. 执行命令。
3. 调用 `Verdict.update()`。
4. 不依赖 LLM, 全部用现有小模型 + 符号逻辑完成。

---

## 6. 关键流程伪代码

```python
# 每步主循环
mode = meta_selector.select(belief_stats)
goal = goal_generator.generate(mode, belief_graph, wm_stats)

if goal.type == "hypothesis_test":
    plan = experiment_planner.plan(goal.hypothesis, belief_graph, world_model)
    pred = plan.predicted
    result = sandbox.execute(plan.cmd)
    actual = workbench.extract_facts(result)
    verdict = verdict_module.judge(goal.hypothesis, pred, actual)
    belief_graph.update_edge(verdict)
    world_model.update(plan.state_emb, plan.thought, plan.intent,
                       result.exit_code, result.output, verdict.reward)
    episodic_memory.store_if_surprising(verdict)
else:
    result = action_executor.run(goal)

meta_selector.update_confidence(world_model.get_confidence())
```

---

## 7. 实施顺序 (从简单到复杂)

### R1: 记录与测量 (1-2 天)
- 给 `FactGraph` 增加 `confidence` / `n_evidence`。
- 在 `OnlineAgent` 中持久化 `(pre_state, action, post_state)` transition 表。
- 不改变行为, 只收集数据。

### R2: 离线因果挖掘 (2-3 天)
- 实现 `TransitionMiner`: 从 transition 表算共现、增益、时序优先。
- 实现 `HypothesisEngine`: 生成 candidate hypotheses, 按 priority 排序。
- 提供一个离线脚本, 人工检查前 10 个假设是否合理。

### R3: 在线实验闭环 (3-4 天)
- 新增 `ExperimentPlanner` 和 `Verdict`。
- 在 `GoalGenerator` 中加入 `hypothesis_test` 候选。
- 只在 LEARN 模式下启用, 控制频率 (每 10 步最多 1 次实验)。
- 所有实验命令走现有安全白名单 + timeout。

### R4: 自我改进 (2-3 天)
- 把 verdict 结果喂给 `GrowingWorldModel` 训练。
- 用 WM 误差驱动 `MetaCognitiveSelector` 切模式。
- 把验证后的因果边自动扩展 schema, 成为后续 gap_fill 的目标。

---

## 8. 评估指标

| 指标 | 目标 |
|------|------|
| 假设生成数 / 100 步 | >= 5 |
| 实验执行数 / 100 步 | >= 3 |
| 验证为真的假设比例 | >= 40% |
| 手写 cross-analysis 规则被替换比例 | >= 50% |
| WM 预测误差 (验证型实验) | 相对 R1 下降 30% |
| 新增 CAUSES/PREDICTS 边数 / 100 步 | >= 2 |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 因果发现把相关当因果 | 中 | 坚持"时序优先 + 干预验证", 置信度低时不写入强边 |
| 假设爆炸导致大量无效实验 | 中 | top-5 限制, 用 RND novelty 去重, 最小持有步数 |
| 主动干预破坏沙箱状态 | 中 | 只允许 /workspace 内写操作, 白名单 + timeout |
| WM 预测不准, 误导 verdict | 中 | verdict 同时看实际输出, 不把 WM 当唯一真理 |
| CPU 开销增加 | 低 | transition 表用滑动窗口, 因果挖掘每 50 步做一次批量 |
| 反馈闭环延迟 | 中 | 优先做低成本读实验, 少做高成本写实验 |

---

## 10. 为什么这是最简可行架构

- **不引入新的大模型**: 复用 MiniLM + GrowingWorldModel V4, 只增加轻量统计模块。
- **不破坏现有层级**: MODE -> GOAL -> ACTION 不变, 只是新增一种 goal 类型和一组后台模块。
- **可逐步验证**: R1 只记录, R2 离线跑, R3 在线小范围启用, 风险可控。
- **与用户哲学一致**: 推理是核心, LLM 只用于报告装饰; 创造是系统自身行为, 不是模板。
