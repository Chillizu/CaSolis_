# P16: 认知推理层 — 预测驱动的因果推理闭环 (PCRL)

> 基于: Kimi 咨询 + 认知科学/AI论文研究
> 状态: 计划阶段（未实现）
> 与之前的区别: 见后文「不是重复造轮子」

---

## 一、现有的 vs 新的

### 会保留的

| 模块 | 保留原因 |
|------|---------|
| **FactGraph** (节点/边/schema/缺口检测) | 基础知识存储, 升级为 BeliefGraph |
| **GrowingWorldModel V4** (核+叶+图注意力+自我反思) | 作为预测引擎, 生成预期输出 |
| **RND** (好奇心模块) | 防止假设重复 |
| **EpisodicMemory** (惊喜检测) | 记录惊讶事件作为反思信号 |
| **MetaCognitiveSelector** (3 MODE) | 新增 LEARN(REASON) 模式 |

### 会删除/替换的

| 旧代码 | 问题 | 替换为 |
|--------|------|--------|
| `InferenceEngine` (rule-based) | 硬编码 if/then, 不是真推理 | BeliefGraph + TransitionMiner 从数据学习 |
| `build_cross_analysis()` | 手写规则 (cpu>8→server类) | 因果边从 transition 表自动挖掘 |
| `_self_generate_experiment()` | 随机挑类别+模板查表 | HypothesisEngine 生成可验证假设 |

### 全新的

| 模块 | 做什么 | 之前存在？ |
|------|--------|-----------|
| **BeliefGraph** | FactGraph 升级: 边带置信度/证据/支持/反驳 | 从未 |
| **TransitionMiner** | 记录三态因果表 + 离线挖掘 | 从未 |
| **HypothesisEngine** | 生成可验证假设, 评分排序 | 从未 |
| **ExperimentPlanner** | 把假设转为沙箱命令, WM 预测 | 从未 |
| **Verdict & Reflection** | 验证/反驳, 更新信念, 反馈 WM | 从未 |

---

## 二、R1: 记录阶段 (今天)

**不改行为, 只收集数据。**

### 2.1 BeliefGraph 基础

在 `FactGraph` 节点/边结构上增加字段:

```python
class BeliefNode:
    confidence: float    # 0~1 (新增)
    n_evidence: int      # 观察到次数 (新增)

class BeliefEdge:
    weight: float        # 因果强度, 可正可负 (新增)
    n_support: int       # 支持数 (新增)
    n_against: int       # 反驳数 (新增)
    hypothesis_key: str  # 来源 (新增)
```

新增边类型:
- `CORRELATES` — 统计共现
- `CAUSES` — 干预验证后
- `PREDICTS` — 世界模型预测关系
- `INHIBITS` — 负相关

### 2.2 Transition 记录

每步记录:

```python
Transition = {
    "step": step_count,
    "pre_state": {key: value for key in changed_keys},
    "action": (intent, params, cmd_string),
    "post_state": {key: new_value for key in changed_keys},
    "exit_code": 0/1,
    "output_len": len(stdout),
    "had_new_facts": bool,
    "reward": float,
}
```

存在 `data/persistent/transitions.jsonl`, 滑动窗口 500 条。

### 2.3 验证标准

```
1. 跑100步, transition 表有 >= 50 条
2. BeliefGraph 所有现有节点自动获得默认 confidence=0.5
3. 不改变任何现有行为, 不增加每步耗时 > 5ms
```

---

## 三、R2: 离线因果挖掘 (下一步)

**从收集的数据中自动发现因果模式。**

### 3.1 TransitionMiner

算法 (轻量级, O(n²) 滑动窗口):

```
输入: transition 表 (最近 500 步)
输出: 候选因果边列表

每对 (A, B) 事实:
  P(B_change) = count(B变化) / total_obs
  P(B_change | A_change) = count(B变化 & A前变化) / count(A变化)
  gain = P(B_change|A_change) - P(B_change)
  
  if gain > 0.3 and A变化先于B变化:
    候选边 A --causes--> B, 权重 = gain
  if gain < -0.3:
    候选边 A --inhibits--> B, 权重 = -gain
  else if |gain| > 0.1:
    候选边 A --correlates--> B, 权重 = |gain|
```

限制:
- 窗口 500 步
- 只考虑变化的节点 (不变的不参与因果)
- 每批最多 50 对新边

### 3.2 HypothesisEngine

从高不确定性节点对生成假设:

```python
Hypothesis = {
    "if_node": "cpu_cores",
    "rel": "causes",
    "then_node": "load_average",
    "prediction": "cpu_cores 增加时 load_average 也增加",
    "uncertainty": 0.7,
    "testability": 0.8,
    "priority": 0.56,
}
```

评分:
```
priority = uncertainty(A) * uncertainty(B) * testability * (1 + |corr|)
```

优先选:
- 跨类别组合 (system × network, package × capability)
- 已有 CORRELATE 边但未验证 CAUSES 的
- RND 新颖度高的 (防重复)

### 3.3 验证标准

```
1. 从 500 步 transition 数据中挖出 >= 5 个可用假设
2. 至少 2 个假设跨类别
3. 没有重复假设 (RND 去重有效)
4. 可用离线脚本检查假设合理性
```

---

## 四、R3: 在线实验闭环 (再下一步)

**让系统自己验证假设。**

### 4.1 ExperimentPlanner

输入: 一个 Hypothesis
输出: 可在沙箱执行的命令 + 世界模型预测

三类实验:

| 类型 | 方法 | 示例 |
|------|------|------|
| 被动观察 | 读能同时看 A/B 的命令 | `cat /proc/loadavg` 看 load vs cpu |
| 主动干预 | 改 A 看 B | echo "stress" > /workspace/test → `uptime` |
| 反事实 | WM simulate() 预测不干预 | 对比实际执行 vs WM 预测 |

### 4.2 Verdict & Reflection

执行后:
```
预测 exit_code: 0, 实际: 0  →  支持 (+1)
预测 output_len: 200, 实际: 50  →  部分支持 (+0.5)
预测 B 变化: 增加, 实际: 没变  →  反驳 (-1)
```

更新 BeliefEdge:
```
weight += lr * (1 - weight)       # 支持
weight -= lr * (1 + weight)       # 反驳
n_support += 1 / n_against += 1
```

如果 `weight < -0.5` 且 `n_support == 0` → 移除该边。

### 4.3 GoalGenerator 集成

在 LEARN 模式下:
```python
if mode == "LEARN":
    hypothesis = hypothesis_engine.propose(belief_graph, top_k=3)
    if hypothesis:
        goal = Goal(type="hypothesis_test", intent="TRY",
                    params={"hypothesis_key": hypothesis.key})
```

### 4.4 安全约束

- 干预只允许在 `/workspace/` 内
- 所有实验命令走现有沙箱 + timeout 10s
- 每 10 步最多 1 次实验 (频率控制)
- 不覆盖 /etc /bin /proc 等重要路径

---

## 五、R4: 自我改进 (最终)

**让推理结果驱动行为变化。**

1. **Verdict 喂 WorldModel**
   - 把验证结果作为训练样本
   - WM 学会预测哪些实验会成功
   - 高置信度假设自动成为 GoalGenerator 的 gap_fill 目标

2. **MetaCognitiveSelector 增强**
   - 新增 `belief_confidence` 和 `wm_error` 输入
   - R9: 高优先级未验证假设 + 系统稳定 → LEARN(REASON)
   - R10: WM 误差持续高 → 强制 LEARN

3. **自动 schema 扩展**
   - 验证通过的因果边 → 注册为 schema 关系
   - 新的 gap_fill 目标从 schema 缺口生成

---

## 六、不是重复造轮子

| 之前试过的 | 问题 | R1-R4 不同 |
|-----------|------|------------|
| `InferenceEngine` (4 种规则) | 硬编码 if/then | 从 transition 数据自动学习 |
| `build_cross_analysis()` | 手写 `if cpu>8: server-class` | 因果边从共现中自发现 |
| `_self_generate_experiment()` | 模板查表随机组合 | 按 uncertainty×testability 选假设 |
| LLM 代码生成 | 76s CPU 太慢 | 不依赖 LLM, 纯统计+小模型 |
| 模板报告 | 我写的格式 | 无输出格式焦虑, 推理优先 |

核心突破: 从「我写规则告诉系统怎么推理」变成「系统自己从数据中发现因果」。
