# P18 最终计划: WorldModel V5 + DoCalculus + Intuition Module

> 2026-06-29 | 基于 6 子代理研究 + 12+ 次 web 搜索验证
> 参考: LoopWM, TRM, Tab-TRM, NextLat, DreamerV3, NEC, DoWhy, pgmpy


> **⚠️ 关键修正 (基于 oracle 架构评审 + 两次 DeepSeek 批评):**
>
> | 原计划问题 | 修正 |
> |-----------|------|
> | RSSM stochastic latent 32×4=58K 在确定环境无意义 | **去掉 stochastic, GRU hidden 256→192 → ~370K** |
> | Do-calculus 需要正确因果图, FactGraph 80% 是相关 | **先做干预评分, 图验证后再加 do-calculus** |
> | Phase 3 依赖 Phase 1 (不正确) | **Phase 3 用现有 thought 向量即可, 立即启动** |
> | 参数预算 ~968K 到天花板 | **降到 ~644K (直接减 324K)** |
> | CounterfactualHead 干预编码未具体化 | **先解决编码问题再实现** |
> | 热管理未考虑 | **训练节流: >95°C 暂停 30s, 85°C 恢复** |
>
> **修正后执行策略:**
> - Week 1 (并行): Phase 1-LITE (确定性 GRU ~370K) + Phase 3-LITE (余弦相似度直觉缓冲)
> - Week 2 (并行): Phase 1b (CF head) + Phase 2-LITE (干预评分)
> - Week 3: Phase 2-FULL (do-calculus, 仅在图验证后)
---

## 当前瓶颈

```
小模型阵列 (~100K-0.5M) → 决策/意图/thought
         ↓
DeepSeek API → 创作/代码/自由输出 (OK)
```

DeepSeek 侧已解决。瓶颈全在小模型阵列:

| 模块 | 现状 | 问题 | 修复方向 |
|------|------|------|----------|
| WorldModel V4 | 仅预测 reward | 不知道 `ls /proc` 会输出什么 | 状态转移预测 (LoopWM) |
| TransitionMiner | P(B\|A)-P(B) | 统计关联 ≠ 因果 | Do-calculus 后门准则 |
| Conductor | 16-dim thought 是压缩 | 不是灵感/直觉 | DND 记忆检索 |

---

## 架构升级总览

所有新模块仍 <1M params, CPU-only, PyTorch, 零新依赖。

```
                    FactGraph (200 nodes)
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
WorldModel V5        DoCalculusEngine      Intuition Module
(LoopWM-style)       (纯 Python)           (DND + 原型记忆)
~250K params          ~500 lines 零依赖      ~100K + 存储
    │                     │                     │
    │                     │                     │
    ▼                     ▼                     ▼
state → next_state    P(B|do(A))-P(B|do(∅)) 熟悉度/这像X/方向
反事实模拟             confounder 发现        直觉模式匹配
```

---

## Phase 1: WorldModel V5 — LoopWM-style

### 为什么不是传统 RSSM

| RSSM | LoopWM-style (选) |
|------|--------------------|
| GRU + stochastic cat 32x4 | **单个共享 transformer block** |
| ~717K params | **~250K params** |
| 固定深度 | **自适应深度 (1-8 次循环)** |
| 时序状态靠 GRU 隐藏 | **靠迭代 refinement** |

LoopWM 论文 (2606.18208, Jun 2026) 证明了 **100x 参数效率**: 参数共享的 transformer block 迭代精炼 latent state, 谱约束保证长期稳定性。

### 架构

```
state_emb (384-dim) + action (intent 3 + params)
         │
    ┌────▼──────────────────────────┐
    │ Prelude MLP (384+32 → 128)    │  ~20K params
    └────┬──────────────────────────┘
         │ e (128-dim conditioning)
    ┌────▼──────────────────────────┐
    │ Recurrent Block × T 次循环     │  ~200K params (共享)
    │ h_{t+1} = A·h_t + B·e + F(h_t)│  谱约束: A=diag(-exp(a))
    │ F = shared transformer block   │  每次循环都一样
    └────┬──────────────────────────┘
         │ h (128-dim final)
    ┌────▼──────────────────────────┐
    │ Prediction Heads              │  ~30K params
    │ next_state, reward, continue  │
    │ counterfactual delta (16.5K)  │
    └───────────────────────────────┘
```

### 关键设计

| 组件 | 细节 |
|------|------|
| Prelude | 2 层 MLP, 编码 state+action →  conditioning e |
| Recurrent Block | 1 层 transformer (self-attn + FFN), 参数共享 |
| 谱约束 | A=diag(-exp(a)), ZOH 离散化 → 特征值在 (0,1) |
| 自适应深度 | 简单状态 1-2 次, 复杂 4-8 次, entropy-based 早退 |
| 训练损失 | MSE(next_state) + 0.1 × KL + reward + continue |
| 反事实头 | Linear(128,128) 预测 delta, 仅 +16.5K params |

### 与现有系统集成

```
Current: WorldModelV4.simulate(state_emb, thought, intent)
New:    WorldModelV5.predict(state_emb, action)
  → next_state_emb (384-dim)  ← 和 MiniLM 嵌入空间一致
  → reward/value
  → counterfactual(next_emb | do(action'))
```

- next_state_emb 可直接与 MiniLM 嵌入比较 (余弦距离)
- counterfactual 输入到 GoalGenerator → "如果做 X 会怎样"
- 训练数据: transition buffer (600 条, 滑动窗口)

### 实现估算

| 子任务 | 代码 | 参数 |
|--------|------|------|
| Prelude + Recurrent Block | ~200 行 | ~220K |
| Prediction Heads (next, reward, continue) | ~100 行 | ~30K |
| CounterfactualHead + 反事实模拟 | ~80 行 | ~16.5K |
| 训练循环 + transition 适配 | ~150 行 | - |
| OnlineAgent 集成 | ~80 行 | - |
| **合计** | **~610 行** | **~266K** |

---

## Phase 2: DoCalculusEngine — 纯 Python 因果推理

### 为什么不是现有库

| 库 | 大小 | 结论 |
|----|------|------|
| DoWhy | ~180MB (numpy+scipy+pandas+sklearn) | ❌ |
| causal-learn | ~250MB | ❌ |
| pgmpy | ~200MB | ❌ |
| **自写 kernel** | **~500 行, 0 依赖** | **✅** |

### 架构

```
FactGraph (200 nodes, CAUSES/PREDICTS edges)
         │
    ┌────▼──────────────────────────┐
    │ DAG Builder                   │
    │ fact_adj = graph→adjacency    │
    │ moralize + ancestral graph    │
    └────┬──────────────────────────┘
         │ DAG
    ┌────▼──────────────────────────┐
    │ DoCalculusEngine              │
    │ d_separated(X,Y,Z) → bool     │
    │ do_operation(nodes) → Graph   │
    │ find_backdoor(T,Y) → set Z    │
    │ find_frontdoor(T,Y) → set Z   │
    │ estimate_ate(T,Y,Z) → float   │
    └────┬──────────────────────────┘
         │ estimand + ATE
    ┌────▼──────────────────────────┐
    │ 集成到现有管道                  │
    │ TransitionMiner → 加调整集     │
    │ HypothesisEngine → ATE 优先级   │
    │ ExperimentPlanner → confounder │
    │ Verdict → ATE 替代 weight     │
    └───────────────────────────────┘
```

### 核心算法 (50 行/个)

- **d-separation**: BFS on moralized ancestral graph, O(V+E)
- **Back-door criterion**: d-sep + ancestor check, O(V²)
- **ATE estimation**: stratification by adjustment set, O(S × |Z|)
- **Do-calculus rules 1-3**: graph manipulation + d-sep, O(V³)

### 实现估算

| 子任务 | 代码 |
|--------|------|
| DAG Builder + d-separation | ~100 行 |
| Back-door + Front-door criterion | ~100 行 |
| ATE estimation | ~80 行 |
| Do-calculus rules + ID algorithm | ~120 行 |
| TransitionMiner / HypothesisEngine / Verdict 集成 | ~150 行 |
| **合计** | **~550 行** |

---

## Phase 3: Intuition Module — DND + 原型记忆

### 为什么不是更大模型

| 方案 | 参数量 | 速度 |
|------|--------|------|
| Larger Conductor | +200K | 慢 |
| **DND (选)** | **~0 (纯存储)** | **<100μs** |
| Prototypical | ~0 | O(K×d) |

### 架构

```
Conductor → thought vector (16-dim)
                  │
           ┌──────▼──────────────────┐
           │ DND (per intent)         │
           │ key: 16-dim thought      │
           │ value: avg Q/reward      │
           │ kd-tree 检索 top-5       │
           │ 4096 entries per intent  │
           └──────┬──────────────────┘
                  │
           ┌──────▼──────────────────┐
           │ 3 种输出                 │
           │ 熟悉度 = softmin(距离)    │
           │ "这像X" = nearest key    │
           │ 方向 = top-5 动作加权    │
           └─────────────────────────┘

Prototypical Buffer (并行):
- 32-128 原型, 在线 EMA 更新
- 每个原型 = 类别的 centroid thought
- familiarity = 到最近原型的距离
```

### 关键设计

| 组件 | 细节 |
|------|------|
| DND write | 每步写 (thought, Q/N-step return), LRU 淘汰 |
| DND read | kd-tree top-5, 内核权重 w = 1/(||h-h_i||² + δ) |
| 原型更新 | EMA: c_k ← (1-α)·c_k + α·h, 当 h 距 c_k < θ |
| 熟悉度 | softmax(-d(c_k, h)) → p(novel) |
| 方向建议 | top-5 邻居的动作 × Q 值加权 |

### 实现估算

| 子任务 | 代码 |
|--------|------|
| DND 核心 (kd-tree, write/read, kernel) | ~150 行 |
| 原型缓冲区 (EMA, 距离, 更新) | ~80 行 |
| OnlineAgent 集成 (读取代 Conductor?) | ~100 行 |
| RND 对比 + 消融 | ~50 行 |
| **合计** | **~380 行** |

---

## 执行策略

### 顺序

| Phase | 名称 | 依赖 | 参数/代码 | 时间估计 |
|-------|------|------|-----------|----------|
| 1 | WorldModel V5 | 无 | ~266K / ~610 行 | 3-5 天 |
| 2 | DoCalculusEngine | 无 | 0 / ~550 行 | 2-3 天 |
| 3 | Intuition Module | 建议有 Phase 1 | ~100K / ~380 行 | 2-3 天 |

Phase 1 和 2 无依赖关系, 可并行。

### 验证指标

| Phase | 通过条件 |
|-------|---------|
| 1 | next_state 预测 cosine similaruty > 0.8 on held-out transitions |
| 1-CF | 反事实预测与实验结果的差 < 0.3 |
| 2 | d-separation 输出与 pgmpy 一致 |
| 2-ATE | confounder 发现能在合成数据中召回真实混淆变量 |
| 3 | 熟悉度与 RND 新颖度相关性 > 0.6 |
| 3-方向 | DND 建议的方向优于随机基线 |

### 风险评估

| 风险 | 缓解 |
|------|------|
| 250K LoopWM 参数量不足以预测 384-dim 状态 | 降维 384→128 作为预测目标, 或加 skip connection |
| 因果图太稀疏 (200 nodes, 边少) | 先用 ExpertimentPlanner 干预验证过的边, 人工构建 DAG |
| DND 4096 条不够 | KD-tree O(log N) 可扩展到 10⁵+ 条 |
| CPU 训练太慢 | 数据量小 (600 transitions), 预期 5-10 分钟/轮 |

---

## 参考

| 论文/代码 | 时间 | 相关度 |
|----------|------|--------|
| LoopWM (2606.18208) | Jun 2026 | ★★★ 直接架构 |
| Tab-TRM (2601.07675) | Jan 2026 | ★★★ 表格版递归推理 |
| TRM 实现 (6.5k stars) | Oct 2025 | ★★★ 有代码可参考 |
| NextLat (2511.05963) | Nov 2025 | ★★★ next-latent 训练 |
| DreamerV3 (Hafner) | 2023-25 | ★★ RSSM 参考 |
| Causal-JEPA (2602.11389) | Feb 2026 | ★★ 反事实掩码 |
| NEC / DND (Pritzel) | 2017 | ★★★ DND 算法来源 |
| R2-Dreamer (2603.18202) | Mar 2026 | ★★ Decoder-free WM |
| Awesome-Loop-Models | 2026 | ★★ 循环架构综述 |
