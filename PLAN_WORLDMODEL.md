# WorldModel 升级计划 — V5 → V6

## 当前架构 (V5)

```
obs (state_emb 128) ───┐
action (418 one-hot) ───┼─→ [GRU] → h_t (256) → [MLP] → pred_next_state (128)
                                                  → [MLP] → pred_reward (1)
```

- 纯确定性 GRU, 364K params
- Loss: MSE(state) + MSE(reward)
- 训练数据: 500 transitions
- 问题: 没有不确定性估计, 不能知道"这个动作带来了意料之外的结果"

---

## 升级路线 (3个Phase, 可独立实施)

### Phase 1: 惊奇度信号 (今天能做的, 零架构改动)

V5 已经预测 state_emb, 已经算了 MSE loss。但训练完成后没有记录"这次预测不准了多少"。

**改动:**
```
在 step() 里, 每次执行一个 action 后:
  1. WorldModel 做预测: pred_next = WM(state_before, action)
  2. 等实际结果: actual_next = state_after
  3. 算 surprise = MSE(pred_next, actual_next)
  4. 存到 self._recent_surprise 列表 (last 50 steps)
```

**GoalGenerator 用 surprise:**
```
surprise 消费逻辑 (替换当前标签计数):
  每步检查 self._recent_surprise
  → 平均 > 0.05: "最近在发现新东西, 继续探索当前方向"
  → 平均 < 0.01: "最近都是重复, 随机跳转方向"
  → 找出 surprise 最高的 action 类别 → 选那个类别做创作
```

代价: 0 行新模型代码。只是把训练时的 loss 拿出来用。

### Phase 2: 随机隐变量 (Stochastic RSSM Lite)

加一个小随机分支, 让模型学会"对不确定的事输出高方差":

```
state_emb ─→ [GRU] → h_t (256) ──→ [MLP] → categorical_logits (16×16=256)
                                      action ─┘
                                      
h_t + z_t (16) ─→ [MLP] → pred_state (128)
h_t + z_t (16) ─→ [MLP] → pred_reward (1)
```

新组件:
- StochasticPredictor: MLP(256+418 → 256 hidden → 16×16 logits)
- Categorical sampling (straight-through)
- KL loss: KL(posterior || prior) with free bits (0.5 nats)
- 参数量: ~40K 新增 (total ~400K)

训练后:
- 高 KL = 模型不知道会发生什么 = 强惊奇信号
- 低 KL = 模型很确定 = 常规操作
- 这个信号比 MSE 更稳定 (不受数值范围影响)

### Phase 3: 动作效果预测 (Observation Prediction)

当前 V5 只预测 state_emb, 不预测"执行后哪些事实变了"。加一个 head:

```
h_t + z_t → [FactHead] → binary_pred (418)  # 每个命令是否更新了事实
h_t + z_t → [CategoryHead] → cats_pred (18)  # 哪些类别新增了事实
```

这比重建原始 obs 更适合文本领域:
- 预测"这个命令会更新事实吗" (binary 分类)
- 预测"哪个类别会有新事实" (multi-label)
- Loss: BCE(fact_pred, actual_new_facts)

**完整 V6 架构:**
```
state_emb ──→ [Encoder MLP] → embed (128)
action ─────┘
                ↓
          [GRU] → h_t (256) → [Stochastic MLP] → categorical z_t (16)
                ↓                                    
          h_t + z_t → [Reward Head] → pred_reward    
          h_t + z_t → [State Head] → pred_next_state  
          h_t + z_t → [Fact Head] → pred_new_facts    
          h_t + z_t → [Continue Head] → pred_done     
                ↓
          KL(posterior(z) || prior(z)) → surprise
          MSE(pred_state, actual_state) → reconstruction_loss
          BCE(pred_facts, actual_facts) → fact_loss
```

---

## 训练策略

当前 500 transition 太小。Phase 1 不需要重训 (直接用现有 WM 算 MSE), Phase 2/3 需要积累更多数据。

```
Phase 1: surprise = MSE(pred, actual)  # 现有模型直接算, 不需重训
Phase 2: 收集 ~2000 transitions → 训练 RSSM Lite
Phase 3: 收集 ~5000 transitions → 训练完整 V6
```

每步 transition = `(state_before, action, state_after, reward, new_facts)`

---

## 惊奇度→创作决策 集成

```
每20步:
  avg_surprise = mean(recent_surprise[-10:])
  max_action = argmax(recent_surprise)  # 哪个动作最令人意外
  
  if avg_surprise > 0.05:
    风格倾向 = 探索当前方向 (code/analysis)
  elif avg_surprise < 0.01:
    风格倾向 = 随机跳转 (report/creative)
  else:
    风格倾向 = 正常加权随机
  
  # max_action 可以映射到创作领域
  # 比如 max_action 是 /proc/net 相关 → 网络安全分析
```

---

## 实现优先级

|Phase|难度|模型改动|数据需求|收益|
|---|---|---|---|---|
|1|低|0|0|立即获得惊奇度信号|
|2|中|~40K 新参数|~2000 步|不确定性估计|
|3|高|~100K 新参数|~5000 步|事实级别预测|

推荐: 先做 Phase 1 (零成本), 同时积累数据, 再做 Phase 2。
