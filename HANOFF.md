# 🎯 Folunar — Handoff 文档

> 生成于: 2026-06-20
> 项目: 自主进化 Linux Agent, CPU 训练, Docker 隔离
> 哲学: 小模型是指挥家, 表达"直觉与思绪"; 大模型/执行层是保姆

---

## 一、当前架构（一句话版）

```
OnlineAgent.step()
  ├─ 50% 试 Conductor → 置信度>0.8? → Nanny翻译 → 模板引擎 → Docker
  └─ 50% 旧分类器(11类, 91%) → 模板引擎 → Docker
```

## 二、现状检查清单

### ✅ 核心系统 (稳定工作)

| 组件 | 文件 | 说明 |
|------|------|------|
| 旧分类器 | `agent/online_agent.py::IntentClassifier` | 86% 验证, 11类, 4809训练样本 |
| 模板引擎 | `benchmark/template_engine.py` | 11意图模板 + CUSTOM |
| 分层选择器 | `agent/command_selector_v2.py` | 38 cluster, UCB+软max+饱足 |
| 命令矿工 | `agent/command_miner.py` | 从ls /usr/bin发现新命令, 黑名单过滤 |
| RND好奇心 | `agent/rnd.py` | 新颖度检测, running avg误差 |
| 经验缓冲区 | `agent/experience.py` | 经验回放, 分层采样 |
| Docker沙箱 | `agent/sandbox_executor.py` | Alpine, 非特权, 无网络, 只读根FS |

### ✅ 指挥家+保姆 (新, 已验证)

| 组件 | 文件 | 指标 |
|------|------|------|
| ConductorHead | `agent/conductor.py` | MiniLM→MLP 384→128→64→[16+11] |
| 训练 | `scripts/train_conductor.py` | CE+对比损失, 87% val, 73% test |
| Nanny | `agent/nanny.py` | thought向量→logits→intent |
| A/B切换 | `agent/online_agent.py` | 50%尝试, 0.8阈值, 100步验证通过 |
| 想法向量 | — | 16维tanh, dim_std=0.63, 无坍缩 |

### ❌ 废弃/存档 (不要用)

| 组件 | 原因 |
|------|------|
| ModelCluster V3 (46M) | 0/30命令生成, 废弃于PLAN_VFINAL |
| BigMamba (19M) | 生成质量不行 |
| 一切mamba_model/mtp/slot_mind | 从零生成文本的路走不通 |
| Qwen3.5/DeepSeek R1/Gemma4 (Ollama) | 结构化参数推理全失败, 仅Kimi设计用 |
| 所有checkpoints/下的旧存档 | 历史痕迹, 删了不影响 |

## 三、活跃文件导览

### agent/ (核心模块)

| 文件 | 功能 | 使用方式 |
|------|------|----------|
| `online_agent.py` | 主循环, A/B切换 | `agent = OnlineAgent(); agent.run(100)` |
| `conductor.py` | 指挥家: 想法向量生成 | `c = Conductor(); c.forward(state_text)` |
| `nanny.py` | 保姆: thought→intent→执行 | `n = Nanny(); n.execute(state_text)` |
| `sandbox_executor.py` | Docker沙箱 | `s = SandboxExecutor(); s.execute("ls")` |
| `command_miner.py` | 命令发现 | `m = CommandMiner(sandbox=s); m.mine(output)` |
| `command_selector_v2.py` | 分层选择 | 38集群, UCB+软max |
| `experience.py` | 经验缓冲区 | 缓存(state, intent, reward, ...) |
| `rnd.py` | 好奇心 | `rnd.compute_novelty(emb)` |
| `world_model.py` | 世界模型 | 输出嵌入预测, 好奇心信号 (WM误差太低, 边缘化) |

### benchmark/ (基础设施)

| 文件 | 功能 |
|------|------|
| `template_engine.py` | 意图→命令翻译, 支持Docker沙箱 |
| `param_extractor.py` | 参数提取, config/param_rules.json |

### scripts/ (训练/工具)

| 文件 | 功能 |
|------|------|
| `train_conductor.py` | 训练指挥家 (CE+对比损失) |
| `train_intent_v2.py` | 训练旧11类分类器 (存档) |
| `verify_cognitive_expansion.py` | 认知扩展验证 |

### checkpoints/ (活跃)

| 文件 | 作用 |
|------|------|
| `checkpoints/intent_classifier/best_head.pt` | **旧分类器** (91%, 生产) |
| `checkpoints/conductor/head.pt` | **指挥家** (87% val, 新) |
| `checkpoints/long_run_5k/` | 5000步长程实验存档 |

### data/ (训练数据)

| 文件 | 样本数 | 用途 |
|------|--------|------|
| `intent_train_v3.jsonl` | 4809 | 全部11类训练数据 (主数据) |
| `intent_benchformat.jsonl` | 827 | 基准测试格式 |
| `intent_discovered.jsonl` | 150 | IntentDiscoverer发现的3类 |
| `intent_multistep.jsonl` | 210 | 多步训练数据 |

## 四、模型指标一览

| 模型 | 参数量 | 准确率 | 活跃? |
|------|--------|--------|-------|
| 旧分类器 IntentClassifier | ~66K新 | 86-91% | ✅ 生产 |
| ConductorHead | ~67K新 | 87% val, 73% test | ✅ A/B |
| MiniLM (冻结) | ~90MB | — | ✅ 编码器 |
| WorldModelNet | ~0.5M | WM误差0.00057 | ❌ 太低 |
| RND | ~0.3M | 新颖度0.005-0.01 | ✅ 好奇心 |

## 五、已知未完成

### P0: 阈值优化
A/B阈值 0.8 太保守, 指挥家只有 19% 使用率。需要做 0.8→0.7→0.6 网格扫描。

执行方法:
```python
agent = OnlineAgent(conductor_gate=0.6)  # 改这个参数
agent.run(n_steps=100)
# 看 agent.ab_stats 对比
```

### P1: 多命令组合
保姆只能翻译成单条命令。需要让模板引擎支持命令序列:
```python
# 期望行为
nanny.execute("了解网络")
# → 执行 ["cat /etc/hosts", "ip addr", "ss -tlnp"]
# → 返回合并结果
```

### P2: Conductor在线训练
指挥家目前是静态的 (训练一次后固定)。需要:
1. 积累足够多指挥家路径的经验 (>500步)
2. 从经验缓冲区采样 (state, thought, reward) 三元组
3. 用reward加权在线微调

### P3: 保姆LLM升级
模板引擎能力有限。长远要用本地LLM (Gemma4) 翻译想法向量到命令序列。
前提: 指挥家使用率>60%, 模板无法覆盖>500例, Gemma4稳定性验证通过。

## 六、运行手册

### 启动闭环
```bash
cd /home/chillizu/Projects/Folunar_
source .venv/bin/activate
PYTHONPATH=. python3 -c "
from agent.online_agent import OnlineAgent
agent = OnlineAgent(conductor_gate=0.8)
agent.run(n_steps=100)
"
```

### 重新训练指挥家
```bash
PYTHONPATH=. python3 scripts/train_conductor.py
```

### 仅测试Docker沙箱
```bash
PYTHONPATH=. python3 -c "
from agent.sandbox_executor import SandboxExecutor
s = SandboxExecutor()
print(s.execute('echo hello && uname -a'))
s.close()
"
```

### 查看执行日志
```bash
tail -f exec_log.jsonl
```

## 七、关键设计决策 (别改)

| 决策 | 原因 |
|------|------|
| 指挥家输出16维tanh(非sigmoid) | 对比损失需要负值, 避免坍缩 |
| 保姆使用class_proj投影(非原型匹配) | 原型向量坍缩到INFO, 投影法更有判别力 |
| A/B门控(非直接替换) | 可回退, 无风险 |
| Docker非特权+只读+无网络 | 隔离最大化, 防弹窗/副作用 |
| CPU训练, 限制4线程 | 热限制105°C, torch.set_num_threads(4) |
| Kimi仅设计用(不在闭环里) | 用户要求, 且API不稳定 |

## 八、与Kimi对话记录

| Session | 内容 |
|---------|------|
| kimi-conductor-v2 | 哲学转变: 指挥家+保姆, 想法语言设计 |
| — (无session) | 架构验证: 16维, 对比损失, 知识状态defer |
| — (无session) | 最新: 优先级A(阈值)>E(多命令)>B(在线训练) |

建议: 用 `pi --provider kimi-coding --continue` 继续对话, 或用 `--name kimi-conductor-plan` 起新session。

## 九、文件管理建议

### 可安全删除 (历史存档)
- `checkpoints/stream-v1/` 到 `stream-v4/` (全是Mamba旧实验, 720个文件)
- `checkpoints/forever-v1/` 到 `forever-v3/` (GRU旧模型)
- `checkpoints/word-v1/` `word-v2/` `native-*` (更早的实验)
- `arch/mamba_model.py` `arch/mamba_mtp.py` `arch/slot_mind.py` (从零生成方向已废弃)
- `scripts/train_mamba.py` `train_mtp.py` `train_*.py` 中的旧训练脚本 (可归档到archive/)

### 必须保留
- `agent/` 下所有.py
- `benchmark/` 下所有.py
- `checkpoints/intent_classifier/` 和 `checkpoints/conductor/`
- `data/intent_train_v3.jsonl`
- `config/param_rules.json`

## 十、一句话给下一任

> 别试从零生成命令了, 46M的ModelCluster证明了这条路走不通。
> 指挥家只输出16维想法向量, 保姆翻译成命令。
> 当前A/B切换已验证, 但阈值0.8太保守, 先降到0.6试试。
> 用户的核心哲学: "你对我说话, 我帮你做事" — 指挥家不需要会写命令。
