# CaSolis_ 项目研究评审请求

## 项目概述

从零构建一个自主 AI agent，在 Docker 沙箱中运行，目标是实现「不依赖人的、自己会涌现创造能力并且每次不一样」的认知架构。

## 当前状态

### 已完成的工程

| 项目 | 详情 |
|------|------|
| 数据 | 1500 条干净训练数据（命令→输出对），BPE tokenizer (2022 tokens) |
| 最优模型 | Mamba 12.6M (d_model=1024)，checkpoint at `checkpoints/big-mamba/model_best.pt` |
| 最新模型 | MambaMTP 18.8M (4-head multi-token prediction + world model), checkpoint at `checkpoints/mtp/model-1.pt` |
| 最新架构 | SlotMind 6.9M (K-slot 工作记忆)，未训练 |

### 架构演化

1. **GRU-based (7.6M)** → 收敛慢，模式匹配
2. **Mamba 7.3M** → 收敛快 10x，但仍然是 next-token predictor
3. **Mamba 12.6M** → 训练稳定，可生成合法 Linux 命令
4. **MambaMTP 18.8M** → 4-head 多 token 预测 + 世界模型头，ep1 LM=1.538
5. **SlotMind 6.9M** → K=4 独立槽 + 跨槽注意力 + 路由门控，未经训练

### 已验证的发现

1. **Mamba 在 CPU 上可行** — 7.3M 模型在 4 线程 CPU 上约 2 min/epoch
2. **干净数据 > 自生数据** — 用脚本生成多样化数据比模型自生数据好得多
3. **系统发现 > 模型猜命令** — compgen -c 比 self-cmd 有效
4. **世界模型头可工作** — W loss 从 0.29 → 0.05，模型在学习预测输出嵌入
5. **warm-start 有效** — MTP 用已训 checkpoint 初始化，ep1 即达到 LM=0.456 (H0)

### 请帮我调研和评判的问题

#### 1. 核心假设验证
- 「思考 = 概率 + 最优选择」— 这个哲学假设是否站得住脚？
- 小模型（<20M）是否可能产生超越模式匹配的行为？
- 当前 literature 中，最小的涌现推理的模型是多大？

#### 2. 我们的架构方向
- K-slot 工作记忆的思路是否正确？有没有类似的研究？
- SlotMind 中哪些部分合理，哪些是噱头？
- 创造力（重组 + 意外驱动 + 质量判断）的实现路径是否可行？

#### 3. 被我们放弃或忽略的方向
- We tried: GRU baseline, Mamba, MTP, world models, curiosity
- We skip: Mixture of Experts, JEPA, distillation from large models, RLHF
- 我们是否错过了重要的东西？

#### 4. 批评与建议
- 当前 Roadmap 最大的问题在哪里？
- 分配太多时间在什么上？太少在什么上？
- 如果只做一件事来提升，应该是什么？
- 是否应该放弃纯 CPU/小模型路线？

### 背景信息（无上下文提供给 agent，以下为纯参考）

**核心成员**: 1 人，梦想做 agent OS，但缺乏系统 ML 知识。
**硬件**: CPU only (Intel Core Ultra 9 185H, 16C/22T, 105°C 热限), 30GB RAM, Docker 可用。
**时间**: 断断续续做了约 2-3 周。
**项目路径**: `/home/chillizu/Projects/CaSolis_/`
**已存档前期工作**: `github.com/Chillizu/Trahexa` (Phase 1, Qwen/DeepSeek pipeline)
