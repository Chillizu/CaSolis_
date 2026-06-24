# Folunar — Handoff 文档

> 生成于: 2026-06-24
> 项目: 自主进化 Linux Agent, CPU 训练, Docker 隔离
> 哲学: 小模型是指挥家, 表达"直觉与思绪"; 大模型/执行层是保姆
> 当前阶段: P9.7 完成, 16意图 100% 稳定

---

## 一、一句话版

```
OnlineAgent.step()
  ├─ 10-70% Conductor → thought向量 → Nanny翻译 → 意图+参数 → TemplateEngine → Docker沙箱
  ├─ 10-90% 分类器 → 意图 → ParameterExtractor → TemplateEngine → Docker沙箱 (A/B自适应)
  ├─ 30% 链式目标 (follow-up) → Workbench
  ├─ 10% 自生成目标 → Workbench.build_write/generate_content
  ├─ 5% 探针 → Workbench._build_dynamic_probes
  └─ 80%(概率) 想象力 → WorldModel → 直接选意图

  每一步:
    → SandboxExecutor.execute → ExecResult
    → Workbench.extract_facts (含通用提取 _extract_generic)
    → RND.compute_novelty (新颖度+兴趣重置)
    → ExperienceBuffer.store
    → 每10步在线训练 (分类器+WM+Conductor)
    → 每20步 RND.interest_reset
    → 每50步 RND.soft_reset + _scan_new_commands
```

---

## 二、P0-P9 完整历程

| 阶段 | 改动 | 成功率 |
|------|------|--------|
| **P0-P5** | 核心闭环: 状态编码→分类器→模板引擎→Docker | 30-50% |
| **P5.0-P5.6** | Conductor, 多样性调度, 逆想象, 元学习 | 50-65% |
| **P6.0-P6.4** | 探针闭环, 事实去噪, 脚本模板, 新颖度维护, 新意图自动连接 | 65-70% |
| **P7.0-P7.2** | CUSTOM维度统一, 统计修复, 核心流修复, 300步稳定 | 70-75% |
| **P8.0-P8.5d** | 错误恢复(34%→87%), 参数推断, 多样性强化(13/13), 训练增强(UCB+LR调度), 模板外化, 动态探针, 想象力修复, 自由扩展 | 65-93% |
| **P9.0: 环境** | `folunar-sandbox:latest` (预装 python3/curl/jq), `--tmpfs /workspace:exec`, `--read-only` 移除, DetailedLogger | 53→64% |
| **P9.1: 热身** | 安全写入 (base64+python3), 脚本执行, CUSTOM 创作优先 | 64→89% |
| **P9.2: WRITE/APPEND** | INTENTS 14→16, 创造奖励 (0.3+字节奖励), 修复 append 模式 | 89→93% |
| **P9.3: 内容生成器** | `build_write_content()`: JSON/Markdown/Shell 三种风格, SEARCH/USB_DEVICES 修复 | 93→99% |
| **P9.4: 100% + 事实突破** | 消灭7%失败 (DISK_USAGE/INFO/CUSTOM/引号/类型), `_extract_generic` 通用提取, 事实奖励, 事实18→40 | 99→100% |
| **P9.5: 内容深度** | `_build_fact_analysis()` 自然语言总结, 脚本从 `echo OK` 升级为真实验证 (对比+WARN+退出码) | 100% (3连) |
| **P9.6: GENERATE + 推理** | INTENTS 16→17, `build_generate_content()` (profile/experiment/discovery_log), `_build_cross_analysis()` 跨事实推理, `_build_change_report()` 变化检测, `_track_fact_history()` 历史追踪 | 100% 16种 |
| **P9.7: 想象力+好奇心** | 想象力 1→7次/100步 (+300%), RND `interest_reset()` 每20步, 新颖度 0.003→0.078 (+2500%), 好奇心不足时强制 CUSTOM 探索 | 100% 16种, 奖励1.65 |

---

## 三、当前架构 (P9.7)

### 核心循环

```
  state_text (环境状态)
       │
       ├─ 链式目标 (follow-up, 从 Workbench 事实缺口推导)
       ├─ 自生成目标 (Workbench.generate_self_goal, 每6步)
       ├─ 探针 (Workbench.get_curiosity_probe, 概率+步数门控)
       ├─ 想象力 (WorldModel 评分, 80%概率, 含好奇心CUSTOM降级)
       ├─ Conductor (thought向量 → Nanny翻译, A/B自适应)
       └─ 分类器 (IntentClassifier, 17类, UCB偏差)
            │
            ▼
      参数提取 (ParameterExtractor + _rescue_params)
            │
            ▼
      模板引擎 (TemplateEngine → build_args → shell命令)
            │
            ▼
      Docker沙箱 (folunar-sandbox:latest, --network none)
            │
            ▼
      执行结果 (exit_code + stdout/stderr)
            │
            ├─ 错误恢复 (ErrorRecovery, 如果失败)
            ├─ 工作栏提取 (Workbench.extract_facts, 含通用+专用)
            ├─ 计算奖励 (exit_code + 新颖度 + 多样性 + 事实发现)
            ├─ RND更新 + 兴趣重置 (每20步)
            ├─ ExperienceBuffer 存储
            ├─ 在线训练 (每10步: 分类器 + WM + Conductor)
            └─ 详细日志 (DetailedLogger → run_logs/*.jsonl)
```

### 模块文件一览

| 文件 | 功能 | 关键方法/属性 |
|------|------|--------------|
| `agent/online_agent.py` | 主循环 | `step()`, `_select_intent()`, `_imagine_intent()`, `_rescue_params()`, `_compute_reward()`, `_validate_custom()`, `_scan_new_commands()` |
| `agent/workbench.py` | 工作栏+事实提取+内容生成 | `extract_facts()`, `_extract_generic()`, `_build_fact_analysis()`, `_build_cross_analysis()`, `_build_change_report()`, `build_write_content()`, `build_generate_content()`, `generate_self_goal()`, `_build_dynamic_probes()` |
| `agent/conductor.py` | 指挥家 (thought 向量) | `N_INTENTS=17`, `ConductorHead: 384→128→64→[16+11]` |
| `agent/nanny.py` | 保姆 (thought→intent) | `INTENTS=17`, `ConductorHead` |
| `agent/world_model.py` | 世界模型 | `expand_intents()`, `rollout()`, `update()`, 自动维度扩展 |
| `agent/rnd.py` | RND 好奇心 | `compute_novelty()`, `soft_reset()`, `interest_reset()` |
| `agent/error_recovery.py` | 错误恢复 | `ErrorClassifier`, `COMMAND_FALLBACKS` (40+) |
| `agent/sandbox_executor.py` | Docker 沙箱 | `folunar-sandbox:latest`, `--tmpfs /workspace:exec` |
| `agent/state_encoder.py` | 状态编码 | state_text 构建 |
| `agent/detailed_logger.py` | 超级日志 | `run_logs/run_*.jsonl`, 12+字段/步 |
| `agent/command_selector_v2.py` | 分层命令选择 | UCB+软max+饱足 |
| `agent/command_clusterer.py` | 命令聚类 | 40+ cluster |
| `agent/command_miner.py` | 命令挖掘 | 从 compgen -c 发现新命令 |
| `agent/experience.py` | 经验回放 | ExperienceBuffer |
| `benchmark/template_engine.py` | 命令模板 | `build_args()`, `execute_multi()`, 安全写入 (base64+python3) |
| `benchmark/param_extractor.py` | 参数提取 | `_infer_from_facts()` |
| `config/command_registry.json` | 命令注册 | 17意图模板 + 19 info_cmds + 37+8 custom_commands + 13 multi_commands |
| `config/param_rules.json` | 参数规则 | 17意图规则 |
| `config/workbench_rules.json` | 工作栏规则 | 动态探针, follow-up 链 |

### INTENTS (17 类)

```
[0] READ        [5] COUNT       [10] DISK_USAGE    [15] APPEND
[1] LIST        [6] EXPLORE     [11] LS_TMP       [16] GENERATE
[2] SEARCH      [7] HELP        [12] ARCH_INFO
[3] INFO        [8] READ_ETC    [13] CUSTOM
[4] INSPECT     [9] USB_DEVICES [14] WRITE
```

N_INTENTS=17, 有效=16 (HELP 无效, 永不触发)

---

## 四、当前指标 (P9.7 最终)

### 沙箱内真实产物 (最后 run, 100步)

| 类型 | 文件数 | 示例 | 大小 |
|------|--------|------|------|
| JSON 事实 | 6 | `facts_97.json` (含 _meta / _summary / _inference / _changes / facts) | ~1.3KB |
| Markdown 报告 | 3 | `report.md` (含 System Analysis / Inference / Raw Facts), `profile_70.md` (含 Overview/Inference/Changes/Fact Inventory) | ~2.6KB |
| Shell 脚本 | 6 | `check_87.sh` (真实验证, 对比+WARN+退出码), `experiment_81.sh` (假设测试) | ~1.2KB |
| 发现日志 | 2 | `discovery_97.json` (JSON, 含分类组织事实) | ~4.7KB |
| **总计** | **15** | — | **28KB** |

### 所有意图 100% 成功率

| 意图 | 成功率 | 意图 | 成功率 |
|------|--------|------|--------|
| READ | 100% | ARCH_INFO | 100% |
| LIST | 100% | CUSTOM | 100% |
| SEARCH | 100% | WRITE | 100% |
| INFO | 100% | APPEND | 100% |
| INSPECT | 100% | GENERATE | 100% |
| COUNT | 100% | USB_DEVICES | 100% |
| EXPLORE | 100% | DISK_USAGE | 100% |
| READ_ETC | 100% | LS_TMP | 100% |

### 累计 (所有 run_logs)

| 指标 | 值 |
|------|-----|
| 总运行步数 | 2,311 |
| WRITE+APPEND 步数 | 191 |
| GENERATE 步数 | 17 |
| 产出总字节 | ~28KB (当前沙箱) |

### 推理输出示例

```
## Inference (来自 _build_cross_analysis)
* Environment: Arch Linux host running Debian container (kernel mismatch typical of containers)
* Storage: persistent volume (102G) larger than root (81G) → external volume mount detected
* Network: hostname 'c9c341a36f9f' is a Docker container ID → no DNS, likely --network none
* Security: single root user — standard container lockdown
* Discovery: 40 facts across 5 categories — good coverage
```

### 模型指标

| 模型 | 参数量 | 准确率 | 状态 |
|------|--------|--------|------|
| 分类器 IntentClassifier | ~66K | ~90% | ✅ 生产 |
| ConductorHead | ~67K | ~87% val | ✅ A/B |
| WorldModelNet | ~0.5M | — | ✅ 动态扩展 |
| MiniLM (冻结) | ~90MB | — | ✅ 编码器 |
| RND | ~0.3M | 新颖度 ~0.08 | ✅ 好奇心 |

---

## 五、系统能创造什么

> 真实盘点: 2311 步自主运行后, 系统能/create/什么?

| 创造力层级 | 能? | 具体产出 |
|-----------|-----|----------|
| **一句话** | Yes | `_build_fact_analysis()`: "System: Debian GNU/Linux 12 (bookworm), kernel 7.0.12, 22 cores" |
| **推理句** | Yes | `_build_cross_analysis()`: "persistent volume (102G) larger than root (81G) → external volume mount detected" |
| **结构化数据** | Yes | JSON: `{"_meta":{...}, "_summary":"...", "_inference":"...", "facts":{...}}` |
| **文档** | Yes (≤5KB) | Markdown 报告/系统画像, 含多节+推理 |
| **Shell 脚本** | Yes (可执行) | 检测脚本 (真实验证), 实验脚本 (假设测试) |
| **Python 脚本** | No | 沙箱有 python3 但系统不主动写 .py 文件 |
| **长文档** (>10KB) | No | 单文件最大 4.7KB |
| **持久知识** | No | 每次重启沙箱归零 |
| **新颖概念** | No | 都是已知事实的格式化重组 |
| **图像/音频** | No | 无工具 |

### 根本限制

```
1. 沙箱 --network none  → 不能装 pip/npm, 不能访问外部知识
2. 事实 18→40          → 事实数量天花板 (沙箱内信息有限)
3. 模板驱动             → 内容格式固定, 不是自由生成
4. 无持久化             → 每次重启归零
5. 固定意图空间(17)     → 新行为需要手动扩展
6. 无长期记忆           → 不能跨 session 积累知识
7. 训练loss发散         → 在线学习不稳定
8. RND 新颖度周期性归零  → 好奇心不能持续
```

---

## 六、已知未完成

### 训练 Loss 发散
- UCB 权重给罕见意图 10x, 噪声样本权重过大
- 需要: 约束 UCB 权重 ≤ 3.0, 或改 importance-weighted sampling
- 当前 loss: 10-14 之间波动, 没真正收敛

### 世界模型想象力稀疏
- `_imagine_intent()` 已升至 7次/100步 (仍不够)
- WM 的 value/agreement/dist 预测对意图选择的影响有限
- 需要: 更丰富的 WM 训练信号, 或更大容量的 predictor

### CUSTOM 收敛
- CUSTOM 在 hostname/cmp/exa 等少数命令上循环
- 新命令扫描 (`_scan_new_commands`) 发现 187 个命令但不被使用
- 需要: 好奇心驱动的命令选择, 新命令奖励

### 事实天花板
- 18→40 后增长停滞 (沙箱内只有这么多信息)
- 需要: 网络访问或更智能的探针策略

### 无持久记忆
- 每次 `docker rm -f folunar-sandbox` 归零
- 经验缓冲区/工作栏/元学习器 只 checkpoint 到磁盘, 不跨 session
- 需要: 沙箱外独立的知识库

---

## 七、架构方向: 层级 + 动态网格

> 以下摘自 P9.7 完成时的方向讨论:

### 当前瓶颈的本质

```
现在: 扁平意图(17) × 扁平事实(40) × 模板生成  →  容易碰壁
目标: 层级意图 × 动态事实图 × 自由生成        →  自增长
```

### "动态网格 + 层级架构" 具体是什么

层级架构 —

```
顶层: 元认知 (MODE)
  ├─ 探索模式: 发现未知事实
  ├─ 创作模式: 从已知事实生成新内容
  └─ 学习模式: 验证假设、修正模型

中层: 目标 (GOAL)
  ├─ 事实缺口驱动: "有 os_name 但没有 os_version → 去读 os-release"
  ├─ 好奇心驱动: "RND 对 /proc/meminfo 预测误差大 → 再读一次"
  └─ 创作驱动: "事实够了 → 生成报告"

底层: 动作 (ACTION) = 当前 17 意图 + 动态新意图
  ├─ 17 个已有意图 (成熟稳定)
  └─ 运行时发现的新意图 (自动接入)
```

动态事实网格 —

```
不是 key-value 字典, 而是:
  每个事实是一个节点
  关系是边 (os_name "runs_on" kernel, disk_root "smaller_than" disk_persistent)
  新事实自动连边 → 产生新缺口 → 驱动新探索
  图是开边的 → 没有 40 条上限
```

### 要实现的组件

| 组件 | 做什么 | 替代什么 |
|------|--------|----------|
| 元认知选择器 | 选模式 (探索/创作/学习) | A/B gate |
| 层级意图分解 | 目标拆解为子任务 | `_select_intent()` |
| 动态事实图 | 图结构知识库 | `Workbench.facts` dict |
| 增长型 WM | 核+叶, 新行为加新叶 | `WorldModel` (固定维度) |
| 情景记忆 | 存储意外转移 | RND (唯一种子) |

### 建议实施顺序

1. **事实图** — 替换 Workbench, 最基础 (2-3天)
2. **层级选择** — 把当前 A/B gate 换成三层 (1-2天)
3. **增长型 WM** — 让世界模型能动态扩展 (3-4天)
4. **情景记忆** — 最后加上 (1天)

---

## 八、运行手册

### 启动闭环 (100步)
```bash
cd /home/chillizu/Projects/Folunar_
source .venv/bin/activate
PYTHONPATH=. python3 scripts/long_run.py --steps 100 --gate 0.5 --name run_name
```

### 启动更长时间
```bash
PYTHONPATH=. python3 scripts/long_run.py --steps 300 --gate 0.5 --name long_run
```

### 查看运行日志
```bash
# 最新 JSONL 日志
cat run_logs/run_$(ls -t run_logs/ | head -1 | sed 's/run_//' | sed 's/.jsonl//').jsonl

# 意图统计
python3 -c "
import json
with open('run_logs/run_*.jsonl') as f:
    steps = [json.loads(l) for l in f if 'intent' in l]
from collections import Counter
c = Counter(s['intent'] for s in steps)
for i, cnt in sorted(c.items()):
    print(f'{i}: {cnt}')
"
```

### 查看沙箱内产物
```bash
PYTHONPATH=. python3 -c "
from agent.sandbox_executor import SandboxExecutor
s = SandboxExecutor()
print(s.execute('ls -la /tmp/').stdout)
"
```

### 清除沙箱 (开始新 run)
```bash
docker rm -f folunar-sandbox
```

---

## 九、关键设计决策 (别改)

| 决策 | 原因 |
|------|------|
| Conductor 输出 16 维 tanh (非 sigmoid) | 对比损失需要负值, 避免坍缩 |
| Nanny 使用 class_proj 投影 (非原型匹配) | 原型向量坍缩到 INFO, 投影法更有判别力 |
| A/B 门控 (非直接替换) | 可回退, 无风险 |
| Docker 非特权+无网络 | 隔离最大化, 防副作用 |
| CPU 训练, 限制 4 线程 | 热限制 105°C, `torch.set_num_threads(4)` |
| Kimi 仅设计用 (不在闭环里) | 用户要求, 且 API 不稳定 |
| INTENTS 从 13→16→17 (非动态无限) | 维度扩展需要模型适配, 不能无限 |
| 安全写入 base64+python3 (非 shell 模板) | 防 shell 注入 |
| `_extract_generic` 始终运行 (非仅 fallback) | 所有输出都可能含信息 |
| RND `interest_reset()` 每 20 步 | 防新颖度永久归零 |
| 想象力 80% 概率每步触发 (非步数门控) | 最大化想象力使用率 |

---

## 十、文件管理建议

### 核心活跃文件
- `agent/` 下所有 .py (13 个核心模块)
- `benchmark/` 下所有 .py (2 个基础设施)
- `config/` 下所有 .json (规则/模板)
- `scripts/long_run.py` (启动脚本)
- `docker/Dockerfile.sandbox` (沙箱镜像)

### 运行日志 (可清理)
- `run_logs/run_*.jsonl` — 可以定期压缩到 archive/
- `exec_log.jsonl` — 单步日志 (每次 run 覆盖)
- `checkpoints/p9_*/` — 各版本 checkpoint (可清理旧版, 保留最新 2-3 个)

### checkpoint 保留策略
- 保留: `checkpoints/world_model/latest.pt`, `checkpoints/conductor/online_aligned.pt`, `checkpoints/intent_classifier/best_head.pt`
- 可清理: `checkpoints/p9_*/` 中的旧实验 (每版 ~10MB, 11 版 ≈ 110MB)

---

## 十一、一句话给下一任

> 系统已经从"频繁报错的原型"进化成"16意图100%稳定的自治 Agent"。
> 它能独立地侦察、推理、创作、验证 — 在 28KB 的产出中包含了 15 个真实可用的文件。
> 下一步不是继续修bug, 而是架构升级: 从扁平走向层级, 从静态走向动态, 从有限走向开放。
> 核心哲学不变: "你对我说话, 我帮你做事" — 指挥家不需要会写命令。
