# Folunar_ 层级架构升级评审 (P9.7 → v10)

> 评审日期: 2026-06-24  
> 当前状态: P9.7 — 17意图100%稳定，扁平架构  
> 目标状态: 层级架构 (MODE→GOAL→ACTION) + 动态事实图 + 增长型世界模型

---

## 一、当前架构诊断

### 1.1 核心问题

| 问题 | 根因 | 影响 |
|------|------|------|
| 训练Loss发散 (UCB 10x噪声权重) | `HierarchicalSelector` 中 UCB bonus 与 reward 量纲不匹配；在线训练时 curiosity + policy gradient + CE + contrastive 多损失简单相加 | 梯度爆炸、WM和Conductor学习不稳定 |
| 想象力稀疏 (7次/100步) | `_imagine_intent()` 依赖 WM 的 value/agreement/distance 信号，但 WM 训练数据少；80%概率门控 + 好奇心注入分流 | WM 训练信号弱，rollout 不准 |
| CUSTOM收敛 (187命令不被用) | `HierarchicalSelector` 的 cluster/command 两层里，CUSTOM 命令选择被硬编码参数池束缚；`_validate_custom` fallback 频繁转成 echo | 发现的新命令没有反馈闭环 |
| 事实天花板 (40条扁平dict) | `Workbench.facts` 是 `dict[str, dict]`，无关系、无类别拓扑、无优先级；LRU淘汰破坏发现链 | 无法表达 "有os_name缺os_version" 这类缺口 |
| 无持久记忆 | Docker `--network none` + 事实只存内存/一次性JSON；重启后 RND、WM、workbench 经验归零 | 无法累积长期世界模型 |

### 1.2 代码结构缺陷

- `online_agent.py` (2142行) 的 `step()` 承担: 状态编码、目标驱动、A/B选择、参数拯救、WM验证、执行、训练、恢复，违反单一职责。
- `Workbench` 与 `StateEncoder` 互相引用，提取规则和链式目标硬编码在 Python 中，自改进只通过 `add_user_rule` 做简单字符串匹配。
- `WorldModel` 的意图用 one-hot 拼接在第一层，扩展意图需改输入层维度；多任务头共享同一 `hidden_dim=128`，小模型容量不足。
- `ConductorHead` 的 `class_proj` 和 `thought_head` 共享 hidden，蒸馏与在线目标冲突。

---

## 二、目标架构设计 (MODE→GOAL→ACTION)

```
顶层: 元认知 (MetaCognitive Mode Selector)
  ├─ EXPLORE  — 发现未知事实、降低不确定性
  ├─ CREATE   — 从已知事实生成内容/脚本/报告
  └─ LEARN    — 验证假设、修正模型、训练新意图

中层: 目标生成器 (Goal Generator)
  ├─ 事实缺口目标   (FactGraph.gaps())
  ├─ 好奇心目标     (RND high-error + WM disagreement)
  ├─ 创作目标       (事实充足 → WRITE/GENERATE)
  └─ 链式/验证目标  (follow-up chains from graph)

底层: 动作执行器 (Action Executor)
  ├─ 17 个已知意图
  ├─ 动态新意图 (IntentDiscoverer → 注册为 leaf)
  └─ CUSTOM + HierarchicalSelector
```

### 2.1 关键原则

1. **分层决策**: MODE 选高层策略，GOAL 把策略转具体任务，ACTION 只负责执行。
2. **事实驱动**: 所有 GOAL 来自 `FactGraph` 的缺口或 RND/WM 的不确定性。
3. **增长型模型**: WM 和 Classifier/Conductor 用 "frozen core + expandable leaves"，新意图只加叶节点。
4. **渐进迁移**: 先做事实图，再做 MODE，最后做增长型 WM；每一步都保留旧路径做 A/B。

---

## 三、事实图设计方案 (Q1)

### 3.1 选型: 纯 dict + 邻接表 (推荐)

不建议 NetworkX (引入大依赖、序列化复杂) 也不建议 PyG (过度设计、沙箱离线)。

**理由**:
- 当前事实数量小 (40→目标500)，纯 dict 足够 O(1) 查询。
- 需要持久化到 JSON，纯 dict 最友好。
- 节点和边可以携带元数据，便于与现有 `Workbench` 集成。
- 无外部依赖，符合 Docker 离线沙箱。

```python
# agent/fact_graph.py
class FactGraph:
    def __init__(self, max_nodes: int = 512):
        self.nodes: dict[str, dict] = {}  # key -> {value, category, confidence, step, source_cmd, ...}
        self.edges: dict[str, list[dict]] = {}  # key -> [{to, rel, weight, step}]
        self._gaps: list[dict] = []  # 缓存缺口
        self._schemas: dict[str, list[str]] = self._load_schemas()  # 类别schema

    # 节点操作
    def add(self, key, value, category="general", confidence=1.0, source=None, step=0)
    def get(self, key) -> dict | None
    def update(self, key, value, step)  # 置信度递增 + track history
    def remove_oldest(self, n=1)  # 按 category priority + step

    # 边操作
    def link(self, from_key, to_key, rel="requires", weight=1.0, step=0)
    def neighbors(self, key) -> list[dict]
    def related(self, key, rel=None) -> list[str]

    # 缺口发现
    def gaps(self) -> list[GoalCandidate]
    def suggest_next(self) -> GoalCandidate | None

    # 与状态文本集成
    def summary(self, max_nodes=20) -> str
    def to_state_text(self) -> str
    def save/load(path)
```

### 3.2 边的种类

| 关系 | 含义 | 例子 |
|------|------|------|
| `requires` | A 的存在暗示应补充 B | `os_name` → `os_version_id` |
| `verifies` | B 验证 A 的可靠性 | `hostname` → `hostname_cmd` |
| `extends` | B 是 A 的详细信息 | `cpu_cores` → `cpu_model` |
| `located_in` | 文件属于某目录 | `dir_etc` → `os-release` (通过文件列表推导) |
| `derived_from` | 由命令输出推导 | `mem_total` → `raw_free` |
| `conflicts_with` | 两个事实矛盾 (待验证) | `is_root=yes` vs `uid_info=uid=1000` |

### 3.3 类别 Schema

```python
SCHEMAS = {
    "system": ["os_name", "os_version_id", "kernel", "architecture", "hostname", "cpu_cores", "cpu_model", "mem_total", "swap_total"],
    "network": ["ip_addr", "mac_addr", "etchosts_hosts", "gateway"],
    "storage": ["disk_root", "disk_persistent", "disk_tmp"],
    "user": ["current_user", "uid_info", "gid_info", "is_root", "users"],
    "explore": ["dir_*"],
    "general": [],
}
```

`gaps()` 从 schema 出发，对每类检查缺失项；同时检查现有节点的 `requires` 边是否未满足。

### 3.4 与 workbench.py 集成

**最小破坏方案**: 保留 `Workbench` 类名和主要接口，内部把 `self.facts` 替换为 `self.graph = FactGraph()`。

需要改的地方:
- `Workbench.__init__`: `self.facts` 改为 `self.graph`。
- `_add_fact`: 调用 `self.graph.add(...)`，并自动根据 category/schema 建立边。
- `get_fact/get_facts_by_category/get_state_summary/get_current_discovery`: 委托给 `FactGraph`。
- `generate_self_goal/get_follow_up`: 从 `self.graph.gaps()` 读取，替换硬编码链。
- 增加 `Workbench.add_link(from_key, to_key, rel)` 供外部调用。
- 保持 `max_facts` 语义，但改为 `max_nodes`。

**向后兼容**: `Workbench.facts` 属性代理到 `self.graph.nodes`。

---

## 四、元认知选择器 (Q2)

### 4.1 设计: 规则 + 小分类器

纯规则足够且可解释；小分类器用于从长期统计中学习 MODE 切换。

```python
# agent/meta_selector.py
class MetaCognitiveSelector:
    MODES = ["EXPLORE", "CREATE", "LEARN"]

    def __init__(self, classifier_head_dim: int = 64):
        # 输入: 统计特征向量
        # [n_facts, fact_growth_rate, rnd_avg, wm_loss, recent_create_ratio, recent_success_rate, gaps_count]
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # EXPLORE/CREATE/LEARN
        )
        self.rule_override = True  # 前100步强制规则
        self.mode_history = []

    def select(self, stats: dict) -> str:
        # 规则层 (可覆盖)
        if stats["n_facts"] < 5:
            return "EXPLORE"
        if stats["fact_growth_rate"] < 0.05 and stats["gaps_count"] == 0 and stats["n_facts"] >= 8:
            return "CREATE"
        if stats["wm_loss"] > 1.0 or stats["rnd_avg"] < 0.01:
            return "LEARN"

        # 小分类器层
        vec = self._build_vector(stats)
        with torch.no_grad():
            return self.MODES[self.net(vec).argmax().item()]
```

### 4.2 MODE 与 GOAL 的映射

| MODE | 目标来源 | 典型 GOAL |
|------|----------|-----------|
| EXPLORE | `FactGraph.gaps()` + RND high-error nodes + schema 缺项 | "补充 os_version_id", "探索 /proc/meminfo" |
| CREATE | `FactGraph` 事实充足时生成内容 | "生成系统画像 report", "写验证脚本" |
| LEARN | WM loss 高 / RND 死亡 / 发现新意图候选 | "验证 CUSTOM 命令是否值得提升为意图", "收集 Conductor 对齐样本" |

### 4.3 实现建议

- 不要直接替换 `step()`，先新增 `MetaCognitiveSelector`，输出 `current_mode` 到 `state_text` 和日志。
- 让 `GoalGenerator` 根据 mode 选择目标池；`ActionExecutor` 不变。
- 前 100 步只用规则，避免小分类器冷启动乱切。

---

## 五、增长型世界模型 (Q3)

### 5.1 核+叶 (Core + Leaves) 架构

```python
# agent/world_model_v4.py
class WorldModelCore(nn.Module):
    """冻结或缓慢更新的共享表示网络"""
    def __init__(self, embed_dim=384, thought_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + thought_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

class IntentLeaf(nn.Module):
    """每个意图一个轻量叶网络"""
    def __init__(self, hidden_dim=128, thought_dim=16):
        super().__init__()
        self.heads = nn.ModuleDict({
            "exit": nn.Linear(hidden_dim, 2),
            "length": nn.Linear(hidden_dim, 3),
            "error": nn.Linear(hidden_dim, 2),
            "value": nn.Linear(hidden_dim, 1),
            "next_thought": nn.Linear(hidden_dim, thought_dim),
            "agreement": nn.Linear(hidden_dim, 2),
        })

class GrowingWorldModel(nn.Module):
    def __init__(self, ...):
        self.core = WorldModelCore(...)
        self.leaves = nn.ModuleDict({
            name: IntentLeaf(...) for name in INTENT_REGISTRY
        })
        self.intent_to_idx = {name: i for i, name in enumerate(INTENT_REGISTRY)}

    def forward(self, state_emb, thought, intent_name):
        x = torch.cat([state_emb, thought], dim=-1)
        h = self.core(x)
        return self.leaves[intent_name](h)

    def add_intent(self, name: str):
        if name not in self.leaves:
            self.leaves[name] = IntentLeaf(...)
            self.intent_to_idx[name] = len(self.intent_to_idx)
```

### 5.2 为什么这样做

- **新意图只加 leaf**: 不需要改输入层、不需要重新训练 core，避免 V3 中 `expand_intents` 重构第一层和优化器状态的麻烦。
- **意图专属参数**: 不同意图的输出分布差异大 (READ 多为短输出，CUSTOM 方差大)，共享 head 会互相拖后腿。
- **稀疏训练没问题**: 每个 leaf 只在该意图出现时更新，小样本也能学。
- **可解释性**: 可以单独分析某个意图的 value/uncertainty。

### 5.3 输入表示

V3 把意图 one-hot 拼在输入第一层；V4 改为:
- core 输入: `state_emb (384) + thought (16) = 400`
- intent 通过 leaf 选择路由，不再作为输入特征

这样状态表示与意图解耦，意图数量变化不影响 core。

### 5.4 好奇心信号增强

- 除 RND 外，使用 WM 每个 leaf 的 `agreement` 熵和 `value` 方差作为不确定性。
- 对每个意图维护一个 leaf-level 的 "预测误差 EMA"，高误差意图获得探索 bonus。

---

## 六、实施顺序 (Q4)

### Phase 0: 准备 (1-2 天)
1. **数据/指标基线**: 跑 500 步 P9.7，记录: 成功步数、事实数、CUSTOM 使用率、WM loss、RND 均值、intents 分布。
2. **抽离 `step()`**: 把 `online_agent.py` 的 `step()` 拆成 `_select_mode` / `_generate_goal` / `_execute_action` / `_learn`，即使内部仍是旧逻辑。
3. **添加持久化目录**: 把 `data/persistent/` 扩展为 facts/、wm/、meta/、buffer/，让 docker run 能 mount。

### Phase 1: 事实图 (最高优先级, 3-5 天)
1. 实现 `agent/fact_graph.py` (纯 dict + 邻接表)。
2. 修改 `Workbench` 内部用 `FactGraph`，保持接口不变。
3. 把硬编码 follow_up 链迁移到 schema + edge 驱动。
4. 在 `StateEncoder` 中加入 `graph.gaps()` 摘要。
5. A/B: 新旧 workbench 各跑 500 步，比较事实数量和发现链长度。

### Phase 2: 元认知 + 目标生成器 (3-4 天)
1. 实现 `agent/meta_selector.py` (规则 + 小分类器)。
2. 实现 `agent/goal_generator.py`，输入 mode + fact_graph + rnd + wm stats，输出 GOAL 队列。
3. 修改 `online_agent.step()` 顶层逻辑: MODE → GOAL → ACTION。
4. 把 diversity/goal-driven/probe/imagination 统一归到 GoalGenerator。
5. A/B: 固定 MODE 对比 (纯 EXPLORE vs 混合)。

### Phase 3: 增长型世界模型 (4-6 天)
1. 实现 `agent/world_model_v4.py` (core + leaves)。
2. 修改训练脚本，对新意图调用 `add_intent()`。
3. 把 `_imagine_intent()` 和 `rollout_top_k()` 迁移到 V4。
4. 在 V4 中加入 leaf-level curiosity bonus。
5. A/B: V3 vs V4，重点看 imagination 触发率和 value 预测准确率。

### Phase 4: 持久化 + 长期运行 (2-3 天)
1. 事实图、RND、WM、buffer、MetaLearner 全部按步 save。
2. 设计 docker `--mount` 到宿主持久目录，重启自动恢复。
3. 长程 10k 步测试，检查事实图增长、WM 是否过拟合、RND 是否死亡。

### Phase 5: 自动化意图增长 (可选，3-4 天)
1. 把 `IntentDiscoverer` 发现的新意图注册为新的 leaf 和分类器输出。
2. 实现 "意图孵化" 流程: CUSTOM 成功 → 聚类 → 临时意图 → 收集 30 条样本 → 正式意图。
3. 评估新增意图是否真的提高成功率。

---

## 七、关键风险 (Q5)

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| **事实图 schema 过于死板** | 中 | 新类别事实无法自动归类，gaps 误导 | 用 LLM/embedding 动态建议 schema；保留 "general" 兜底；定期人工 review |
| **MODE 小分类器冷启动乱切** | 高 | 前100步在 CREATE/LEARN 间振荡，事实没积累就生成垃圾 | 强制规则覆盖前100步；MODE 切换需最小持有步数 (如30步) |
| **WM core+leaves 训练不稳定** | 中 | leaves 数量多，梯度稀疏，某些 leaf 不被更新 | 使用 per-leaf optimizer (或共享 optimizer 但按 leaf mask 梯度)；叶子冷启动用 bootstrap |
| **RND 死亡/过拟合** | 高 | 长期运行后 RND error → 0，探索停止 | `interest_reset()` 已存在，但需按事实图 coverage 触发而非固定步数；引入 "curriculum reset" |
| **GoalGenerator 过于激进** | 中 | 链式目标无限延伸，agent 陷入局部验证循环 | 给 goal 加 "budget" (最大链长) 和 "utility threshold"，低效用目标 prune |
| **新意图扩展导致分类器坍缩** | 中 | 加新 leaf/output 后旧意图 recall 下降 | 每次扩展后跑 offline 蒸馏对齐；用 EWC / replay buffer 防遗忘 |
| **持久化/恢复引入状态不一致** | 高 | 恢复时 WM 与 fact_graph 版本不匹配 | 给 checkpoint 加版本号；恢复时校验 n_intents/n_nodes；不一致时软重置 |
| **重构 online_agent.py 引入回归** | 高 | step() 太复杂，任何拆分都可能破坏已知 stable 行为 | 保留旧 `step_legacy()` 路径做 A/B；新路径默认关闭，逐步提升比例 |

**最大翻车点**: 同时做 "事实图 + MODE + WM V4 + 持久化" 四个大改，导致无法定位 bug。必须分 Phase，每 Phase 跑完整基线。

---

## 八、具体代码改动 (Q6)

### 新增文件

| 文件 | 职责 |
|------|------|
| `agent/fact_graph.py` | `FactGraph` 类，节点/边/schema/gaps |
| `agent/meta_selector.py` | `MetaCognitiveSelector` (规则 + 小分类器) |
| `agent/goal_generator.py` | `GoalGenerator` (MODE + FactGraph + RND/WM → GOAL) |
| `agent/action_executor.py` | 封装 `ParameterExtractor + TemplateEngine + ErrorRecovery` |
| `agent/world_model_v4.py` | `GrowingWorldModel` core+leaves |
| `agent/persistent_store.py` | 统一 save/load: facts, wm, rnd, buffer, meta |

### 修改文件

| 文件 | 改动 |
|------|------|
| `agent/workbench.py` | 内部 `facts` → `FactGraph`；保持接口；边自动建立；gaps 驱动 follow-up |
| `agent/online_agent.py` | `step()` 拆为 `_select_mode` / `_generate_goal` / `_execute_action` / `_learn`；保留 legacy 路径；引入 GoalGenerator 和 MetaSelector |
| `agent/state_encoder.py` | state_text 加入 `gaps_summary` 和 `current_mode` |
| `agent/conductor.py` | `expand_intents` 改为支持新意图输出；thought_head 与 class_proj 解耦训练 |
| `agent/intent_classifier` (在 online_agent.py 内) | `expand_intents` 保留，但每次扩展后 freeze 旧输出并 replay |
| `agent/intent_discoverer.py` | 发现新意图时返回完整元数据，供 V4 `add_intent` 和 classifier 扩展 |
| `agent/rnd.py` | 增加基于 fact_graph coverage 的 reset 触发；novelty 可针对 graph 节点 |
| `agent/experience.py` | 经验增加 `mode`, `goal`, `fact_delta` 字段 |
| `scripts/train_online.py` | 训练循环支持 V4；保存 checkpoint 含版本号 |
| `docker/` 或启动脚本 | 添加 `--mount` 到宿主持久目录 |

### 不急着改的文件

- `agent/command_selector_v2.py`: 仍可被 CUSTOM 调用，但未来应被 GoalGenerator 统一。
- `agent/nanny.py`:  translator 逻辑稳定，主要改 Conductor 扩展方式。

---

## 九、可立即执行的小步实验

如果你想先验证方向，建议先做这 3 个低风险实验:

1. **事实图最小版**: 新建 `agent/fact_graph.py`，把 `Workbench.facts` 包装成 `FactGraph.nodes`，跑 200 步看 `gaps()` 输出是否合理。
2. **MODE 规则版**: 不训练小分类器，只写规则 `_select_mode`，在日志里打印 MODE，不改 action 选择。
3. **WM V4 原型**: 用 5 个意图做 core+leaves 训练，比较 V3 和 V4 在 replay buffer 上的 loss。

每个实验都应在 `CHANGELOG.md` 记录指标。

---

## 十、总结

P9.7 到层级架构的升级**不是一次重构能完成的**。正确顺序是:

1. **事实图先行** — 它是 MODE 和 GOAL 的输入，没有图就无法做事实缺口驱动。
2. **元认知与目标生成器其次** — 把扁平的 `step()` 决策逻辑收束到可解释的层次。
3. **增长型 WM 最后** — 它依赖稳定的事实表示和目标信号，否则 leaf 训练数据太稀疏。
4. **全程 A/B + 基线** — 任何大改都保留旧路径，逐步提高新路径比例。

最大风险是急于求成同时改多处；最大收益是把 `online_agent.py` 从 2142 行的决策泥球变成可组合、可观测、可增长的认知栈。
