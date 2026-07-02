# CaSolis_ P10 后层级架构升级评审

> 评审范围: P10 已完成层级架构 (MODE->GOAL->ACTION), 16 intent 100% stable
> 目标: 升级为 层级架构 + 动态事实图 + 增长型世界模型 + 长期自主学习
> 最后更新: 2026-06-25

---

## 0. 现状诊断: P10 已解决什么, 还缺什么

### 0.1 P10 已落地 [OK]

| 模块 | 状态 |
|------|------|
| 16 intent 分类 | stable, 100% 成功 |
| MetaCognitiveSelector | 3 MODE (EXPLORE/CREATE/LEARN), 规则 + 小 MLP |
| GoalGenerator | 基于 FactGraph gaps + RND + WM loss 生成目标 |
| FactGraph | dict + 邻接表, schema gaps, JSON 序列化 |
| GrowingWorldModel V4 | core + per-intent leaves, auto-expand |
| EpisodicMemory | surprise ring buffer + cosine recall |
| CreativeWriter | Ollama gemma4:e4b, 4 风格, fallback |

### 0.2 还缺什么 [FAIL]

1. **持久化不完整**: Workbench 快照存到 `data/persistent/`, 但 classifier head / WM V4 / experience buffer / meta selector 的 checkpoint 散落, 跨 run 恢复不统一。
2. **知识面太窄**: 只读 `/proc/cpuinfo`, `/etc/os-release`, `free -h`, 没系统性探索 `/sys`, `/proc/net`, `/usr/bin`, `dpkg`, `man`。
3. **不会自造工具**: 能写 shell 脚本, 不会生成并注册可复用的 Python 工具。
4. **LLM 太慢**: gemma4:e4b 9GB CPU 60s, 阻塞主循环。
5. **无跨 session 学习**: checkpoint 能存, 但恢复逻辑不统一, 知识增长在 docker 重启后归零。
6. **模板束缚**: WRITE/GENERATE 输出仍依赖 `profile.md`, `discovery.json`, `check.sh` 模板。
7. **无解释能力**: 只能报告事实, 不能解释 "这些事实放在一起意味着什么"。
8. **无网络**: `--network none` 下无法获取外部知识。

---

## 1. 目标架构: 三层 + 两图 + 一记忆

把人脑的分层概念映射到代码, 但保持工程可实现。

```
┌─────────────────────────────────────────────────────────────┐
│  顶层: 元认知层 (MetaCognitive) — 前额叶                    │
│  MetaCognitiveSelector 3 MODE: EXPLORE / CREATE / LEARN     │
│  输入: FactGraph 覆盖度、RND、WM loss、历史 MODE 时长        │
│  输出: 当前 MODE + intent bias                               │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  中层: 目标层 (Goal) — 海马/计划                             │
│  GoalGenerator                                             │
│  输入: MODE + FactGraph.gaps() + RND + WM 分歧              │
│  输出: Goal 队列 (fact_fill / curiosity / create / verify)  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  底层: 动作层 (Action) — 运动皮层                            │
│  ActionExecutor                                            │
│  输入: Goal + state_text                                    │
│  输出: 命令执行 + 结果回写 FactGraph                          │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌──────────┐        ┌──────────┐        ┌──────────────┐
  │ Dynamic  │        │ Growing  │        │ Episodic     │
  │ FactGraph│        │ World    │        │ Memory       │
  │          │        │ Model V5 │        │              │
  │ 节点=事实 │◄──────►│ core+叶  │◄──────►│ 惊喜事件     │
  │ 边=关系   │        │ 预测+注意│        │ 相似召回     │
  │ schema=缺 │        │ 自扩展   │        │              │
  └──────────┘        └──────────┘        └──────────────┘
```

### 1.1 关键接口契约

```python
# MetaCognitiveSelector
mode = meta.select(state_summary, fact_graph.stats(), rnd_stats, wm_loss, step)
bias = meta.get_intent_bias(mode)  # dict[intent, weight]

# GoalGenerator
goals = goal_gen.generate(mode, fact_graph, workbench, step, top_k=3)
# Goal = namedtuple("Goal", ["type", "intent", "params", "priority", "source", "description", "budget"])

# ActionExecutor
result = action_exec.run(goal, state_text)
# result -> (success, reward, fact_delta, stdout_sample)

# FactGraph (已是动态图, 需增强)
fact_graph.add_node(key, value, category, confidence, step, source_cmd)
fact_graph.add_edge(from_key, to_key, rel, weight)
gaps = fact_graph.find_gaps()                    # schema 缺什么
new_categories = fact_graph.suggest_categories() # 自动发现新类别

# GrowingWorldModel V5
pred = world_model.simulate(state_emb, thought, intent_name)  # 已有
attention = world_model.attend_over_facts(fact_graph, state_emb)  # 新增
world_model.reflect(pred, actual)  # 元认知: 预测 vs 实际 → 模型置信度
```

---

## 2. 核心升级设计

### 2.1 动态事实图 (Dynamic FactGraph) 增强

当前 FactGraph 已经很好, 但 schema 是硬编码的。升级为:

```python
class FactGraph:
    def __init__(self, max_nodes=200):
        self.nodes: dict[str, Node]
        self.edges: dict[str, list[Edge]]
        self.schemas: dict[str, list[str]]          # 可扩展
        self.category_embeddings: dict[str, Tensor] # 用于自动归类
        self.discovery_log: list[DiscoveryEvent]    # 历史发现

    def suggest_category(self, key: str, value: str) -> str:
        """基于 key/value 与现有 schema 的 embedding 相似度, 建议类别"""

    def auto_expand_schema(self, key: str, category: str):
        """新 key 出现时自动加入 schema[category]"""

    def build_summary(self) -> str:
        """把图翻译成自然语言摘要, 供 LLM/state_text 使用"""
```

新增边类型:

- `requires` / `verifies` / `extends` / `located_in` / `derived_from` / `conflicts` / `same_as` — 已有
- `enables` — A 使 B 成为可能 (e.g. python3 -> 可运行 Python 工具)
- `part_of` — A 是 B 的一部分 (e.g. cpu_cores -> cpu)
- `has_capability` — 系统能力 (e.g. system -> can_run_docker)

### 2.2 增长型世界模型 V5 (GrowingWorldModel)

V4 的 core+leaves 是好的。下一步加三层脑启发机制:

#### A. 图注意力 (Graph Attention over Facts)

在 forward 之前, 用 GAT 或简化注意力从 FactGraph 节点中提取 "当前最相关的事实子图", 与 state_emb 拼接。

```python
class FactAttention(nn.Module):
    def __init__(self, node_dim, query_dim, n_heads=2):
        ...
    def forward(self, query: Tensor, nodes: list[Node]) -> Tensor:
        # 计算 query 与每个 node 的 attention weight
        # 输出加权聚合的 node representation
```

作用: 让模型 "注意到" 与当前意图最相关的事实, 而不是把所有事实平等对待。

#### B. 递归自我反思 (Recursive Self-Reflection)

每步执行后:

1. WM 预测 next_state / reward / exit / agreement
2. 实际结果返回
3. 计算 `reflection_error = |pred - actual|` 在多个维度上的加权
4. 把 reflection_error 写回 MetaCognitiveSelector 作为输入
5. 如果 reflection_error 持续高, 强制切到 LEARN MODE

```python
class SelfReflection:
    def __init__(self, window=20):
        self.errors = deque(maxlen=window)

    def update(self, pred: dict, actual: dict):
        err = self.compute_error(pred, actual)
        self.errors.append(err)

    def model_confidence(self) -> float:
        # 误差低且稳定 -> 高置信度
        if len(self.errors) < 5: return 0.5
        return 1.0 - min(1.0, mean(self.errors) + std(self.errors))
```

#### C. 元认知置信度门控 (Thalamus Gating)

在 ACTION 选择前, 先问 WM:

- "我对这个意图的预测置信度是多少?"
- 如果置信度 < 0.3, 降级到 EXPLORE (收集更多数据)
- 如果置信度 > 0.8, 允许 CREATE (生成复杂输出)

这就是 "dynamic layers like human brain network": 不是所有层同时激活, 而是根据置信度和 MODE 动态选择哪些层参与决策。

### 2.3 目标生成器 (GoalGenerator) 升级

当前 GoalGenerator 已经按 MODE 生成候选, 但缺少:

1. **目标预算 (budget)**: 防止链式目标无限延伸。
2. **目标可行性评估**: 太难了切小, 太简单了跳过。
3. **跨 MODE 目标转换**: 例如 EXPLORE 发现足够多事实后, 自动插入一个 CREATE 目标。

```python
@dataclass
class Goal:
    type: str                # gap_fill / curiosity / create / verify / tool_build
    intent: str
    params: dict
    priority: float
    source: str              # 可追溯来源
    description: str
    budget: int = 3          # 最大后续链长
    min_confidence: float = 0.0  # 执行前 WM 置信度门槛

class GoalGenerator:
    def generate(self, mode, fact_graph, workbench, step, top_k=3) -> list[Goal]:
        candidates = []
        if mode == "EXPLORE":
            candidates += self._gap_goals(fact_graph)
            candidates += self._curiosity_goals(rnd, wm)
            candidates += self._systematic_mapper_goals(step)  # 知识 campaign
        elif mode == "CREATE":
            candidates += self._report_goals(fact_graph)
            candidates += self._tool_build_goals(fact_graph)   # 自造工具
        elif mode == "LEARN":
            candidates += self._verify_goals(fact_graph)
            candidates += self._train_goals(wm_loss)
        # 过滤 + 排序 + budget 检查
        return self._rank(candidates)[:top_k]
```

---

## 3. 回答六个问题

### Q1: 演进顺序

你提出的顺序基本正确, 但需要微调:

```
P10 ─► P11 持久化 (必须先有, 否则后面全是白做)
   │
   ▼
P12 知识拓展 campaign (让 FactGraph 真正 "了解系统")
   │
   ▼
P13 自造工具工厂 (把知识变成可复用能力)
   │
   ▼
P14 LLM 双模型 (解决速度瓶颈)
   │
   ▼
P15 脑启发层 (图注意力 + 自我反思 + 元认知门控)
```

**为什么 persistence 必须第一?**

- 知识 campaign 会产生大量新事实和 schema, 如果不持久, 每次 docker 重启都从零开始, 探索无意义。
- 自造工具需要注册表持久化, 否则每次重启要重新发现工具。
- WM V4 的 leaves 需要持续训练, 丢失 leaves 等于丢失学到的世界模型。

**为什么 LLM 放在 tool-building 之后?**

- 工具工厂可以先基于模板/规则生成 Python 工具, 不完全依赖 LLM, 先验证机制再优化速度。
- LLM 是加速器, 不是先决条件。

### Q2: 持久化策略

#### 现状问题

- Workbench 快照: `data/persistent/workbench_snapshot.json` [OK] 已挂载到 docker
- WM V4: `checkpoints/world_model/v4_latest.pt` [FAIL] 不在 docker mount 里 (如果 agent 本身在 docker 内)
- classifier head: `checkpoints/intent_classifier/best_head.pt` [FAIL]
- experience buffer: `checkpoints/online_agent/experience.jsonl` [FAIL]
- meta selector MLP: `checkpoints/meta_selector/mlp.pt` [FAIL] (可能有)

#### 推荐方案: Unified PersistentStore + 双挂载

新增 `agent/persistent_store.py`:

```python
class PersistentStore:
    """统一持久化: 结构数据用 SQLite, torch 模型用文件, 全部放在 /persistent"""
    def __init__(self, base_path: str = "data/persistent"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.base / "casolis.db")
        self._init_tables()

    def save_fact_graph(self, graph: FactGraph):
        self.db.execute("REPLACE INTO fact_graph VALUES (?, ?)", ("main", json.dumps(graph.to_dict())))
        self.db.commit()

    def load_fact_graph(self) -> FactGraph:
        row = self.db.execute("SELECT data FROM fact_graph WHERE id='main'").fetchone()
        return FactGraph.from_dict(json.loads(row[0])) if row else FactGraph()

    def save_torch(self, name: str, state_dict: dict):
        torch.save(state_dict, self.base / "models" / f"{name}.pt")

    def load_torch(self, name: str) -> dict | None:
        path = self.base / "models" / f"{name}.pt"
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
```

表设计:

```sql
CREATE TABLE fact_graph (id TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP);
CREATE TABLE experience (id INTEGER PRIMARY KEY, step INT, state TEXT, intent TEXT, reward REAL, mode TEXT, goal TEXT);
CREATE TABLE discovery_log (step INT, key TEXT, value TEXT, category TEXT, source_cmd TEXT);
CREATE TABLE tool_registry (name TEXT PRIMARY KEY, path TEXT, description TEXT, uses INT, last_used INT);
CREATE TABLE meta_history (step INT, mode TEXT, reason TEXT);
```

#### Docker 侧

如果 agent 本身跑在 docker 里, 启动时挂载:

```bash
docker run -d \
  -v "$(pwd)/data/persistent:/app/data/persistent" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  --network none \
  --name casolis-agent \
  casolis-agent:latest
```

如果 agent 跑在宿主, 只有 sandbox 在 docker 里, 则保证 sandbox 的 `/persistent` mount 继续保留即可。

#### 版本兼容

- checkpoint 加 `version` 字段 (如 `{"version": "P11", "n_intents": 16, "n_nodes": 120}`)
- load 时检查, 不一致时软重置 (保留旧数据作为 replay)

### Q3: 知识拓展 campaign

#### 设计: `KnowledgeMapper` 系统性 BFS

把系统信息空间抽象成两张图:

1. **命令空间**: `/usr/bin`, `/bin`, `/sbin`, `/usr/sbin` 里的可执行文件
2. **文件系统空间**: `/sys`, `/proc`, `/etc`, `/dev` 等伪文件系统

```python
class KnowledgeMapper:
    def __init__(self, sandbox, fact_graph, budget=50):
        self.sandbox = sandbox
        self.fg = fact_graph
        self.budget = budget          # 每轮最大探索步数
        self.visited_cmds = set()
        self.visited_paths = set()

    def campaign(self, phase: str):
        if phase == "commands_inventory":
            self._inventory_commands()
        elif phase == "command_introspect":
            self._introspect_commands()
        elif phase == "filesystem_bfs":
            self._bfs_filesystem()
        elif phase == "package_map":
            self._map_packages()

    def _inventory_commands(self):
        result = self.sandbox.execute("find /usr/bin /bin /sbin /usr/sbin -maxdepth 1 -type f -executable 2>/dev/null | sort")
        cmds = result.stdout.strip().splitlines()
        for cmd in cmds[:self.budget]:
            name = Path(cmd).name
            self.fg.add_node(f"cmd_{name}", name, category="command", source_cmd="find commands")
            self.fg.add_edge("system", f"cmd_{name}", "has_capability")
```

#### Campaign 阶段

```
Phase A: 静态清单 (只读, 安全)
  - ls /usr/bin /bin /sbin
  - dpkg -l | head -200
  - ls /proc /sys /etc
  - find /sys/class -maxdepth 2 -type d
  - find /proc/net -maxdepth 1 -type f

Phase B: 命令自描述 (低风险)
  - cmd --help / -h (timeout 2s)
  - man cmd (取 NAME/SYNOPSIS 第一段)
  - whatis cmd
  - which cmd

Phase C: 子系统 BFS (中风险, 只读)
  - /sys/class/cpu, /sys/class/net, /sys/class/block
  - /proc/net/dev, /proc/net/route, /proc/net/arp
  - /proc/filesystems, /proc/mounts
  - /etc/* release/os/network config

Phase D: 动态执行 (只读命令)
  - uname -a, lsb_release -a, hostnamectl
  - ip addr, ip route
  - df -h, lsblk
  - ps aux --no-headers | head -50
  - env

Phase E: 能力推断
  - 根据已有命令推断 "system can_run_python", "system can_run_docker"
  - 根据 /proc/net 推断 network topology
```

#### Schema 自动扩展

探索过程中发现新类别时, `FactGraph.auto_expand_schema()` 自动加入:

```python
# 发现大量 cmd_* 节点 -> 创建 command schema
if self.fg.node_count_by_category("command") > 5 and "command" not in self.fg.schemas:
    self.fg.schemas["command"] = ["name", "path", "help_summary", "package"]

# 发现 /sys/class/net -> 创建 network_interface schema
if self.fg.has_node("sys_class_net"):
    self.fg.schemas.setdefault("network_interface", []).extend(["name", "mac", "ip", "state"])
```

#### 安全限制

- 只读命令白名单
- 禁止: `rm`, `mkfs`, `dd`, `iptables -F`, `passwd`
- 每个命令 timeout 2s, 输出截断 4KB
- `--network none` 天然防止外部连接

### Q4: 自造工具机制

#### 最小可运行设计: ToolFactory + ToolRegistry

```python
# agent/tool_factory.py
class ToolFactory:
    def __init__(self, llm=None):
        self.llm = llm

    def design_tool(self, need: str, facts: FactGraph) -> dict:
        """根据需求和事实生成 Python 工具"""
        prompt = self._build_prompt(need, facts)
        code = self.llm.generate(prompt) if self.llm else self._rule_based_template(need)
        return {
            "name": f"tool_{need}_{step}",
            "description": need,
            "code": code,
        }

    def _rule_based_template(self, need: str) -> str:
        # 无 LLM 时也能生成简单工具
        if "list" in need and "packages" in need:
            return "import subprocess\ndef run(env):\n    r=subprocess.run(['dpkg','-l'],capture_output=True,text=True)\n    return {'packages': r.stdout.splitlines()[:50]}\n"
        ...
```

工具必须满足统一接口:

```python
# /persistent/tools/tool_list_packages.py
def run(env: dict) -> dict:
    """env 包含 fact_graph, sandbox 等上下文"""
    import subprocess
    result = subprocess.run(["dpkg", "-l"], capture_output=True, text=True)
    return {"packages": result.stdout.splitlines()[:100]}
```

#### ToolRegistry

```python
class ToolRegistry:
    def __init__(self, store: PersistentStore):
        self.store = store
        self.tools: dict[str, callable] = {}
        self.discover()

    def discover(self):
        tools_dir = Path("data/persistent/tools")
        for path in tools_dir.glob("tool_*.py"):
            self._register_from_file(path)

    def _register_from_file(self, path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "run"):
            self.tools[path.stem] = mod.run
            self.store.upsert_tool(path.stem, str(path), getattr(mod, "__doc__", ""))

    def call(self, name: str, env: dict) -> dict:
        return self.tools[name](env)
```

#### 与 GoalGenerator 集成

在 CREATE mode 下, 如果事实显示某个操作重复出现, 生成 `tool_build` 目标:

```python
def _tool_build_goals(self, fact_graph):
    goals = []
    # 如果系统里有很多 package 相关事实, 但缺少一个 "list all packages" 的工具
    if fact_graph.has_node("cmd_dpkg") and "tool_list_packages" not in self.registry.tools:
        goals.append(Goal(
            type="tool_build",
            intent="WRITE",
            params={"path": "/persistent/tools/tool_list_packages.py", "content": ...},
            priority=0.75,
            source="missing_tool:package_inventory",
            description="自造工具: 列出已安装软件包"
        ))
    return goals
```

#### 工具复用

- 注册后, `ActionExecutor` 新增 `TOOL` 意图
- 参数: `tool_name`, `args`
- 执行时调用 `registry.call(tool_name, env)`, 结果写回 FactGraph

### Q5: 脑启发下一步

V4 core+leaves 是 "静态结构 + 增长叶子"。下一步是 **动态层激活**。

#### 三个脑启发升级

**1. 图注意力 (Cortical Column Attention)**

不是把整个 FactGraph 喂给模型, 而是根据当前意图和状态, 动态选择最相关的节点子集。

```python
# 在 WM forward 前
relevant_nodes = fact_attention(query=state_emb, nodes=fact_graph.nodes.values(), top_k=10)
fact_vector = aggregate(relevant_nodes)
input_vector = torch.cat([state_emb, thought, fact_vector], dim=-1)
```

**2. 递归自我反思 (Prefrontal Reflection)**

每步形成一个 "感知-预测-执行-比较-更新置信度" 的闭环。

```python
pred = world_model.simulate(state_emb, thought, intent)
actual_reward, actual_exit = execute(goal)
reflection.update(pred, {"reward": actual_reward, "exit": actual_exit})
meta_confidence = reflection.model_confidence()
# 低置信度 -> 禁止 CREATE, 强制 EXPLORE
```

**3. 元认知门控 (Thalamus / Mode Gate)**

根据置信度和不确定性, 动态决定哪些认知层参与:

| 条件 | 激活层 |
|------|--------|
| 高置信度 + 事实充足 | CREATE (LLM / tool building) |
| 高不确定性 + 新刺激 | EXPLORE (knowledge mapper) |
| 模型预测持续错误 | LEARN (training + verification) |
| 工具可用 | TOOL layer 优先于 CUSTOM |

```python
class ModeGate:
    def select_layers(self, meta_confidence, fact_coverage, tool_available):
        layers = ["classifier"]
        if meta_confidence > 0.7 and fact_coverage > 0.6:
            layers.append("creative")
        if meta_confidence < 0.4:
            layers.append("explore")
        if tool_available:
            layers.append("tool")
        return layers
```

这三个合起来, 就是 "dynamic layers like human brain network"。

### Q6: LLM 策略

#### 推荐: 双模型分层 (Small-Fast + Big-Slow)

| 模型 | 角色 | 调用时机 | 预算 |
|------|------|----------|------|
| qwen3.5:0.8b | 快思考 / 边缘皮层 | 每步决策、意图粗分、目标草拟、工具选择 | < 2s |
| gemma4:e4b | 慢思考 / 前额叶 | 深度报告、代码生成、分析解释、工具设计 | 允许 10-60s |

#### 具体分工

**qwen3.5:0.8b (实时)**:

- 把 state_text 压缩成 "认知摘要"
- 判断当前是否适合调用大模型
- 快速选择 goal type (EXPLORE/CREATE/LEARN/TOOL)
- 简单工具参数填充

**gemma4:e4b (深度)**:

- 生成系统报告 / 分析 (CREATE mode)
- 设计复杂 Python 工具
- 解释事实图 "这些事实意味着什么"
- 长文本创作

#### 调度器

```python
class LLMRouter:
    def __init__(self, small, big):
        self.small = small
        self.big = big

    def route(self, task: str, facts: FactGraph, thermal_ok: bool) -> dict:
        # 先用小模型判断是否需要大模型
        need_big = self.small.classify(task)
        if need_big and thermal_ok and task in {"report", "code", "analysis"}:
            return self.big.generate(task, facts)
        return self.small.generate(task, facts)
```

#### 优化 gemma4 CPU 速度

- 用 `llama.cpp` / `ollama` 的量化版本 (4-bit)
- 如果主机有 Intel Arc, 尝试 IPEX-LLM / OpenVINO
- 异步生成: CreativeWriter 已经支持 async, 让 LLM 在后台跑, agent 继续 EXPLORE
- thermal gating: 已经实现, 保持

---

## 4. 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 持久化状态不一致 | 高 | 恢复失败, 训练数据丢失 | 版本号 + 软重置 + replay buffer |
| 知识 campaign 触发危险命令 | 中 | 沙箱损坏/数据丢失 | 只读白名单 + timeout + 输出截断 |
| 自造工具生成恶意代码 | 中 | 沙箱内破坏 | 沙箱隔离 + 工具执行前静态检查 + 禁止网络 |
| MODE 频繁切换 | 中 | 无法完成任何目标 | 最小持有步数 + 置信度平滑 |
| LLM 幻觉污染 FactGraph | 中 | 错误事实持久化 | 事实只接受命令输出, LLM 输出进 "opinion" 节点 |
| RND 死亡/过拟合 | 高 | 探索停止 | 按 coverage 触发 reset + curriculum reset |
| 新意图扩展导致分类器坍缩 | 中 | recall 下降 | freeze 旧输出 + replay + EWC |
| 图注意力增加推理延迟 | 中 | 实时性下降 | 节点数上限 200, 注意力 top_k=10 |
| 过度工程 | 高 | 拖延, 基线不稳定 | 每 Phase 跑 A/B, 保留 legacy 路径 |

---

## 5. 实施路线图

### Phase 0: 基线与拆分 (1-2 天)

- [ ] 跑 300 步 P10 基线, 记录: 事实数、 intents 分布、WM loss、RND 均值、MODE 时长
- [ ] 把 `online_agent.step()` 拆成 `_select_mode` / `_generate_goal` / `_execute_action` / `_learn`
- [ ] 保留 `step_legacy()` 做 A/B

### Phase 1: 统一持久化 (P11, 2-3 天)

- [ ] 新增 `agent/persistent_store.py` (SQLite + torch file)
- [ ] 统一保存/加载: FactGraph, WM V4, classifier head, experience buffer, meta selector, tool registry
- [ ] checkpoint 加版本号与校验
- [ ] Docker 启动脚本挂载 `data/persistent` 和 `checkpoints`
- [ ] 启动时自动 load, 结束时自动 save, 每 50 步 checkpoint

### Phase 2: 知识拓展 campaign (P12, 3-5 天)

- [ ] 新增 `agent/knowledge_mapper.py`
- [ ] Phase A/B/C/D/E 分阶段探索
- [ ] FactGraph schema 自动扩展
- [ ] 安全白名单与 timeout
- [ ] 在 EXPLORE mode 下周期性触发 campaign

### Phase 3: 自造工具工厂 (P13, 3-4 天)

- [ ] 新增 `agent/tool_factory.py` + `agent/tool_registry.py`
- [ ] 统一工具接口 `run(env) -> dict`
- [ ] 无 LLM 规则模板 + LLM 深度生成双路径
- [ ] 新增 `TOOL` 意图到 ActionExecutor
- [ ] CREATE mode 下检测缺失工具并生成

### Phase 4: LLM 双模型 (P14, 2-3 天)

- [ ] 部署 qwen3.5:0.8b
- [ ] 新增 `agent/llm_router.py`
- [ ] 小模型负责实时路由, 大模型负责深度生成
- [ ] CreativeWriter 改为异步 + thermal gate

### Phase 5: 脑启发层 (P15, 4-6 天)

- [ ] FactGraph 图注意力接入 WM V5
- [ ] 自我反思模块 `SelfReflection`
- [ ] 元认知门控 `ModeGate`
- [ ] 动态层激活: 根据置信度选择参与模块
- [ ] 集成到主循环, A/B 测试

---

## 6. 下一步可立即执行的三件事

1. **创建 `agent/persistent_store.py` 并替换散落 save/load**
   - 这是收益最大、风险最小的第一步
   - 让 P10 状态真正跨 run 存活

2. **扩展 SandboxExecutor docker run 挂载**
   - 把 `checkpoints/` 也 mount 进容器 (如果 agent 在容器内)
   - 或者确保宿主 `checkpoints/` 与 `data/persistent/` 备份一致

3. **写一个 `KnowledgeMapper` 的最小原型**
   - 先做 Phase A (命令清单 + dpkg + /proc /sys 列表)
   - 跑 100 步, 看 FactGraph 节点从 ~20 增长到多少
   - 用数据验证 campaign 价值

---

## 7. 总结

P10 已经完成 "从扁平到层级" 的质变。P11-P15 的核心是 **从 "能稳定跑" 变成 "能长期学"**:

- 持久化让学习有记忆
- 知识 campaign 让记忆有内容
- 自造工具让内容有能力
- 双 LLM 让能力有速度
- 脑启发层让速度有方向

最大风险是同时改多处。务必每 Phase 做 A/B, 保留 legacy 路径, 用数据说话。
