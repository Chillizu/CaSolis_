"""
统一持久化存储 — PersistentStore

核心职责:
  1. 统一管理所有状态的 save/load 入口
  2. SQLite 存结构数据 (FactGraph 元数据/工具注册/统计)
  3. PyTorch 模型存 data/persistent/models/
  4. JSON/JSONL 存可读数据 (FactGraph 快照/经验缓冲)
  5. Version 控制 + 自动迁移
  6. 所有路径在 data/persistent/ 下, Docker volume 挂载

用法:
    store = PersistentStore()
    store.save_all(agent)
    agent = store.load_all(intents=INTENTS, ...)
"""

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import torch

PERSISTENT_DIR = Path("data/persistent")

# ── 目录结构 ──
MODELS_DIR = PERSISTENT_DIR / "models"
TOOLS_DIR = PERSISTENT_DIR / "tools"
DB_PATH = PERSISTENT_DIR / "folunar.db"
VERSION_FILE = PERSISTENT_DIR / "version.txt"

SCHEMA_VERSION = 1


class PersistentStore:
    """统一持久化存储"""

    def __init__(self, base_dir: str | Path = PERSISTENT_DIR):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.tools_dir = self.base_dir / "tools"
        self.db_path = self.base_dir / "folunar.db"
        self._lock = threading.Lock()

        # 确保目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        # SQLite 初始化
        self._init_db()

        # Version 检查
        self._check_version()

    # ── SQLite 初始化 ──

    def _init_db(self):
        """初始化 SQLite 数据库"""
        self._db_conn = sqlite3.connect(str(self.db_path))
        self._db_conn.row_factory = sqlite3.Row
        c = self._db_conn.cursor()

        # 工具注册表
        c.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                name TEXT PRIMARY KEY,
                tool_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                source TEXT DEFAULT 'factory',
                created_step INTEGER DEFAULT 0,
                last_used_step INTEGER DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                avg_bytes_created REAL DEFAULT 0
            )
        """)

        # 运行统计
        c.execute("""
            CREATE TABLE IF NOT EXISTS run_stats (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                n_steps INTEGER,
                success_rate REAL,
                n_intents_covered INTEGER,
                total_reward REAL,
                fact_graph_nodes INTEGER,
                fact_graph_edges INTEGER,
                schema_coverage REAL,
                llm_calls INTEGER,
                llm_success INTEGER,
                llm_fallback INTEGER
            )
        """)

        # 探索进度 (KnowledgeMapper)
        c.execute("""
            CREATE TABLE IF NOT EXISTS exploration_progress (
                phase TEXT PRIMARY KEY,
                completed_at TEXT,
                n_discovered_commands INTEGER,
                n_discovered_files INTEGER,
                n_discovered_packages INTEGER,
                n_new_facts INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS key_value (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        self._db_conn.commit()

    # ── Version 控制 ──

    def _check_version(self):
        """检查版本, 不一致则迁移"""
        current = self._read_version()
        if current is None:
            self._write_version(SCHEMA_VERSION)
            return
        if current < SCHEMA_VERSION:
            self._migrate(current, SCHEMA_VERSION)
            self._write_version(SCHEMA_VERSION)
        elif current > SCHEMA_VERSION:
            print(f"  ⚠️ PersistentStore: 数据版本 {current} > 代码版本 {SCHEMA_VERSION}, 降级可能有损")

    def _read_version(self) -> Optional[int]:
        if VERSION_FILE.exists():
            try:
                return int(VERSION_FILE.read_text().strip())
            except (ValueError, OSError):
                return None
        return None

    def _write_version(self, v: int):
        VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VERSION_FILE.write_text(str(v))

    def _migrate(self, old_v: int, new_v: int):
        """版本迁移"""
        print(f"  ⚠️ PersistentStore: 迁移 v{old_v} → v{new_v}")
        if old_v < 1:
            pass  # 初始版本, 无需迁移
        # 未来: if old_v < 2: ...

    # ── KV 存储 ──

    def _kv_get(self, key: str, default: Any = None) -> Any:
        c = self._db_conn.cursor()
        c.execute("SELECT value FROM key_value WHERE key=?", (key,))
        row = c.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return default

    def _kv_set(self, key: str, value: Any):
        import json
        c = self._db_conn.cursor()
        if isinstance(value, (dict, list, int, float, bool)):
            val = json.dumps(value)
        else:
            val = str(value)
        c.execute("INSERT OR REPLACE INTO key_value (key, value) VALUES (?, ?)", (key, val))
        self._db_conn.commit()

    # ── FactGraph ──

    def save_fact_graph(self, graph: Any, path: Optional[str] = None):
        """保存 FactGraph 到 JSON"""
        path = path or str(self.base_dir / "fact_graph.json")
        if hasattr(graph, 'save_json'):
            graph.save_json(path)
            self._kv_set("fact_graph_path", path)
            self._kv_set("fact_graph_n_nodes", len(graph.nodes))
            self._kv_set("fact_graph_n_edges", sum(len(e) for e in graph.edges.values()))

    def load_fact_graph(self, graph_cls: type = None, path: Optional[str] = None):
        """加载 FactGraph"""
        path = path or str(self.base_dir / "fact_graph.json")
        if not os.path.exists(path):
            return None
        try:
            if graph_cls and hasattr(graph_cls, 'load_json'):
                return graph_cls.load_json(path)
            # 兜底: 直接读 dict
            from agent.fact_graph import FactGraph
            return FactGraph.load_json(path)
        except Exception as e:
            print(f"  ⚠️ FactGraph 加载失败: {e}")
            return None

    # ── Workbench ──

    def save_workbench(self, workbench: Any):
        """保存 Workbench 状态"""
        if hasattr(workbench, 'save'):
            workbench.save()
            self._kv_set("workbench_last_save_step", str(getattr(workbench, 'step', 0)))

    def load_workbench(self, workbench: Any) -> int:
        """加载 Workbench, 返回恢复的事实数"""
        if hasattr(workbench, 'load'):
            return workbench.load()
        return 0

    # ── World Model V3 ──

    def save_world_model_v3(self, world_model: Any):
        """保存 V3 世界模型"""
        path = str(self.models_dir / "wm_v3.pt")
        if hasattr(world_model, 'save'):
            world_model.save(path)
            self._kv_set("wm_v3_path", path)

    def load_world_model_v3(self, world_model: Any, path: Optional[str] = None):
        """加载 V3 世界模型"""
        path = path or str(self.models_dir / "wm_v3.pt")
        if not os.path.exists(path):
            return False
        try:
            if hasattr(world_model, 'load'):
                world_model.load(path)
                return True
        except Exception as e:
            print(f"  ⚠️ WM V3 加载失败: {e}")
        return False

    # ── World Model V4 ──

    def save_world_model_v4(self, wm_v4: Any):
        """保存 V4 增长型世界模型"""
        path = str(self.models_dir / "wm_v4.pt")
        if hasattr(wm_v4, 'save'):
            wm_v4.save(path)
            self._kv_set("wm_v4_path", path)
            self._kv_set("wm_v4_n_leaves", len(getattr(wm_v4, 'leaf_predictors', {})))

    def load_world_model_v4(self, wm_v4: Any, path: Optional[str] = None):
        """加载 V4 世界模型"""
        path = path or str(self.models_dir / "wm_v4.pt")
        if not os.path.exists(path):
            return False
        try:
            if hasattr(wm_v4, 'load'):
                wm_v4.load(path)
                return True
        except Exception as e:
            print(f"  ⚠️ WM V4 加载失败: {e}")
        return False

    # ── Classifier ──

    def save_classifier(self, classifier: Any):
        """保存分类器头"""
        path = str(self.models_dir / "classifier_head.pt")
        if hasattr(classifier, 'save'):
            classifier.save(path)
            self._kv_set("classifier_path", path)

    def load_classifier(self, classifier: Any, path: Optional[str] = None):
        """加载分类器"""
        path = path or str(self.models_dir / "classifier_head.pt")
        if not os.path.exists(path):
            return False
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if hasattr(classifier, 'head'):
                classifier.head.load_state_dict(sd)
                classifier.head.eval()
            return True
        except Exception as e:
            print(f"  ⚠️ 分类器加载失败: {e}")
        return False

    # ── Conductor ──

    def save_conductor(self, conductor: Any):
        """保存 Conductor head"""
        if hasattr(conductor, 'save'):
            conductor.save(str(self.models_dir / "conductor_head.pt"))
            self._kv_set("conductor_path", str(self.models_dir / "conductor_head.pt"))

    def load_conductor(self, conductor: Any, path: Optional[str] = None):
        """加载 Conductor"""
        path = path or str(self.models_dir / "conductor_head.pt")
        if not os.path.exists(path):
            return False
        try:
            if hasattr(conductor, 'load'):
                conductor.load(path)
                return True
        except Exception as e:
            print(f"  ⚠️ Conductor 加载失败: {e}")
        return False

    # ── Experience Buffer ──

    def save_experience(self, buffer: Any):
        """保存经验缓冲"""
        path = str(self.base_dir / "experience.jsonl")
        if hasattr(buffer, 'save'):
            buffer.save(path)
            self._kv_set("experience_path", path)
            self._kv_set("experience_n", getattr(buffer, 'size', 0))

    def load_experience(self, buffer: Any):
        """加载经验缓冲"""
        path = str(self.base_dir / "experience.jsonl")
        if not os.path.exists(path):
            return 0
        try:
            if hasattr(buffer, 'load'):
                buffer.load(path)
                return getattr(buffer, 'size', 0)
        except Exception as e:
            print(f"  ⚠️ 经验缓冲加载失败: {e}")
        return 0

    # ── Episodic Memory ──

    def save_episodic_memory(self, memory: Any):
        """保存情景记忆"""
        if hasattr(memory, 'save'):
            memory.save(str(self.base_dir / "episodic_memory.jsonl"))
            self._kv_set("episodic_memory_n", getattr(memory, 'n_episodes', 0))

    def load_episodic_memory(self, memory: Any):
        """加载情景记忆"""
        path = str(self.base_dir / "episodic_memory.jsonl")
        if not os.path.exists(path):
            return False
        try:
            if hasattr(memory, 'load'):
                memory.load(path)
                return True
        except Exception as e:
            print(f"  ⚠️ 情景记忆加载失败: {e}")
        return False

    # ── Meta Selector ──

    def save_meta_selector(self, meta: Any):
        """保存元选择器状态"""
        # 保存 MLP
        if hasattr(meta, 'mlp') and meta.mlp is not None:
            torch.save(meta.mlp.state_dict(), str(self.models_dir / "meta_mlp.pt"))
        # 保存历史
        history = getattr(meta, 'mode_history', [])
        with open(str(self.base_dir / "mode_history.json"), "w") as f:
            f.write(json.dumps(history[-200:], ensure_ascii=False))

    def load_meta_selector(self, meta: Any):
        """加载元选择器"""
        # MLP
        mlp_path = str(self.models_dir / "meta_mlp.pt")
        if os.path.exists(mlp_path) and hasattr(meta, 'mlp'):
            try:
                meta.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
                meta.mlp_active = True
            except Exception:
                pass
        # 历史
        hist_path = str(self.base_dir / "mode_history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        meta.mode_history = data
            except Exception:
                pass

    # ── Meta Learner ──

    def save_meta_learner(self, meta_learner: Any):
        """保存元学习器"""
        if hasattr(meta_learner, 'save'):
            meta_learner.save()
            self._kv_set("meta_learner_n", len(getattr(meta_learner, 'data', {})))

    def load_meta_learner(self, meta_learner: Any):
        """加载元学习器"""
        if hasattr(meta_learner, 'load'):
            meta_learner.load()

    # ── Run Stats ──

    def save_run_stats(self, run_id: str, stats: dict):
        """保存运行统计"""
        c = self._db_conn.cursor()
        import json as _json
        c.execute(
            """INSERT OR REPLACE INTO run_stats
               (run_id, started_at, finished_at, n_steps, success_rate,
                n_intents_covered, total_reward, fact_graph_nodes,
                fact_graph_edges, schema_coverage, llm_calls,
                llm_success, llm_fallback)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                stats.get("started_at", ""),
                stats.get("finished_at", ""),
                stats.get("n_steps", 0),
                stats.get("success_rate", 0.0),
                stats.get("n_intents_covered", 0),
                stats.get("total_reward", 0.0),
                stats.get("fact_graph_nodes", 0),
                stats.get("fact_graph_edges", 0),
                stats.get("schema_coverage", 0.0),
                stats.get("llm_calls", 0),
                stats.get("llm_success", 0),
                stats.get("llm_fallback", 0),
            ),
        )
        self._db_conn.commit()

    def get_last_run_stats(self) -> Optional[dict]:
        """获取最近一次运行统计"""
        c = self._db_conn.cursor()
        c.execute("SELECT * FROM run_stats ORDER BY rowid DESC LIMIT 1")
        row = c.fetchone()
        if row:
            return dict(row)
        return None

    # ── 全部保存 / 加载 (OnlineAgent 集成) ──

    def save_all(self, agent: Any, run_stats: Optional[dict] = None):
        """一次保存所有状态"""
        with self._lock:
            # FactGraph
            if hasattr(agent, 'workbench') and hasattr(agent.workbench, 'graph'):
                self.save_fact_graph(agent.workbench.graph)

            # Workbench
            if hasattr(agent, 'workbench'):
                self.save_workbench(agent.workbench)

            # WM V3
            if hasattr(agent, 'world_model'):
                self.save_world_model_v3(agent.world_model)

            # WM V4
            if hasattr(agent, 'world_model_v4'):
                self.save_world_model_v4(agent.world_model_v4)

            # Classifier
            if hasattr(agent, 'classifier'):
                self.save_classifier(agent.classifier)

            # Conductor
            cond_active = getattr(agent, 'conductor_path_active', False)
            if hasattr(agent, 'nanny') and cond_active:
                self.save_conductor(agent.nanny.conductor)

            # Experience
            if hasattr(agent, 'buffer'):
                self.save_experience(agent.buffer)

            # Episodic Memory
            if hasattr(agent, 'episodic_memory'):
                self.save_episodic_memory(agent.episodic_memory)

            # Meta Selector
            if hasattr(agent, 'meta_selector'):
                self.save_meta_selector(agent.meta_selector)

            # Meta Learner
            if hasattr(agent, 'meta'):
                self.save_meta_learner(agent.meta)

            # Run stats
            if run_stats:
                self.save_run_stats(run_stats.get("run_id", "unknown"), run_stats)

            print(f"  ✅ PersistentStore: 所有状态已保存到 {self.base_dir}")

    def load_all(self, agent: Any) -> int:
        """加载所有状态, 返回恢复的主要组件数"""
        count = 0
        with self._lock:
            # Workbench (含 FactGraph)
            if hasattr(agent, 'workbench'):
                n = self.load_workbench(agent.workbench)
                if n > 0:
                    count += 1
                # FactGraph (workbench.load 可能已加载, 这里补加载)
                fg = self.load_fact_graph()
                if fg is not None and hasattr(agent.workbench, 'graph'):
                    if len(agent.workbench.graph.nodes) < len(fg.nodes):
                        agent.workbench.graph = fg
                        count += 1

            # WM V3
            if hasattr(agent, 'world_model'):
                if self.load_world_model_v3(agent.world_model):
                    count += 1

            # WM V4
            if hasattr(agent, 'world_model_v4'):
                if self.load_world_model_v4(agent.world_model_v4):
                    count += 1

            # Classifier
            if hasattr(agent, 'classifier'):
                if self.load_classifier(agent.classifier):
                    count += 1

            # Conductor
            cond_active = getattr(agent, 'conductor_path_active', False)
            if hasattr(agent, 'nanny') and cond_active:
                if self.load_conductor(agent.nanny.conductor):
                    count += 1

            # Experience
            if hasattr(agent, 'buffer'):
                n = self.load_experience(agent.buffer)
                if n > 0:
                    count += 1

            # Episodic Memory
            if hasattr(agent, 'episodic_memory'):
                if self.load_episodic_memory(agent.episodic_memory):
                    count += 1

            # Meta Selector
            if hasattr(agent, 'meta_selector'):
                self.load_meta_selector(agent.meta_selector)
                count += 1

            # Meta Learner
            if hasattr(agent, 'meta'):
                self.load_meta_learner(agent.meta)
                count += 1

            print(f"  ✅ PersistentStore: 从 {self.base_dir} 恢复 {count} 个组件")
            return count

    # ── 工具注册表 ──

    def register_tool(self, name: str, tool_type: str, description: str = "",
                      source: str = "factory", created_step: int = 0):
        c = self._db_conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO tools
               (name, tool_type, description, source, created_step)
               VALUES (?, ?, ?, ?, ?)""",
            (name, tool_type, description, source, created_step),
        )
        self._db_conn.commit()

    def log_tool_use(self, name: str, step: int, success: bool, bytes_created: int = 0):
        c = self._db_conn.cursor()
        c.execute(
            """UPDATE tools SET
               last_used_step = MAX(last_used_step, ?),
               use_count = use_count + 1,
               success_rate = (success_rate * use_count + (?)) / (use_count + 1),
               avg_bytes_created = (avg_bytes_created * use_count + ?) / (use_count + 1)
               WHERE name = ?""",
            (step, 1.0 if success else 0.0, bytes_created, name),
        )
        self._db_conn.commit()

    def get_available_tools(self, min_use_count: int = 0) -> list[dict]:
        c = self._db_conn.cursor()
        c.execute(
            "SELECT * FROM tools WHERE use_count >= ? ORDER BY success_rate DESC, use_count DESC",
            (min_use_count,),
        )
        return [dict(row) for row in c.fetchall()]

    # ── 清理 ──

    def close(self):
        if hasattr(self, '_db_conn'):
            self._db_conn.close()
