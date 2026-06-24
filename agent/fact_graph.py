"""
FactGraph — 动态事实图

替代 Workbench 的扁平 dict, 用图结构存储事实。
节点 = 事实, 边 = 关系, schema = 类别模板

设计原则:
  - 纯 dict + 邻接表, 无外部依赖 (不用 NetworkX/PyG)
  - JSON 序列化友好
  - 自动缺口检测 (gaps) 驱动探索
  - O(1) 节点查询, O(degree) 边查询
"""

import json
from typing import Optional


# ── 边类型 ──
EDGE_REQUIRES = "requires"       # A 缺少 B 时应补充 (os_name → os_version_id)
EDGE_VERIFIES = "verifies"       # A 验证 B (hostname → hostname_cmd)
EDGE_EXTENDS = "extends"         # A 深化 B (cpu_cores → cpu_model)
EDGE_LOCATED_IN = "located_in"   # 文件在目录下
EDGE_DERIVED_FROM = "derived"    # 事实从某命令输出推导
EDGE_CONFLICTS = "conflicts"     # A 与 B 矛盾
EDGE_SAME_AS = "same_as"         # A 等价 B


class Node:
    __slots__ = ("value", "category", "confidence", "step", "source_cmd", "count")
    
    def __init__(self, value: str, category: str = "general",
                 confidence: float = 1.0, step: int = 0,
                 source_cmd: str = ""):
        self.value = value
        self.category = category
        self.confidence = confidence
        self.step = step
        self.source_cmd = source_cmd[:50]
        self.count = 1

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "category": self.category,
            "confidence": self.confidence,
            "step": self.step,
            "source_cmd": self.source_cmd,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        n = cls(d["value"], d.get("category", "general"),
                d.get("confidence", 1.0), d.get("step", 0),
                d.get("source_cmd", ""))
        n.count = d.get("count", 1)
        return n


class FactGraph:
    """
    事实图: 节点 + 边 + schema

    用法:
      g = FactGraph()
      g.add_node("os_name", "Debian", category="system", step=1)
      g.add_edge("os_name", "os_version_id", "requires")
      gaps = g.find_gaps()  # → [("os_name", "os_version_id", "requires")]
    """

    def __init__(self, max_nodes: int = 200):
        self.nodes: dict[str, Node] = {}       # key → Node
        self.edges: dict[str, list[dict]] = {}  # key → [{to, rel, weight, step}]
        self.max_nodes = max_nodes
        # schema: category → [required_keys...]
        self.schemas: dict[str, list[str]] = {
            "system": [
                "os_name", "os_version_id", "kernel", "architecture",
                "cpu_cores", "mem_total", "current_user",
            ],
            "network": [
                "ip_addr", "etchosts_hosts",
            ],
            "explore": [],
            "general": [],
        }
        self._current_discovery: Optional[str] = None

    # ── 节点操作 ──

    def add_node(self, key: str, value: str, category: str = "general",
                 confidence: float = 1.0, step: int = 0,
                 source_cmd: str = "") -> bool:
        """添加或更新节点, 返回是否为新节点"""
        if not self._is_valid_value(value):
            return False

        is_new = key not in self.nodes
        if is_new:
            if len(self.nodes) >= self.max_nodes:
                # LRU 淘汰: 移除最旧节点
                oldest = min(self.nodes, key=lambda k: self.nodes[k].step)
                self._remove_node(oldest)
            self.nodes[key] = Node(value, category, confidence, step, source_cmd)
            self._current_discovery = key
            # 自动建边: 根据 schema 关系
            self._auto_link(key, category, step)
        else:
            old = self.nodes[key]
            old.value = value[:100]
            old.confidence = min(old.confidence + 0.15, 1.0)
            old.step = step
            old.count += 1
            old.category = category
            if source_cmd:
                old.source_cmd = source_cmd[:50]

        return is_new

    def get_node(self, key: str) -> Optional[Node]:
        return self.nodes.get(key)

    def get_value(self, key: str) -> Optional[str]:
        n = self.nodes.get(key)
        return n.value if n else None

    def node_count(self) -> int:
        return len(self.nodes)

    def all_keys(self) -> list[str]:
        return list(self.nodes.keys())

    def get_nodes_by_category(self, category: str) -> list[str]:
        return [k for k, n in self.nodes.items() if n.category == category]

    def categories(self) -> set[str]:
        return {n.category for n in self.nodes.values()}

    # ── 边操作 ──

    def add_edge(self, src: str, dst: str, rel: str, weight: float = 1.0,
                 step: int = 0):
        """添加一条边 src --rel--> dst"""
        if src not in self.nodes or dst not in self.nodes:
            return
        if src not in self.edges:
            self.edges[src] = []
        # 去重: 同 dst+rel 不重复加
        for e in self.edges[src]:
            if e["to"] == dst and e["rel"] == rel:
                e["weight"] = max(e["weight"], weight)
                e["step"] = step
                return
        self.edges[src].append({"to": dst, "rel": rel, "weight": weight, "step": step})

    def get_edges(self, key: str, rel: Optional[str] = None) -> list[dict]:
        """获取节点的出边, 可选按关系过滤"""
        edges = self.edges.get(key, [])
        if rel:
            return [e for e in edges if e["rel"] == rel]
        return edges

    def has_edge(self, src: str, dst: str, rel: str) -> bool:
        """检查边是否存在"""
        for e in self.edges.get(src, []):
            if e["to"] == dst and e["rel"] == rel:
                return True
        return False

    # ── 缺口检测 ──

    def find_gaps(self) -> list[tuple[str, str, str]]:
        """
        发现事实缺口: (from_key, missing_key, relation)

        逻辑:
          1. schema 缺口: 同一 category 里缺了 required key
          2. 边缺口: 有 requires 边但目标节点不存在
          3. 关联缺口: 成对事实缺一个
        """
        gaps = []

        # 1. Schema 缺口
        cat_nodes: dict[str, list[str]] = {}
        for k, n in self.nodes.items():
            cat_nodes.setdefault(n.category, []).append(k)

        for cat, required in self.schemas.items():
            if not required:
                continue
            present = set(cat_nodes.get(cat, []))
            for req_key in required:
                if req_key not in present:
                    # 找同类节点作为 from
                    candidates = [k for k in present if k != req_key]
                    if candidates:
                        gaps.append((candidates[0], req_key, "schema"))
                    else:
                        gaps.append(("(root)", req_key, "schema"))

        # 2. requires 边缺口
        for src in self.edges:
            for e in self.edges[src]:
                if e["rel"] == EDGE_REQUIRES and e["to"] not in self.nodes:
                    gaps.append((src, e["to"], EDGE_REQUIRES))

        return gaps

    def _compute_schema_coverage(self) -> float:
        """计算 schema 覆盖度 (0~1)"""
        total = 0
        covered = 0
        for cat, required in self.schemas.items():
            if not required:
                continue
            present = {k for k, n in self.nodes.items() if n.category == cat}
            for req in required:
                total += 1
                if req in present:
                    covered += 1
        return covered / max(total, 1)

    # ── 内部辅助 ──

    def _remove_node(self, key: str):
        """移除一个节点及其所有相关边"""
        self.nodes.pop(key, None)
        self.edges.pop(key, None)
        # 移除其他节点指向该 key 的边
        for src in list(self.edges.keys()):
            self.edges[src] = [e for e in self.edges[src] if e["to"] != key]

    def _auto_link(self, key: str, category: str, step: int):
        """新节点自动建立 schema 预设边"""
        pairs = [
            ("os_name", "os_version_id", EDGE_REQUIRES),
            ("os_version_id", "os_version_codename", EDGE_EXTENDS),
            ("kernel", "kernel_release", EDGE_EXTENDS),
            ("cpu_cores", "cpu_model", EDGE_EXTENDS),
            ("mem_total", "swap_total", EDGE_REQUIRES),
            ("hostname", "etchosts_hosts", EDGE_VERIFIES),
            ("hostname", "hostname_cmd", EDGE_VERIFIES),
            ("current_user", "uid_info", EDGE_EXTENDS),
            ("users", "current_user", EDGE_REQUIRES),
            ("ip_addr", "mac_addr", EDGE_EXTENDS),
            ("disk_root", "disk_persistent", EDGE_EXTENDS),
        ]
        for a, b, rel in pairs:
            if key == a and b in self.nodes:
                self.add_edge(a, b, rel, step=step)
            elif key == b and a in self.nodes:
                self.add_edge(a, b, rel, step=step)

        # 类别链路: 同一 category 的节点自动 requires 连接
        cat_required = self.schemas.get(category, [])
        if key in cat_required:
            idx = cat_required.index(key)
            if idx > 0:
                prev = cat_required[idx - 1]
                if prev in self.nodes:
                    self.add_edge(prev, key, EDGE_REQUIRES, step=step)

    def get_current_discovery(self) -> Optional[str]:
        """返回最新添加的关键事实 key"""
        return self._current_discovery

    def _is_valid_value(self, val: str) -> bool:
        if not val or len(val) < 1:
            return False
        stripped = val.strip("=._-:;~!@#$%^&*()[]{}\"'")
        if not stripped:
            return False
        if val.startswith(("=", "_", ".", ")", "(")):
            return False
        letter_count = sum(c.isalnum() for c in val)
        if letter_count < 2:
            return False
        return True

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "nodes": {k: n.to_dict() for k, n in self.nodes.items()},
            "edges": self.edges,
            "schemas": self.schemas,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactGraph":
        g = cls()
        for k, nd in d.get("nodes", {}).items():
            g.nodes[k] = Node.from_dict(nd)
        g.edges = d.get("edges", {})
        g.schemas = d.get("schemas", g.schemas)
        return g

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "FactGraph":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ── 统计 ──

    def stats(self) -> dict:
        """统计信息"""
        cat_dist = {}
        for n in self.nodes.values():
            cat_dist[n.category] = cat_dist.get(n.category, 0) + 1
        return {
            "n_nodes": len(self.nodes),
            "n_edges": sum(len(v) for v in self.edges.values()),
            "categories": cat_dist,
            "schema_coverage": self._compute_schema_coverage(),
            "n_gaps": len(self.find_gaps()),
        }

    def get_state_summary(self, max_keys: int = 5) -> str:
        """生成 state_text 用的事实摘要"""
        if not self.nodes:
            return "无"
        recent = sorted(self.nodes.values(), key=lambda n: -n.step)[:max_keys]
        # 找最新的 keys
        keys_by_step = sorted(self.nodes.keys(),
                              key=lambda k: self.nodes[k].step, reverse=True)
        top = keys_by_step[:max_keys]
        parts = []
        for k in top:
            n = self.nodes[k]
            parts.append(f"{k}={n.value[:15]}")
        return ", ".join(parts)

    # ── 事实历史追踪 (从 Workbench 迁移) ──

    def get_fact_history(self) -> dict[str, list[tuple[int, str]]]:
        """获取事实值变化历史"""
        if not hasattr(self, "_fact_history"):
            self._fact_history: dict[str, list[tuple[int, str]]] = {}
        return self._fact_history

    def track_fact_history(self):
        """追踪事实值变化"""
        history = self.get_fact_history()
        for k, n in self.nodes.items():
            val = n.value
            hist = history.setdefault(k, [])
            if not hist or hist[-1][1] != val:
                hist.append((n.step, val))
                if len(hist) > 10:
                    hist.pop(0)

    def build_change_report(self) -> str:
        """检测事实变化, 生成报告"""
        history = self.get_fact_history()
        changes = []
        for k, hist in history.items():
            if len(hist) >= 2:
                old_step, old_val = hist[-2]
                new_step, new_val = hist[-1]
                if old_val != new_val:
                    changes.append(
                        f"- {k}: {old_val} (step {old_step}) → {new_val} (step {new_step})"
                    )
        if changes:
            return "\n".join(["## Changes Detected"] + changes[-5:])
        return ""

    def build_cross_analysis(self) -> str:
        """跨事实推理"""
        def v(k):
            n = self.nodes.get(k)
            return n.value if n else None

        inferences = []

        # 1. 环境推断
        kernel = v("kernel")
        os_name = v("os_name")
        os_ver = v("os_version_id")
        arch = v("architecture")
        if kernel and os_name:
            if "debian" in (os_name or "").lower():
                inferences.append(
                    f"* Environment: Debian {os_ver or ''} container on Linux "
                    f"{kernel[:15]} — typical sandbox setup"
                )

        # 2. 资源画像
        cpu_cores = v("cpu_cores")
        mem = v("mem_total")
        if cpu_cores:
            cores = int(cpu_cores) if cpu_cores.isdigit() else 0
            if cores >= 16:
                inferences.append(f"* Workload: {cpu_cores}-core CPU — build server")
            elif cores >= 4:
                inferences.append(f"* Workload: {cpu_cores}-core CPU — general purpose")
            else:
                inferences.append(f"* Workload: {cpu_cores}-core CPU — lightweight")

        # 3. 存储分析
        disk_root = v("disk_root")
        disk_persistent = v("disk_persistent")
        if disk_persistent and disk_root:
            try:
                root_gb = int(disk_root.rstrip("G").rstrip("g"))
                persist_gb = int(disk_persistent.rstrip("G").rstrip("g"))
                if persist_gb > root_gb:
                    inferences.append(
                        f"* Storage: persistent ({disk_persistent}) > root "
                        f"({disk_root}) → external volume mount"
                    )
            except ValueError:
                pass

        # 4. 网络
        hostname_val = v("hostname") or v("node_name")
        if hostname_val and len(hostname_val) == 12 and \
           all(c in "0123456789abcdef" for c in hostname_val.lower()):
            inferences.append(
                f"* Network: hostname '{hostname_val}' is Docker ID → --network none"
            )

        # 5. 安全
        users = v("users")
        if users and "root" in users:
            inferences.append("* Security: single root user — standard container")

        # 6. 覆盖统计
        n_cats = len(self.categories())
        n_facts = len(self.nodes)
        if n_facts >= 10 and n_cats >= 3:
            inferences.append(
                f"* Discovery: {n_facts} facts across {n_cats} categories — good"
            )

        if not inferences:
            return ""
        return "## Inference\n" + "\n".join(inferences)
