"""
DoCalculusEngine — 纯 Python 因果推理内核 (Phase 2)

零依赖, ~500 行. 提供:
1. DAG 构建 (从 FactGraph CAUSES/PREDICTS 边)
2. d-separation (BFS on moralized ancestral graph)
3. 后门准则 (confounder 检测)
4. 干预评分 (causal_score, 无需完整图)
5. ATE 估计 (分层调整)
"""
from collections import defaultdict, deque
import itertools
from typing import Optional


class DAG:
    """
    有向无环图 — 邻接表表示

    nodes: set of node keys
    parents: dict[node → set[parent]]
    children: dict[node → set[child]]
    """

    def __init__(self):
        self.nodes: set[str] = set()
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)

    def add_edge(self, u: str, v: str):
        """u → v"""
        self.nodes.add(u)
        self.nodes.add(v)
        self.parents[v].add(u)
        self.children[u].add(v)

    def remove_node(self, n: str):
        """删除节点及关联边"""
        self.nodes.discard(n)
        for p in list(self.parents.get(n, [])):
            self.children[p].discard(n)
        self.parents.pop(n, None)
        for c in list(self.children.get(n, [])):
            self.parents[c].discard(n)
        self.children.pop(n, None)

    def ancestors(self, n: str) -> set[str]:
        """所有祖先 (包含 n 自己)"""
        result = {n}
        queue = deque([n])
        visited = {n}
        while queue:
            node = queue.popleft()
            for p in self.parents.get(node, []):
                if p not in visited:
                    result.add(p)
                    visited.add(p)
                    queue.append(p)
        return result

    def moralize(self) -> 'DAG':
        """Moralize: 添加夫妻边, 去掉方向"""
        m = DAG()
        for n in self.nodes:
            m.nodes.add(n)
        for u in self.nodes:
            for v in self.children.get(u, []):
                m.add_edge(u, v)
                m.add_edge(v, u)
        # 夫妻边: 共享子节点的父节点之间加无向边
        for v in self.nodes:
            ps = self.parents.get(v, [])
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    m.add_edge(ps[i], ps[j])
                    m.add_edge(ps[j], ps[i])
        return m

    def d_separated(self, x: set[str], y: set[str],
                    z: set[str]) -> bool:
        """
        d-separation: X ⟂ Y | Z ?

        Pearl's definition: X and Y are d-separated by Z if
        there is no active path between X and Y given Z.
        Active path: colliders (→c←) or descendants of colliders
        """ # docstring continues (kept from above)
        # Step 1: Ancestral graph (without Unicode chars in comments)
        # Step 1: Ancestral graph
        ancestry = set()
        for n in itertools.chain(x, y, z):
            ancestry |= self.ancestors(n)

        # Step 2: Moralize ancestral graph
        moral = DAG()
        for n in ancestry:
            moral.nodes.add(n)
        for u in ancestry:
            for v in self.children.get(u, []):
                if v in ancestry:
                    moral.add_edge(u, v)
                    moral.add_edge(v, u)
        for v in ancestry:
            ps = list(self.parents.get(v, []))
            ps = [p for p in ps if p in ancestry]
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    moral.add_edge(ps[i], ps[j])
                    moral.add_edge(ps[j], ps[i])

        # Step 3: Remove Z
        for n in z:
            moral.remove_node(n)

        # Step 4: BFS
        visited = set(x)
        queue = deque(x)
        while queue:
            node = queue.popleft()
            if node in y:
                return False
            for neighbor in moral.children.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return True

    def minimal_dseparator(self, x: str, y: str) -> Optional[set[str]]:
        """找到最小的 d-separator (如果有)"""
        others = self.nodes - {x, y}
        for k in range(len(others) + 1):
            for candidate in itertools.combinations(others, k):
                if self.d_separated({x}, {y}, set(candidate)):
                    return set(candidate)
        return None


class DoCalculusEngine:
    """
    因果推理引擎

    用法:
        engine = DoCalculusEngine()
        engine.add_edge("cpu_cores", "load", "causes")
        score = engine.causal_score("cpu_cores", "load", transitions)
        backdoor = engine.find_backdoor_set("cpu_cores", "load")
    """

    def __init__(self):
        self.graph = DAG()
        self.edge_types: dict[tuple[str, str], str] = {}  # (u,v) → type

    def add_edge(self, u: str, v: str, edge_type: str = "correlates"):
        """从 FactGraph 添加边"""
        self.graph.add_edge(u, v)
        self.edge_types[(u, v)] = edge_type

    def add_from_factgraph(self, factgraph) -> int:
        """从 FactGraph 对象批量导入边"""
        count = 0
        if not hasattr(factgraph, 'edges'):
            return 0
        for src, edge_list in factgraph.edges.items():
            for e in edge_list:
                dst = e.get("to", "")
                rel = e.get("rel", "correlates")
                if src in factgraph.nodes and dst in factgraph.nodes:
                    # 只导入 causal/predicts 边 (不导入 correlates)
                    if rel in ("causes", "predicts", "inhibits"):
                        self.add_edge(src, dst, rel)
                        count += 1
        return count

    # ── 后门准则 ──

    def find_backdoor_set(self, treatment: str,
                          outcome: str) -> Optional[set[str]]:
        """
        找到后门调整集 Z:
        - Z 不能是 treatment 的后代
        - Z d-separates X and Y in the back-door graph
          (去除从 X 出发的出边后的图)
        """
        if treatment not in self.graph.nodes:
            return None
        if outcome not in self.graph.nodes:
            return None

        # 构建后门图: 去掉从 treatment 出发的出边
        bd = DAG()
        for n in self.graph.nodes:
            bd.nodes.add(n)
        for u in self.graph.nodes:
            for v in self.graph.children.get(u, []):
                if u != treatment:  # 去掉 treatment → *
                    bd.add_edge(u, v)

        # 候选: 非 outcome, 非 treatment 后代的节点
        treatment_desc = self._descendants(treatment)
        candidates = self.graph.nodes - {treatment, outcome} - treatment_desc

        # 找最小的 d-separator
        for k in range(len(candidates) + 1):
            for z in itertools.combinations(candidates, k):
                z_set = set(z)
                if bd.d_separated({treatment}, {outcome}, z_set):
                    return z_set
        return None

    # ── 前门准则 ──

    def find_frontdoor_set(self, treatment: str,
                           outcome: str) -> Optional[set[str]]:
        """
        找到前门调整集 M (mediator):
        - M 拦截所有 treatment→outcome 的 directed paths
        - 没有从 treatment 到 M 的后门路径
        - M→outcome 的所有后门路径被 treatment blocked
        """
        if treatment not in self.graph.nodes:
            return None
        if outcome not in self.graph.nodes:
            return None

        candidates = self.graph.nodes - {treatment, outcome}
        for k in range(1, len(candidates) + 1):
            for m in itertools.combinations(candidates, k):
                m_set = set(m)
                # 条件 1: M 拦截所有 directed paths
                if not self._intercepts_all_directed(treatment, outcome, m_set):
                    continue
                # 条件 2: no backdoor from treatment to M
                bd_tm = find_backdoor_manual(self.graph, treatment, m_set)
                if bd_tm is not None:
                    continue
                # 条件 3: all backdoor M→O blocked by treatment
                bd_mo = find_backdoor_manual(self.graph, m_set, outcome, {treatment})
                if bd_mo is not None:
                    continue
                return m_set
        return None

    # ── 干预评分 (LITE 版, 无需图) ──

    def causal_score(self, a_var: str, b_var: str,
                     transitions: list[dict]) -> float:
        """
        干预因果评分: P(B|do(A)) - P(B|do(∅))

        只统计有明确干预证据的 transition pairs.
        不需要完整因果图.
        """
        # 提取 A 变化 → B 跟随的数据
        a_changed = []
        b_given_a_change = []
        b_given_a_stable = []

        for i in range(1, len(transitions)):
            prev = transitions[i - 1]
            curr = transitions[i]

            a_prev = prev.get("pre_state", {}).get(a_var)
            a_curr = curr.get("pre_state", {}).get(a_var)

            b_prev = prev.get("post_state", {}).get(b_var)
            b_curr = curr.get("post_state", {}).get(b_var)

            if a_prev is not None and a_curr is not None and \
               b_prev is not None and b_curr is not None:
                if a_prev != a_curr:
                    a_changed.append(True)
                    b_given_a_change.append(b_prev != b_curr)
                else:
                    a_changed.append(False)
                    b_given_a_stable.append(b_prev != b_curr)

        if len(b_given_a_change) < 3:
            return 0.0  # 证据不足

        p_b_change_given_a_change = sum(b_given_a_change) / len(b_given_a_change)
        p_b_change_given_a_stable = (
            sum(b_given_a_stable) / len(b_given_a_stable)
            if b_given_a_stable else 0.0
        )

        return p_b_change_given_a_change - p_b_change_given_a_stable

    # ── ATE 估计 ──

    def estimate_ate(self, treatment: str, outcome: str,
                     transitions: list[dict],
                     adjustment_set: Optional[set[str]] = None) -> float:
        """
        ATE = E[Y|do(X=1)] - E[Y|do(X=0)]

        通过后门调整集 Z 分层:
        ATE = Σ_Z (E[Y|X=1,Z] - E[Y|X=0,Z]) × P(Z)

        无调整集 → 简单差分
        """
        if not adjustment_set:
            # 简单差分 (有偏)
            x1_y = []
            x0_y = []
            for t in transitions:
                val = t.get("pre_state", {}).get(treatment)
                y_val = t.get("post_state", {}).get(outcome)
                if val is not None and y_val is not None:
                    if val:
                        x1_y.append(y_val)
                    else:
                        x0_y.append(y_val)
            ate = (sum(x1_y) / max(len(x1_y), 1)) - \
                  (sum(x0_y) / max(len(x0_y), 1))
            return ate if isinstance(ate, (int, float)) else 0.0

        # 分层调整
        strata: dict[str, list[dict]] = defaultdict(list)
        for t in transitions:
            z_key = tuple(
                str(t.get("pre_state", {}).get(z, ""))
                for z in sorted(adjustment_set)
            )
            strata[z_key].append(t)

        ate = 0.0
        total_w = 0.0
        for z_key, z_trans in strata.items():
            w = len(z_trans) / max(len(transitions), 1)
            x1, x0 = [], []
            for t in z_trans:
                val = t.get("pre_state", {}).get(treatment)
                y_val = t.get("post_state", {}).get(outcome)
                if val is not None and y_val is not None:
                    if val:
                        x1.append(y_val)
                    else:
                        x0.append(y_val)
            diff = (sum(x1) / max(len(x1), 1)) - \
                   (sum(x0) / max(len(x0), 1))
            ate += w * diff
            total_w += w

        return ate / max(total_w, 0.001)

    # ── 工具 ──

    def _descendants(self, n: str) -> set[str]:
        """所有后代"""
        result = set()
        queue = deque([n])
        while queue:
            node = queue.popleft()
            for c in self.graph.children.get(node, []):
                if c not in result:
                    result.add(c)
                    queue.append(c)
        return result

    def _intercepts_all_directed(self, src: str, dst: str,
                                  mediators: set[str]) -> bool:
        """M 是否拦截所有 src→dst 的有向路径"""
        # 找到所有 src→dst 的有向路径, 检查是否都经过 M
        all_paths = self._all_directed_paths(src, dst)
        if not all_paths:
            return False
        for path in all_paths:
            if not any(n in mediators for n in path[1:-1]):
                return False
        return True

    def _all_directed_paths(self, src: str, dst: str,
                            max_depth: int = 10):
        """找到所有有向路径 (DFS, 限制深度)"""
        paths = []

        def dfs(current, path, depth):
            if depth > max_depth:
                return
            if current == dst:
                paths.append(path + [current])
                return
            for c in self.graph.children.get(current, []):
                if c not in path:
                    dfs(c, path + [current], depth + 1)

        dfs(src, [], 0)
        return paths


def find_backdoor_manual(graph: DAG, x: str | set[str],
                         y: set[str],
                         given: Optional[set[str]] = None) -> Optional[set[str]]:
    """手动找后门调整集 (工具函数)"""
    g = DAG()
    for n in graph.nodes:
        g.nodes.add(n)
    for u in graph.nodes:
        for v in graph.children.get(u, []):
            if u not in (x if isinstance(x, set) else {x}):
                g.add_edge(u, v)

    x_set = {x} if isinstance(x, str) else x
    z = given or set()
    candidates = graph.nodes - x_set - y - z

    for k in range(len(candidates) + 1):
        for z_add in itertools.combinations(candidates, k):
            z_full = z | set(z_add)
            if g.d_separated(x_set, y, z_full):
                return z_full
    return None
