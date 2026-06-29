"""
TransitionMiner — P16 R2: 离线因果挖掘

从 transition 记录中发现因果模式:
  每对 (A, B) 事实, 算 P(B_change) 和 P(B_change|A_change)
  gain > 阈值 → 候选因果边

用法:
  miner = TransitionMiner()
  edges = miner.mine("data/persistent/transitions.jsonl", window=500)
"""
import json
import os
from collections import defaultdict
from typing import Optional

from agent.fact_graph import (
    FactGraph, EDGE_CAUSES, EDGE_INHIBITS, EDGE_CORRELATES,
)


class TransitionMiner:
    """从 transition JSONL 中挖掘候选因果边"""

    def __init__(self, gain_threshold_cause: float = 0.3,
                 gain_threshold_inhibit: float = -0.3,
                 gain_threshold_correlate: float = 0.1,
                 max_edges_per_batch: int = 50):
        self.gain_threshold_cause = gain_threshold_cause
        self.gain_threshold_inhibit = gain_threshold_inhibit
        self.gain_threshold_correlate = gain_threshold_correlate
        self.max_edges = max_edges_per_batch

    def load_transitions(self, path: str, window: int = 500) -> list[dict]:
        """从 JSONL 加载 transition, 只保留最近 window 条"""
        if not os.path.exists(path):
            return []
        with open(path) as f:
            lines = f.readlines()
        recent = lines[-window:]
        transitions = []
        for line in recent:
            line = line.strip()
            if line:
                try:
                    transitions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return transitions

    def _detect_changes(self, tx: dict) -> set[str]:
        """返回 transition 中值发生变化的节点 key 集合"""
        pre = tx.get("pre_state", {})
        post = tx.get("post_state", {})
        changed = set()
        all_keys = set(pre.keys()) | set(post.keys())
        for k in all_keys:
            old_val = pre.get(k, "")
            new_val = post.get(k, "")
            if old_val != new_val:
                changed.add(k)
        return changed

    def mine(self, path: str = "data/persistent/transitions.jsonl",
             graph: Optional[FactGraph] = None,
             window: int = 500) -> list[dict]:
        """
        挖掘因果边。

        返回: [{src, dst, rel, weight, step, ...}] 按 weight 降序
        """
        transitions = self.load_transitions(path, window)
        if len(transitions) < 10:
            return []

        # 提取每步的变化集
        step_changes: list[tuple[int, set[str]]] = []
        for tx in transitions:
            step = tx.get("step", 0)
            changed = self._detect_changes(tx)
            if changed:
                step_changes.append((step, changed))

        if len(step_changes) < 5:
            return []  # 变化太少, 无统计意义

        # 收集所有出现过的节点
        all_nodes: set[str] = set()
        for _, changes in step_changes:
            all_nodes.update(changes)

        n_obs = len(step_changes)

        # 计算 P(B_change) 和 P(B_change | A_change & A在前)
        # 对: A change 在先 → B change (同一步或下一步)
        p_b = defaultdict(float)      # P(B变化)
        p_b_given_a = defaultdict(float)  # P(B变化 | A变化)
        count_b = defaultdict(int)
        count_a = defaultdict(int)
        count_ab = defaultdict(int)   # A变化且B随后变化

        for _, changes in step_changes:
            for node in changes:
                count_b[node] += 1
        for node in all_nodes:
            p_b[node] = count_b[node] / max(n_obs, 1)

        # 滑动窗口统计共现
        for i in range(len(step_changes) - 1):
            _, changes_i = step_changes[i]
            _, changes_next = step_changes[i + 1]
            for a_node in changes_i:
                count_a[a_node] += 1
                for b_node in changes_next:
                    if a_node != b_node:
                        count_ab[(a_node, b_node)] += 1

        for (a_node, b_node), cnt in count_ab.items():
            p_b_given_a[(a_node, b_node)] = cnt / max(count_a.get(a_node, 1), 1)

        # 生成候选边
        candidates = []
        for (a_node, b_node), p_ba in p_b_given_a.items():
            p_b_val = p_b.get(b_node, 0)
            gain = p_ba - p_b_val

            if gain > self.gain_threshold_cause:
                rel = EDGE_CAUSES
                weight = gain
            elif gain < self.gain_threshold_inhibit:
                rel = EDGE_INHIBITS
                weight = -gain
            elif abs(gain) > self.gain_threshold_correlate:
                rel = EDGE_CORRELATES
                weight = abs(gain)
            else:
                continue

            # 跨类别加分 (如果 graph 可用)
            cross_bonus = 0.0
            if graph:
                cat_a = graph.nodes[a_node].category if a_node in graph.nodes else "general"
                cat_b = graph.nodes[b_node].category if b_node in graph.nodes else "general"
                if cat_a != cat_b:
                    cross_bonus = 0.1

            candidates.append({
                "src": a_node,
                "dst": b_node,
                "rel": rel,
                "weight": round(weight + cross_bonus, 4),
                "gain": round(gain, 4),
                "p_b_given_a": round(p_ba, 4),
                "p_b": round(p_b_val, 4),
                "n_obs": count_ab[(a_node, b_node)],
                "cross_category": cross_bonus > 0,
            })

        # 按 weight 降序, 限 max_edges
        candidates.sort(key=lambda e: -e["weight"])
        return candidates[:self.max_edges]

    def apply_to_graph(self, candidates: list[dict], graph: FactGraph,
                       step: int = 0):
        """将候选边写入 FactGraph (不覆盖现有高权重边)"""
        for c in candidates:
            if c["weight"] < 0.05:
                continue
            # 检查是否已有更强/相同边
            existing = graph.get_edges(c["src"], c["rel"])
            already = any(e["to"] == c["dst"] for e in existing)
            if already:
                for e in existing:
                    if e["to"] == c["dst"]:
                        e["n_support"] = e.get("n_support", 0) + 1
                        e["hypothesis_key"] = "transition_miner"
                        if c["weight"] > e["weight"]:
                            e["weight"] = c["weight"]
                        break
            else:
                graph.add_edge(
                    c["src"], c["dst"], c["rel"],
                    weight=c["weight"], step=step,
                    n_support=1, n_against=0,
                    hypothesis_key="transition_miner",
                )
