"""
HypothesisEngine — P16 R2: 假设生成

从 TransitionMiner 候选边 + FactGraph 不确定性, 生成可验证假设。
按 priority = uncertainty(A) * uncertainty(B) * testability * (1 + |corr|) 排序。

用法:
  engine = HypothesisEngine()
  hypotheses = engine.generate(candidates, graph)
"""
from agent.fact_graph import (
    FactGraph, EDGE_CAUSES, EDGE_CORRELATES, EDGE_INHIBITS,
)


class HypothesisEngine:
    """从候选因果边生成可验证假设"""

    def __init__(self, top_k: int = 5, min_priority: float = 0.1):
        self.top_k = top_k
        self.min_priority = min_priority
        self._generated_keys: set[str] = set()

    def _node_uncertainty(self, graph: FactGraph, key: str) -> float:
        """节点不确定度: 1 - confidence * evidence_factor"""
        node = graph.nodes.get(key)
        if not node:
            return 0.8
        evidence_factor = min(1.0, node.n_evidence / 10.0) if hasattr(node, 'n_evidence') else 0.5
        return max(0.1, 1.0 - node.confidence * evidence_factor)

    def _testability(self, src: str, dst: str, rel: str) -> float:
        """可测试性评分: CAUSES 需干预(0.6), CORRELATES 被动观察(0.9), INHIBITS(0.5)"""
        base = {
            EDGE_CAUSES: 0.6,
            EDGE_CORRELATES: 0.9,
            EDGE_INHIBITS: 0.5,
        }.get(rel, 0.5)
        easy_to_read = {"os_name", "kernel", "architecture", "cpu_cores",
                        "mem_total", "hostname", "current_user", "disk_root"}
        if dst in easy_to_read:
            base = min(1.0, base + 0.15)
        return base

    def _build_prediction(self, src: str, dst: str, rel: str) -> str:
        """生成假设描述"""
        if rel == EDGE_CAUSES:
            return f"{src} changes → {dst} tends to change (causal)"
        elif rel == EDGE_INHIBITS:
            return f"{src} increases → {dst} tends to stay/decrease (inhibits)"
        elif rel == EDGE_CORRELATES:
            return f"{src} and {dst} co-change (correlates)"
        return f"{src} → {dst} ({rel})"

    def generate(self, candidates: list[dict], graph: FactGraph) -> list[dict]:
        """
        从候选边生成假设。

        Args:
          candidates: TransitionMiner 的候选边列表
          graph: FactGraph

        Returns:
          [{if_node, rel, then_node, prediction, uncertainty, testability, priority, ...}]
        """
        hypotheses = []

        for c in candidates:
            src = c["src"]
            dst = c["dst"]
            rel = c["rel"]
            corr = abs(c.get("gain", c.get("weight", 0)))

            key = f"{src}:{rel}:{dst}"
            if key in self._generated_keys:
                continue

            unc_a = self._node_uncertainty(graph, src)
            unc_b = self._node_uncertainty(graph, dst)
            test = self._testability(src, dst, rel)
            priority = unc_a * unc_b * test * (1 + corr)

            if priority < self.min_priority:
                continue

            hypotheses.append({
                "if_node": src,
                "rel": rel,
                "then_node": dst,
                "prediction": self._build_prediction(src, dst, rel),
                "uncertainty": round(unc_a * unc_b, 3),
                "testability": round(test, 2),
                "priority": round(priority, 4),
                "n_obs": c.get("n_obs", 0),
                "cross_category": c.get("cross_category", False),
                "_key": key,
            })

        hypotheses.sort(key=lambda h: -h["priority"])
        result = hypotheses[:self.top_k]
        for h in result:
            self._generated_keys.add(h["_key"])
            del h["_key"]
        return result

    def reset(self):
        """清空已生成记录"""
        self._generated_keys.clear()
