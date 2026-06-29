"""
Verdict — P16 R3: 实验验证

比较预测 vs 实际, 更新信念边。
"""
from agent.fact_graph import FactGraph


class Verdict:
    """实验验证器: 评估结果, 更新 FactGraph 边"""

    def __init__(self, lr: float = 0.3):
        self.lr = lr
        self.history: list[dict] = []

    def evaluate(self, plan: dict, result: dict,
                 graph: FactGraph) -> dict:
        """
        评估实验结果, 更新边。

        Args:
          plan: ExperimentPlanner 生成的计划
          result: execute_plan 返回的结果
          graph: FactGraph

        Returns:
          {verdict, support_delta, n_support_added, n_against_added, edge_removed}
        """
        verdict = {
            "verdict": "unknown",
            "score": 0.0,
            "n_support": 0,
            "n_against": 0,
            "edge_removed": False,
            "details": [],
        }

        hypothesis_key = plan.get("hypothesis_key", "")
        if not hypothesis_key or not graph:
            return verdict

        # 解析 hypothesis_key → src, rel, dst
        parts = hypothesis_key.split(":", 2)
        if len(parts) != 3:
            return verdict
        src, rel, dst = parts

        # 找对应的边
        target_edge = None
        for e in graph.get_edges(src, rel):
            if e["to"] == dst:
                target_edge = e
                break

        if target_edge is None:
            return verdict

        # 预测 vs 实际
        pred_exit = plan.get("predicted_exit", 0)
        pred_len = plan.get("predicted_output_len", 50)
        actual_exit = result.get("exit_code", -1)
        actual_len = result.get("output_len", 0)
        actual_success = result.get("success", False)

        score = 0.0
        n_sup = 0
        n_ag = 0

        # 1. Exit code 匹配
        if actual_exit == pred_exit:
            score += 0.5
            n_sup += 1
            verdict["details"].append(f"exit match: pred={pred_exit}, actual={actual_exit} (+0.5)")
        else:
            score -= 0.3
            n_ag += 1
            verdict["details"].append(f"exit mismatch: pred={pred_exit}, actual={actual_exit} (-0.3)")

        # 2. Output length 近似 (在 2x 范围内)
        if pred_len > 0 and actual_len > 0:
            ratio = max(actual_len, pred_len) / max(min(actual_len, pred_len), 1)
            if ratio < 2.0:
                score += 0.3
                n_sup += 1
                verdict["details"].append(f"len match: pred≈{pred_len}, actual={actual_len} (+0.3)")
            else:
                score -= 0.2
                n_ag += 1
                verdict["details"].append(f"len mismatch: pred={pred_len}, actual={actual_len} (-0.2)")

        # 3. 执行成功
        if actual_success:
            score += 0.2
            n_sup += 1
            verdict["details"].append(f"successful execution (+0.2)")

        # 更新边
        old_weight = target_edge.get("weight", 0)
        if score >= 0:
            # 支持: weight += lr * (1 - weight)
            new_weight = old_weight + self.lr * (1 - old_weight)
        else:
            # 反驳: weight -= lr * (1 + weight)
            new_weight = old_weight - self.lr * (1 + old_weight)
            # 下限保护
            new_weight = max(-1.0, new_weight)

        target_edge["weight"] = round(new_weight, 4)
        target_edge["n_support"] = target_edge.get("n_support", 0) + n_sup
        target_edge["n_against"] = target_edge.get("n_against", 0) + n_ag
        target_edge["step"] = max(target_edge.get("step", 0), 
                                  plan.get("_step", 0))

        # 移除条件: weight < -0.5 且 n_support == 0
        edge_removed = False
        if new_weight < -0.5 and target_edge.get("n_support", 0) == 0:
            # 从图中移除
            edges = graph.edges.get(src, [])
            graph.edges[src] = [e for e in edges if not (e["to"] == dst and e["rel"] == rel)]
            edge_removed = True

        verdict["verdict"] = "support" if score >= 0 else "refute"
        verdict["score"] = round(score, 3)
        verdict["n_support"] = n_sup
        verdict["n_against"] = n_ag
        verdict["edge_removed"] = edge_removed
        verdict["old_weight"] = round(old_weight, 3)
        verdict["new_weight"] = round(new_weight, 3)

        self.history.append({
            "hypothesis_key": hypothesis_key,
            "verdict": verdict["verdict"],
            "score": score,
        })

        return verdict
