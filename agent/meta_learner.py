"""
P5.4: 元学习器 — 行为效用追踪 + 自我淘汰/强化

追踪什么:
  - 提取规则: 规则是否真正提取到了有用的事实?
  - 脚本模板: 脚本产出多少事实? 有没有被复用?
  - 探针路径: 探针是否导向了链式发现?
  - 意图选择: 想象力 vs 分类器 vs 指挥家, 谁更可靠?

生命周期:
  record(id, delta, step) → 更新效用滑动平均
  prune() → 淘汰 n>=5 且 utility < -0.3 的行为
  get_best(type, n) → 推荐最有效的 N 个行为
  save/load → /persistent/metadata/meta.json
"""

import os
import json
from typing import Optional


STORAGE_PATH = "data/persistent/metadata/meta.json"


class MetaLearner:
    """元学习器: 追踪每个行为的效果, 只保留有用的"""

    def __init__(self, path: str = ""):
        if not path:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, STORAGE_PATH)
        self.path = path
        self.data: dict[str, dict] = {}  # id → {type, params, utility, n, last_step, ...}
        self.load()

    # ── 注册 & 记录 ──

    def register(self, behavior_id: str, btype: str, params: dict = None,
                 created_step: int = 0):
        """注册一个新行为"""
        if behavior_id not in self.data:
            self.data[behavior_id] = {
                "type": btype,
                "params": params or {},
                "utility": 0.0,
                "n": 0,
                "last_step": 0,
                "created_step": created_step,
            }

    def record(self, behavior_id: str, utility_delta: float, step: int):
        """记录行为效果, 更新滑动平均效用

        utility_delta:
          +1.0 = 成功提取事实
          +0.5 = 脚本产出新事实
          -0.5 = 规则未命中
          -1.0 = 规则提取了错误/无用的信息
        """
        if behavior_id not in self.data:
            return
        b = self.data[behavior_id]
        b["n"] += 1
        b["last_step"] = step
        # EMA 滑动平均
        alpha = 1.0 / min(b["n"], 20)  # 从1衰减到0.05
        b["utility"] = (1 - alpha) * b["utility"] + alpha * utility_delta

    # ── 查询 ──

    def get_best(self, btype: str = "", n: int = 3,
                 min_trials: int = 2) -> list[dict]:
        """返回效用最高的 N 个行为"""
        candidates = []
        for bid, b in self.data.items():
            if btype and b.get("type") != btype:
                continue
            if b.get("n", 0) < min_trials:
                continue
            candidates.append({"id": bid, **b})
        candidates.sort(key=lambda x: -x.get("utility", -999))
        return candidates[:n]

    def get_worst(self, btype: str = "", n: int = 3,
                  min_trials: int = 3) -> list[dict]:
        """返回效用最低的 N 个行为 (用于淘汰)"""
        candidates = []
        for bid, b in self.data.items():
            if btype and b.get("type") != btype:
                continue
            if b.get("n", 0) < min_trials:
                continue
            candidates.append({"id": bid, **b})
        candidates.sort(key=lambda x: x.get("utility", 999))
        return candidates[:n]

    def get_stats(self) -> dict:
        """摘要统计"""
        by_type = {}
        for bid, b in self.data.items():
            t = b.get("type", "unknown")
            if t not in by_type:
                by_type[t] = {"count": 0, "avg_utility": 0.0, "total_trials": 0}
            by_type[t]["count"] += 1
            by_type[t]["avg_utility"] += b.get("utility", 0)
            by_type[t]["total_trials"] += b.get("n", 0)
        for t in by_type:
            by_type[t]["avg_utility"] /= max(by_type[t]["count"], 1)
        return {
            "total_behaviors": len(self.data),
            "by_type": by_type,
        }

    # ── 淘汰 & 清理 ──

    def prune(self, min_trials: int = 5, threshold: float = -0.3,
              max_age: int = 500, current_step: int = 0) -> int:
        """淘汰低效用或过期的行为

        Args:
            min_trials: 至少试过这么多次才考虑淘汰
            threshold: utility 低于此值就淘汰
            max_age: 超过此步数未使用就淘汰
            current_step: 当前步数 (用于 age 检查)

        Returns:
            淘汰的数量
        """
        to_remove = []
        for bid, b in self.data.items():
            # 低效用淘汰
            if b.get("n", 0) >= min_trials and b.get("utility", 0) < threshold:
                to_remove.append(bid)
                continue
            # 过期淘汰 (old + unused)
            if (current_step > 0 and b.get("last_step", 0) > 0
                    and current_step - b.get("last_step", 0) > max_age
                    and b.get("n", 0) <= 3):
                to_remove.append(bid)
                continue
        for bid in to_remove:
            del self.data[bid]
        return len(to_remove)

    def get_summary(self, top_n: int = 5) -> str:
        """可读摘要"""
        best = self.get_best(n=top_n)
        worst = self.get_worst(n=top_n)
        stats = self.get_stats()
        lines = [
            f"元学习器: {stats['total_behaviors']} 个行为",
        ]
        for bt, info in stats.get("by_type", {}).items():
            lines.append(f"  {bt}: {info['count']}个, 均效用={info['avg_utility']:.2f}")
        if best:
            lines.append(f"  最佳:")
            for b in best:
                lines.append(f"    {b['id']}: {b['utility']:.2f} ({b['n']}次)")
        if worst:
            lines.append(f"  最差:")
            for b in worst:
                lines.append(f"    {b['id']}: {b['utility']:.2f} ({b['n']}次)")
        return "\n".join(lines)

    # ── 持久化 ──

    def save(self):
        """持久化到 JSON"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def load(self):
        """从 JSON 加载"""
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}
