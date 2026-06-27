"""
推理引擎 — 从 FactGraph 节点自动发现事实之间的关系

不做 LLM, 不做手写规则。而是用图结构推导模式:
  1. 数值对比: 同类事实中谁比谁大/小
  2. 模式匹配: 值的格式属于什么类型 (hex/ip/path/containerID)
  3. 类别聚合: 某类别有多少节点
  4. 变化检测: 同一 key 在不同时间的值变化
  5. 相关性: 高置信度系统事实之间的隐含关系
"""

import re


class InferenceEngine:
    """从 FactGraph 自发现事实之间的关系"""

    def __init__(self, graph):
        self.graph = graph
        self._inferred_history: set[str] = set()  # 已推断过的结论, 防重复

    def infer_all(self, step: int) -> int:
        """运行所有推断, 返回新产生的推断数"""
        if not self.graph or not self.graph.nodes:
            return 0

        n_new = 0
        n_new += self._infer_numeric_comparisons(step)
        n_new += self._infer_patterns(step)
        n_new += self._infer_aggregations(step)
        n_new += self._infer_cross_category(step)
        return n_new

    def _add_inference(self, key: str, value: str, step: int,
                       confidence: float = 0.7):
        """添加一个推断节点到 FactGraph"""
        if key in self._inferred_history:
            return
        self._inferred_history.add(key)
        self.graph.add_node(key, value, category="inference",
                            confidence=confidence, step=step,
                            source_cmd="inference_engine")

    # ── 1. 数值对比 ──

    def _infer_numeric_comparisons(self, step: int) -> int:
        """找出数值事实中 'A 大于/小于 B' 的关系"""
        n_new = 0
        # 从所有节点中找数值
        numeric: dict[str, float] = {}
        for key, node in self.graph.nodes.items():
            try:
                # 尝试解析数字值 (支持 "30Gi", "81G", "22" 等格式)
                val_str = str(node.value)
                num = float(re.search(r'(\d+\.?\d*)', val_str).group(1))
                numeric[key] = num
            except (ValueError, AttributeError, IndexError):
                continue

        if len(numeric) < 2:
            return 0

        # 找相关事实 (名称共享子串的)
        keys_list = list(numeric.keys())
        # 只比较同类事实 (共享前缀的)
        compared = 0
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                if compared > 5:  # 最多 5 个比较
                    break
                k1, k2 = keys_list[i], keys_list[j]
                v1, v2 = numeric[k1], numeric[k2]
                # 只比较共享前缀的 (如 disk_xxx vs disk_yyy)
                prefix = ''
                for pi in range(min(len(k1), len(k2))):
                    if k1[pi] == k2[pi]:
                        prefix += k1[pi]
                    else:
                        break
                if len(prefix) < 3:  # 前缀至少 3 字符
                    continue

                compared += 1
                ratio = v1 / v2 if v2 != 0 else 0
                if ratio > 1.5:
                    key = f"inf_{k1}_gt_{k2}"
                    self._add_inference(key, f"{v1:.0f} > {v2:.0f} (x{ratio:.1f})",
                                        step, confidence=0.6)
                    n_new += 1
                elif 0 < ratio < 0.67:
                    inv = 1.0 / ratio if ratio > 0 else 0
                    key = f"inf_{k1}_lt_{k2}"
                    self._add_inference(key, f"{v1:.0f} < {v2:.0f} (x{inv:.1f})",
                                        step, confidence=0.6)
                    n_new += 1

        return n_new

    # ── 2. 模式匹配 ──

    def _infer_patterns(self, step: int) -> int:
        """从值格式推断含义"""
        n_new = 0
        patterns = {
            r'^[a-f0-9]{12}$': 'looks_like_container_id',
            r'^\d+\.\d+\.\d+\.\d+/\d+$': 'looks_like_ip',
            r'^([a-f0-9]{2}:){5}[a-f0-9]{2}$': 'looks_like_mac',
            r'^\d+\.\d+\.\d+\.\d+$': 'looks_like_kernel_version',
            r'^/': 'looks_like_path',
            r'^[a-z]{2,6}_[A-Z]': 'looks_like_env_variable',
        }

        max_patterns = 20
        for key, node in self.graph.nodes.items():
            if n_new >= max_patterns:
                break
            val = str(node.value)
            for pattern, conclusion in patterns.items():
                if re.match(pattern, val.strip()):
                    inf_key = f"inf_{key}_pattern"
                    self._add_inference(inf_key, conclusion, step, confidence=0.5)
                    n_new += 1
                    break

        return n_new

    # ── 3. 类别聚合 ──

    def _infer_aggregations(self, step: int) -> int:
        """统计每种类别的节点数"""
        n_new = 0
        cat_count: dict[str, int] = {}
        for node in self.graph.nodes.values():
            cat_count[node.category] = cat_count.get(node.category, 0) + 1

        for cat, count in cat_count.items():
            if count >= 3 and cat not in ("general", "command", "script"):
                key = f"inf_n_{cat}_facts"
                self._add_inference(key, f"{count} {cat} facts found",
                                    step, confidence=0.8)
                n_new += 1

        return n_new

    # ── 4. 跨类别推断 ──

    def _infer_cross_category(self, step: int) -> int:
        """跨类别事实的组合推断"""
        n_new = 0
        # 收集各类别的事实
        by_cat: dict[str, list[tuple[str, str]]] = {}
        for key, node in self.graph.nodes.items():
            by_cat.setdefault(node.category, []).append((key, str(node.value)))

        # 系统 + 能力 → 能做什么
        sys_facts = dict(by_cat.get("system", []))
        cap_facts = dict(by_cat.get("capability", []))
        net_facts = dict(by_cat.get("network", []))

        if sys_facts:
            # CPU + 内存 → 服务器等级
            cpu = sys_facts.get("cpu_cores", "0")
            mem = sys_facts.get("mem_memtotal", "0")
            try:
                cpu_n = int(re.search(r'\d+', cpu).group())
                mem_n = float(re.search(r'(\d+\.?\d*)', mem).group())
                if cpu_n >= 8 and mem_n >= 8000000:
                    self._add_inference("inf_server_class",
                                        f"server-class: {cpu_n} cores, {mem_n/1000000:.0f}G RAM",
                                        step, confidence=0.7)
                    n_new += 1
            except (ValueError, AttributeError):
                pass

        # 有 python3 + gcc → 可以开发
        dev_tools = []
        if "capability_python" in cap_facts:
            dev_tools.append("python3")
        if "capability_compile" in cap_facts:
            dev_tools.append("gcc")
        if dev_tools and len(dev_tools) >= 2:
            self._add_inference("inf_can_develop",
                                f"development ready: {', '.join(dev_tools)}",
                                step, confidence=0.7)
            n_new += 1

        # 无网络接口 → 隔离环境
        if len(net_facts) <= 1:
            self._add_inference("inf_isolated",
                                "isolated environment (no/minimal network)",
                                step, confidence=0.6)
            n_new += 1

        return n_new

    def get_stats(self) -> dict:
        return {
            "n_inferences": len(self._inferred_history),
        }
