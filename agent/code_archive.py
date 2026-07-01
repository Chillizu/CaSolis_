"""CodeArchive — 代码存档 + 自评 + 成长建议

功能:
  1. 持久化每次 DeepSeek 生成的代码到 data/code_archive/
  2. 追踪大小/复杂度/导入数趋势
  3. 为 GoalGenerator 提供成长相关观测
  4. 检测代码组合机会
"""

import os, json, re
from pathlib import Path
from typing import Optional


class CodeArchive:
    """代码存档 + 自评引擎"""

    def __init__(self, archive_dir: str = "data/code_archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.archive_dir / "archive_meta.jsonl"
        self.metadata: list[dict] = []
        self._load_metadata()

    def _load_metadata(self):
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.metadata.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    def save(self, filename: str, content: str, step: int,
             idea: str = "", success: bool = True) -> Optional[dict]:
        """保存代码文件 + 元数据"""
        if not content or len(content) < 20:
            return None

        safe_name = f"{step:06d}_{Path(filename).name}"
        dest = self.archive_dir / safe_name
        try:
            with open(dest, "w") as f:
                f.write(content)
        except OSError:
            return None

        meta = {
            "step": step,
            "filename": safe_name,
            "size": len(content),
            "lines": content.count("\n") + 1,
            "functions": len(re.findall(r'^\s*def\s+\w+\s*\(', content, re.MULTILINE)),
            "classes": len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE)),
            "imports": self._extract_imports(content),
            "idea": idea[:300],
            "success": success,
        }
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.metadata.append(meta)

        # 触发自动总结 (每 10 条)
        if len(self.metadata) % 10 == 0:
            self._log_summary()

        return meta

    def _extract_imports(self, content: str) -> list[str]:
        """提取 import 语句中的模块名"""
        imports = []
        for line in content.split("\n")[:30]:
            line = line.strip()
            if line.startswith("import "):
                parts = line[7:].split(",")
                for p in parts:
                    name = p.strip().split()[0] if p.strip() else ""
                    name = name.split(".")[0] if name else ""
                    if name and not name.startswith("#"):
                        imports.append(name)
            elif line.startswith("from "):
                name = line.split()[1] if len(line.split()) > 1 else ""
                name = name.split(".")[0] if name else ""
                if name and not name.startswith("#"):
                    imports.append(name)
        return sorted(set(imports))

    def get_stats(self) -> dict:
        """整体成长统计"""
        if not self.metadata:
            return {
                "count": 0, "max_size": 0, "avg_size": 0,
                "total_lines": 0, "total_functions": 0,
            }

        sizes = [m["size"] for m in self.metadata]
        lines = [m["lines"] for m in self.metadata]
        funcs = [m.get("functions", 0) for m in self.metadata]
        recent = self.metadata[-10:]

        n = len(self.metadata)
        half = n // 2

        result = {
            "count": n,
            "max_size": max(sizes),
            "avg_size": sum(sizes) / n,
            "total_lines": sum(lines),
            "total_functions": sum(funcs),
            "recent_avg": sum(m["size"] for m in recent) / max(len(recent), 1),
        }

        # 前后半对比 (成长趋势)
        if half > 0:
            first_half = sizes[:half]
            last_half = sizes[half:]
            result["first_avg"] = sum(first_half) / len(first_half)
            result["last_avg"] = sum(last_half) / len(last_half)
            result["growth_ratio"] = result["last_avg"] / max(result["first_avg"], 1)
        else:
            result["first_avg"] = result["avg_size"]
            result["last_avg"] = result["avg_size"]
            result["growth_ratio"] = 1.0

        # 导入多样性
        all_imports = set()
        for m in self.metadata:
            all_imports.update(m.get("imports", []))
        result["unique_imports"] = len(all_imports)
        result["import_list"] = sorted(all_imports)

        return result

    def get_observations(self) -> list[tuple[str, str]]:
        """为 GoalGenerator 提供观测"""
        stats = self.get_stats()
        obs = []
        if stats["count"] == 0:
            return obs

        # 基础: 存档存在
        obs.append(("code-archive",
                     f"code_archive: wrote {stats['count']} scripts, "
                     f"total {stats['total_lines']} lines, "
                     f"{stats['total_functions']} functions"))

        # 成长: 脚本大小增长
        if stats["count"] >= 4 and stats["growth_ratio"] > 1.3:
            obs.append(("code-growth",
                         f"scripts growing: {stats['first_avg']:.0f}B "
                         f"-> {stats['last_avg']:.0f}B "
                         f"({stats['growth_ratio']:.1f}x)"))

        # 停滞: 数量多但没增长
        if stats["count"] >= 6 and stats["growth_ratio"] < 1.1:
            obs.append(("code-stuck",
                         f"writing same size ({stats['avg_size']:.0f}B) for "
                         f"{stats['count']} scripts — need bigger challenge"))

        # 组合: 多个不同 imports 的脚本可合并
        if stats["count"] >= 3 and stats["unique_imports"] >= 3:
            combos = self._find_composition_targets()
            for tag, txt in combos[:2]:
                obs.append((tag, txt))

        # 里程碑
        milestones = []
        for m in self.metadata[-3:]:
            if m["size"] > 3000:
                milestones.append(m["size"])
        if milestones:
            obs.append(("code-milestone",
                         f"recent scripts hit {max(milestones)}B — "
                         f"getting substantial"))

        return obs

    def _find_composition_targets(self) -> list[tuple[str, str]]:
        """找可组合的脚本对"""
        if len(self.metadata) < 2:
            return []

        recent = self.metadata[-8:]
        combos = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                a, b = recent[i], recent[j]
                ai = set(a.get("imports", []))
                bi = set(b.get("imports", []))
                # 不同的 import = 互补
                shared = ai & bi
                unique_a = ai - bi
                unique_b = bi - ai
                if len(unique_a) >= 1 and len(unique_b) >= 1 and len(shared) == 0:
                    combos.append(("code-compose",
                        f"combine: {a['filename'][:20]} ({ai}) + "
                        f"{b['filename'][:20]} ({bi})"))
        return combos

    def _log_summary(self):
        """每 10 条打印一次摘要"""
        stats = self.get_stats()
        print(f"  [ARCHIVE] {stats['count']} scripts, "
              f"avg {stats['avg_size']:.0f}B, "
              f"{stats['total_functions']} funcs, "
              f"{stats['unique_imports']} unique imports")
