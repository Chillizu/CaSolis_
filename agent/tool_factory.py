"""
ToolFactory — 无模板, 从 418 命令池自造工具

每个"工具"是一个 JSON 文件:
  {"cmd": ["command", "--help"], "desc": "show command help", "cat": "discovered"}

ToolRegistry 读取 JSON 直接执行, 不再需要 Python 模板。
"""

import json, os, random
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path("data/persistent/tools")

class ToolFactory:
    """从 418 命令池自造工具 — 无模板, 无 Python 代码生成"""

    def __init__(self, tools_dir: str = "data/persistent/tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def get_available_templates(self) -> list[dict]:
        return []  # 无模板

    def generate(self, template_name: str, created_step: int = 0) -> Optional[str]:
        return None  # 无模板

    def generate_all(self, created_step: int = 0) -> list[str]:
        return []  # 不再使用, 由 discover_new_tools 替代

    def get_tool_info(self, template_name: str) -> Optional[dict]:
        return None

    def discover_new_tools(self, knowledge_mapper, n: int = 3) -> list[str]:
        """
        从 KnowledgeMapper 的 418 命令池中挑 n 个未注册过的命令,
        对每个命令生成工具 (JSON 文件描述: cmd + args)

        Args:
            knowledge_mapper: KnowledgeMapper 实例
            n: 最多生成 n 个工具

        Returns:
            生成的工具文件名列表
        """
        generated = []
        all_cmds = getattr(knowledge_mapper, '_all_available_commands', [])

        if not all_cmds:
            return generated

        # 提取已注册的工具命令 (去重)
        existing_tools = set()
        if self.tools_dir.exists():
            for f in self.tools_dir.iterdir():
                if f.suffix == '.json':
                    try:
                        data = json.loads(f.read_text())
                        if 'cmd' in data:
                            existing_tools.add(tuple(data['cmd']))
                    except Exception:
                        pass

        # 只取带参数的 command --help 模式
        tool_variants = []
        for cmd in all_cmds:
            if (cmd,) not in existing_tools:
                tool_variants.append((cmd,))

        # 不够时加 --help 变体
        if len(tool_variants) < n:
            for cmd in all_cmds:
                if (cmd,) in existing_tools or (cmd, '--help') in existing_tools:
                    continue
                if (cmd,) not in existing_tools:  # 如果原命令未注册, 先注册
                    continue  # 已包含在上面的循环里
                # 否则加 --help 变体
                tool_variants.append((cmd, '--help'))
                if len(tool_variants) >= n * 2:
                    break

        # 如果没有足够的工具候选, 补充 --version
        if len(tool_variants) < n:
            for cmd in all_cmds:
                if (cmd,) in existing_tools and (cmd, '--version') not in existing_tools:
                    tool_variants.append((cmd, '--version'))
                    if len(tool_variants) >= n * 2:
                        break

        random.shuffle(tool_variants)
        selected = tool_variants[:n]

        for variant in selected:
            cmd_args = list(variant)
            cmd_name = cmd_args[0]
            fname = f"tool_{cmd_name}.json"

            # 写 JSON 工具定义
            tool_def = {
                "cmd": cmd_args,
                "desc": f"run {cmd_name} with arguments",
                "cat": "discovered",
                "source": "tool_factory",
            }
            (self.tools_dir / fname).write_text(json.dumps(tool_def, indent=2))
            generated.append(fname)

        return generated

    def auto_generate_from_facts(self, fact_graph, knowledge_mapper=None) -> list[str]:
        """从 FactGraph 中发现新类别, 但在新系统中直接由 discover_new_tools 替代"""
        return []
