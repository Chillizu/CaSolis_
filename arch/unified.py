"""
Stream Core v0.2 — 统一 Token 空间

核心思想：Token 本身就是操作符。
- 0-95:  思考 token（内部语言，不影响环境）
- 96-111: 命令 token（直接映射为 shell 命令）
- 112-127: 特殊 token

模型只做一件事：预测下一个 token。
环境拦截命令 token，执行后将输出编码回 token。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCore(nn.Module):
    """统一 Token 架构 — 一个核心，一个 Head，Token 即操作"""

    def __init__(
        self,
        vocab_size: int = 128,       # 总词表大小
        embed_dim: int = 64,         # 嵌入维度
        hidden_dim: int = 256,       # 隐藏状态维度
        cmd_start: int = 96,         # 命令 token 起始索引
        cmd_end: int = 112,          # 命令 token 结束索引（不含）
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.cmd_start = cmd_start
        self.cmd_end = cmd_end
        self.n_commands = cmd_end - cmd_start  # 16 个命令槽

        # Token 嵌入 — 包含所有 token（思考 + 命令 + 特殊）
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # 循环核心 — 单个 GRU
        self.rnn = nn.GRUCell(embed_dim, hidden_dim)

        # 统一预测头 — 预测下一个 token
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(param)

        # 命令 token 的嵌入用小值初始化（避免初始偏见）
        with torch.no_grad():
            self.token_embed.weight[self.cmd_start:self.cmd_end] *= 0.1

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def is_command(self, token_id: int) -> bool:
        return self.cmd_start <= token_id < self.cmd_end

    def get_command_index(self, token_id: int) -> int:
        """将命令 token 转换为命令索引（供环境查找具体命令）"""
        return token_id - self.cmd_start

    def step(
        self,
        h: torch.Tensor,
        obs_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        单步前向传播

        参数:
            h: 隐藏状态 (batch, hidden_dim)
            obs_token: 观察值 token (batch,)

        返回:
            h_new: 新隐藏状态
            outputs: {'token_logits': (batch, vocab_size) 下一个 token 预测}
        """
        if obs_token.dim() == 0:
            obs_token = obs_token.unsqueeze(0)

        # Token 嵌入
        emb = self.token_embed(obs_token)  # (batch, embed_dim)

        # RNN 更新
        h_new = self.rnn(emb, h)
        h_new = self.layer_norm(h_new)

        # 下一个 token 预测
        token_logits = self.lm_head(h_new)

        return h_new, {"token_logits": token_logits}


class UnifiedEnv:
    """
    统一环境 — 命令 token 直接映射为 shell 命令

    当模型生成命令 token，环境执行对应命令，
    并将输出编码为观察 token 流返回给模型。
    """

    def __init__(self, vocab_size: int = 128, cmd_start: int = 96, cmd_end: int = 112):
        self.vocab_size = vocab_size
        self.cmd_start = cmd_start
        self.cmd_end = cmd_end

        # 命令映射：命令索引 → shell 命令
        self.commands = [
            "ls -la",          # 0: 列出文件
            "pwd",             # 1: 当前路径
            "date",            # 2: 日期时间
            "whoami",          # 3: 当前用户
            "df -h",           # 4: 磁盘使用
            "cat /etc/hostname", # 5: 主机名
            "uname -a",        # 6: 系统信息
            "echo hello",      # 7: 测试输出
            "ls /tmp",         # 8: 临时目录
            "id",              # 9: 用户/组信息
            "uptime",          # 10: 运行时间
            "free -h",         # 11: 内存使用
            "du -sh /tmp",     # 12: 目录大小
            "dmesg | tail -3", # 13: 内核日志
            "who -b",          # 14: 启动时间
            "ls /etc | head -5", # 15: 配置列表
        ]

        # 历史记录
        self.action_history = []
        self.obs_history = []
        self.step_count = 0

    def get_initial_obs(self) -> int:
        """返回初始观察 token，作为模型启动时的输入"""
        return 112  # BOS token

    def step(self, token_id: int) -> tuple[int, dict]:
        """
        执行一步

        参数:
            token_id: 模型生成的 token

        返回:
            next_token: 下一个观察 token（可能是命令输出或思考 token）
            info: 额外信息
        """
        self.step_count += 1

        if self.cmd_start <= token_id < self.cmd_end:
            # 这是命令 token → 执行
            cmd_idx = token_id - self.cmd_start
            cmd = self.commands[cmd_idx]

            # 将命令输出编码为观察 token
            # 使用确定性编码：命令索引 → 输出 token 流
            # 简单起见：用 token = cmd_idx + 112 (特殊区域) 表示"命令已执行"
            # 实际应返回命令输出，但为了简化词汇需求，我们用一个唯一的 token 表示结果
            result_token = 112 + cmd_idx  # 112-127 特殊区域

            info = {
                "type": "command",
                "command_index": cmd_idx,
                "command": cmd,
                "result_token": result_token,
            }
            self.action_history.append(cmd)
            return result_token, info

        else:
            # 思考 token → 无环境操作
            info = {
                "type": "thought",
                "token_id": token_id,
                "token_emotion": self._token_meaning(token_id),
            }
            self.obs_history.append(token_id)
            return token_id, info  # 思考 token 作为观察值原样返回

    def _token_meaning(self, token_id: int) -> str:
        """给 token 一些语义意义（方便调试）"""
        if token_id < 16:
            return f"intent_topic_{token_id}"
        elif token_id < 32:
            return f"question_topic_{token_id}"
        elif token_id < 48:
            return f"plan_topic_{token_id}"
        elif token_id < 64:
            return f"action_describe_{token_id}"
        elif token_id < 80:
            return f"reflect_topic_{token_id}"
        elif token_id < 96:
            return f"explore_topic_{token_id}"
        elif token_id < 112:
            return "command"
        elif token_id < 128:
            return f"result_token"
        return "unknown"
