"""
Stream Core v0.3 — 双通道架构

思考通道 (token 0-95)：内部 RNN 动态，不输出给环境
行动通道 (token 96-111)：唯一影响环境的方式
结果观察 (token 112-127)：来自环境的反馈

每个周期：观察 → 隐藏状态更新(思考) → 决定行动 → 世界模型预测 → 环境执行 → 学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualChannelCore(nn.Module):
    """
    双通道核心

    架构：
    - RNN 隐藏状态 = "意识流"（持续更新）
    - 观察嵌入 → RNN 更新 = 思考
    - 共享投射层 → 行动头 + 世界模型 + 好奇心门
    """

    def __init__(
        self,
        thought_vocab: int = 96,
        action_vocab: int = 16,
        obs_vocab: int = 16,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.thought_vocab = thought_vocab
        self.action_vocab = action_vocab
        self.obs_vocab = obs_vocab
        self.total_vocab = thought_vocab + action_vocab + obs_vocab
        self.hidden_dim = hidden_dim

        # 统一 Token 嵌入
        self.token_embed = nn.Embedding(self.total_vocab, embed_dim)

        # 循环核心 — 这就是"思考"发生的地方
        self.rnn = nn.GRUCell(embed_dim, hidden_dim)

        # 共享层
        self.shared_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # 行动头
        self.action_head = nn.Linear(hidden_dim // 2, action_vocab)

        # 世界模型（预测行动结果）
        self.world_head = nn.Linear(hidden_dim // 2, obs_vocab)

        # 好奇心门
        self.curiosity_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def observe_and_think(self, h: torch.Tensor, obs_token: torch.Tensor) -> torch.Tensor:
        """
        观察并思考：将观察 token 嵌入，更新隐藏状态

        "思考" = RNN 将新观察融合进持续的隐藏状态
        """
        if obs_token.dim() == 0:
            obs_token = obs_token.unsqueeze(0)
        emb = self.token_embed(obs_token)
        return self.rnn(emb, h)

    def decide(self, h: torch.Tensor, action_embed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从隐藏状态 +（可选）行动嵌入做出决定

        关键：世界模型需要知道选择了哪个行动，才能预测结果。
        行动嵌入作为条件信息输入世界模型。

        返回:
            action_logits: 行动概率
            world_logits: 世界模型预测（给定行动后的结果）
            curiosity: 好奇心门值
        """
        shared = self.shared_net(h)
        action_logits = self.action_head(shared)

        # 世界模型预测：需要行动信息
        if action_embed is not None:
            world_input = torch.cat([shared, action_embed], dim=-1)
            world_logits = self.world_head(world_input)
        else:
            # 推理时不知道行动，用均值代替
            world_logits = self.world_head(shared)

        curiosity = self.curiosity_gate(shared)
        return action_logits, world_logits, curiosity


class DualChannelEnv:
    """
    双通道环境

    仅响应行动 token (96-111)：
    - 执行对应的 shell 命令
    - 返回结果 token (112-127)
    """

    def __init__(self):
        self.obs_offset = 112

        self.commands = [
            "ls -la",           # 0
            "pwd",              # 1
            "date",             # 2
            "whoami",           # 3
            "df -h",            # 4
            "hostname",         # 5
            "uname -a",         # 6
            "echo hello",       # 7
            "ls /tmp",          # 8
            "id",               # 9
            "uptime",           # 10
            "free -h",          # 11
            "du -sh /tmp",      # 12
            "dmesg | tail -3",  # 13
            "who -b",           # 14
            "ls /etc | head -5", # 15
        ]

        self.step_count = 0

    def act(self, action_index: int) -> tuple[int, str]:
        """执行行动，返回结果观察 token + 命令名"""
        self.step_count += 1
        cmd = self.commands[action_index]
        result_token = self.obs_offset + action_index
        return result_token, cmd
