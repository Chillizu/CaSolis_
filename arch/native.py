"""
Arch v0.4 — 原生交互架构

核心思想：
- 词表 = ASCII 字符 (0-95) + 行动 token (96-111) + 特殊 (112-127)
- 环境返回真实字符串（如 "total 128\ndrwxr-xr-x ..."）
- 字符串编码为字符 token 序列
- 模型一边"想"（生成字符 token 流），一边"做"（生成行动 token）
- 行动 token = 原生操作，没有格式层

环境交互：
  模型输出 'l' 's' ' ' '-' 'l' 'a' → 这些是思考，不执行
  模型输出 [ACTION_LS] → 环境拦截，执行 ls -la，返回结果文本
  模型收到 "total 128\ndrwxr-xr-x 2 ..." → 继续阅读、思考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 字符集 ──────────────────────────────────────────────────
# 0-95: ASCII 可打印字符（包括空格、换行等）
# 96-111: 16 个行动 token
# 112: BOS (begin of sequence)
# 113-127: 保留

CHARSET = ''.join(chr(i) for i in range(32, 128))  # 空格到 ~
CHAR_TO_ID = {c: i for i, c in enumerate(CHARSET)}  # 0-95
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}

N_CHAR = len(CHARSET)         # 96
N_ACTION = 16                 # 16
N_SPECIAL = 16                # 16
VOCAB_SIZE = N_CHAR + N_ACTION + N_SPECIAL

# Token 范围
CHAR_START = 0                # 0-95: 字符
ACTION_START = N_CHAR         # 96-111: 行动
SPECIAL_START = N_CHAR + N_ACTION  # 112-127: 特殊

BOS_TOKEN = SPECIAL_START     # 112


def text_to_tokens(text: str) -> list[int]:
    """将字符串编码为字符 token 序列。不可识别字符替换为空格。"""
    return [CHAR_TO_ID.get(c, CHAR_TO_ID[' ']) for c in text]


def tokens_to_text(tokens: list[int]) -> str:
    """将字符 token 序列解码为字符串。过滤非字符 token。"""
    chars = []
    for t in tokens:
        if t < N_CHAR:
            chars.append(ID_TO_CHAR[t])
    return ''.join(chars)


class ActionMap:
    """行动 token → shell 命令映射"""

    def __init__(self):
        self.actions = [
            "ls -la",           # 0
            "pwd",              # 1
            "date",             # 2
            "whoami",           # 3
            "df -h",            # 4
            "cat /etc/hostname",# 5
            "uname -a",         # 6
            "echo hello",       # 7
            "ls /tmp",          # 8
            "id",               # 9
            "uptime",           # 10
            "free -h",          # 11
            "du -sh /tmp",      # 12
            "echo test",        # 13
            "ls -d .",          # 14
            "who -b",           # 15
        ]

    def get_cmd(self, action_idx: int) -> str:
        return self.actions[action_idx]

    def token_to_action_idx(self, token: int) -> int | None:
        if ACTION_START <= token < ACTION_START + N_ACTION:
            return token - ACTION_START
        return None


class NativeCore(nn.Module):
    """
    原生交互核心

    - 词表：ASCII 字符 (0-95) + 行动 (96-111) + 特殊 (112-127)
    - RNN 持续处理字符流（= 阅读环境输出 + 思考）
    - 行动 token 被环境拦截并执行
    - 没有 prompt，没有格式，永远在跑
    """

    def __init__(self, hidden_dim: int = 256, embed_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Token 嵌入（字符 + 行动 + 特殊）
        self.token_embed = nn.Embedding(VOCAB_SIZE, embed_dim)

        # RNN 核心 — 持续思考
        self.rnn = nn.GRUCell(embed_dim, hidden_dim)

        # 输出层
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_act = nn.GELU()

        # 下一个 token 预测（语言模型头）
        self.lm_head = nn.Linear(hidden_dim, N_CHAR)  # 只预测字符，不预测行动

        # 行动头（特殊 token 触发）
        self.action_head = nn.Linear(hidden_dim, N_ACTION)

        # 世界模型头（预测行动结果的关键信息）
        self.world_head = nn.Linear(hidden_dim + embed_dim, N_CHAR)

        # 好奇心门
        self.curiosity_gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.orthogonal_(p, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(p)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def is_action(self, token: int) -> bool:
        return ACTION_START <= token < ACTION_START + N_ACTION

    def step(self, h: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        单步推理

        参数:
            h: 隐藏状态 (batch, hidden)
            token: 输入的 token (batch,) — 可以是字符、行动、或特殊

        返回:
            h_new: 新隐藏状态
            outputs: {
                'char_logits': 下一个字符预测
                'action_logits': 行动概率
                'curiosity': 好奇心门值
            }
        """
        if token.dim() == 0:
            token = token.unsqueeze(0)

        emb = self.token_embed(token)  # (batch, embed_dim)

        # RNN 思考一步
        h_new = self.rnn(emb, h)
        h_norm = self.out_norm(h_new)
        h_proj = self.out_act(self.out_proj(h_norm))

        # 预测下一个字符（语言模型）
        char_logits = self.lm_head(h_proj)

        # 行动预测
        action_logits = self.action_head(h_proj)

        # 好奇心门
        curiosity = self.curiosity_gate(h_proj)

        return h_new, {
            "char_logits": char_logits,
            "action_logits": action_logits,
            "curiosity": curiosity,
        }
