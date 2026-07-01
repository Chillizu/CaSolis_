"""
Word-Level Core — 词级思考链架构

词表:
  0-917:  BPE 词 token
  918-933: 行动 token (16 个命令)
  934-939: 特殊 token

思考链: 词 token = 内心独白, 行动 token = 操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# ── 加载 tokenizer ────────────────────────────────────────
TOKENIZER_PATH = "data/tokenizer-2k.json"
_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = _tokenizer.get_vocab_size()  # ~918

# 特殊 token 索引
SPECIAL_TOKENS = {
    "BOS": VOCAB_SIZE,     # 918
    "EOS": VOCAB_SIZE + 1, # 919
    "PAD": VOCAB_SIZE + 2, # 920
    "ACT": VOCAB_SIZE + 3, # 921 — 行动 token 起始
    "THK": VOCAB_SIZE + 4, # 922 — 思考标记（保留）
}

N_ACTION = 16
ACTION_START = SPECIAL_TOKENS["ACT"]          # 921
ACTION_END = ACTION_START + N_ACTION          # 937
SPECIAL_START = ACTION_END                     # 937
BOS_TOKEN = SPECIAL_TOKENS["BOS"]              # 918
TOTAL_VOCAB = SPECIAL_START + 3               # 940

COMMANDS = [
    "ls", "ls -la", "pwd", "date -u", "whoami", "id",
    "cat /etc/hostname", "uname -a", "df -h /", "free -h",
    "uptime", "echo hello", "hostname", "who -b",
    "ls /tmp", "du -sh /tmp",
]


def encode(text: str) -> list[int]:
    """将文本编码为 token 序列"""
    return _tokenizer.encode(text).ids


def decode(tokens: list[int]) -> str:
    """将 token 序列解码为文本（过滤非词 token）"""
    word_tokens = [t for t in tokens if t < VOCAB_SIZE]
    return _tokenizer.decode(word_tokens)


class WordLevelCore(nn.Module):
    """词级思考链核心"""

    def __init__(
        self,
        hidden_dim: int = 512,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = TOTAL_VOCAB
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Token 嵌入
        self.token_embed = nn.Embedding(TOTAL_VOCAB, embed_dim)

        # RNN 核心
        self.rnn = nn.GRUCell(embed_dim, hidden_dim)

        # 共享层
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 下一个词预测头（语言模型）
        self.lm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

        # 行动头
        self.action_head = nn.Linear(hidden_dim, N_ACTION)

        # 世界模型（预测行动输出，需要行动信息）
        self.world_head = nn.Linear(hidden_dim + embed_dim, VOCAB_SIZE)

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
        # 行动嵌入用小值
        with torch.no_grad():
            self.token_embed.weight[ACTION_START:ACTION_END] *= 0.1

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def is_action(self, token_id: int) -> bool:
        return ACTION_START <= token_id < ACTION_END

    def token_type(self, token_id: int) -> str:
        if token_id < VOCAB_SIZE: return "word"
        elif token_id < ACTION_END: return "action"
        else: return "special"

    def step(self, h: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """单步推理

        返回:
            h_new: 新隐藏状态
            outputs: {lm_logits, action_logits, world_logits, curiosity}
        """
        if token.dim() == 0:
            token = token.unsqueeze(0)

        emb = self.token_embed(token)
        h_new = self.rnn(emb, h)
        s = self.shared(h_new)

        # 下一个词预测
        lm_logits = self.lm_head(s)

        # 行动预测
        action_logits = self.action_head(s)

        # 世界模型（需要行动嵌入）
        action_emb = self.token_embed(
            torch.tensor([ACTION_START], dtype=torch.long)
        ).expand(emb.shape[0], -1)
        world_input = torch.cat([s, action_emb], dim=-1)
        world_logits = self.world_head(world_input)

        # 好奇心
        curiosity = self.curiosity_gate(s)

        return h_new, {
            "lm_logits": lm_logits,
            "action_logits": action_logits,
            "world_logits": world_logits,
            "curiosity": curiosity,
        }

    def step_with_action(
        self, h: torch.Tensor, token: torch.Tensor, action_idx: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """带真实行动信息的世界模型预测"""
        if token.dim() == 0:
            token = token.unsqueeze(0)
        if action_idx.dim() == 0:
            action_idx = action_idx.unsqueeze(0)

        emb = self.token_embed(token)
        h_new = self.rnn(emb, h)
        s = self.shared(h_new)

        lm_logits = self.lm_head(s)
        action_logits = self.action_head(s)

        # 世界模型用真实行动嵌入
        action_tok = ACTION_START + action_idx
        action_emb = self.token_embed(action_tok)
        world_input = torch.cat([s, action_emb], dim=-1)
        world_logits = self.world_head(world_input)

        curiosity = self.curiosity_gate(s)

        return h_new, {
            "lm_logits": lm_logits,
            "action_logits": action_logits,
            "world_logits": world_logits,
            "curiosity": curiosity,
        }


def count_parameters(model: WordLevelCore) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = WordLevelCore(hidden_dim=512, embed_dim=128)
    print(f"WordLevelCore 参数: {count_parameters(model):,}")
    print(f"词表: {TOTAL_VOCAB} (词 {VOCAB_SIZE} + 行动 {N_ACTION} + 特殊)")
    print()

    # 测试
    h = model.init_state()
    tok = torch.tensor([BOS_TOKEN], dtype=torch.long)
    h, out = model.step(h, tok)
    print(f"lm_logits: {out['lm_logits'].shape}")
    print(f"action_logits: {out['action_logits'].shape}")
    print(f"world_logits: {out['world_logits'].shape}")
    print(f"curiosity: {out['curiosity'].item():.4f}")
