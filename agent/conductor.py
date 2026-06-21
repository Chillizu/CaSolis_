"""
指挥家 (Conductor) V2 — 想法向量生成器

蒸馏法: 利用现有 11 类分类器的 logits 作为训练信号
修正语义坍缩问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sentence_transformers import SentenceTransformer


N_DIMS = 16
INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE", "LS_TMP", "ARCH_INFO", "CUSTOM"]
N_INTENTS = 13  # 有效意图数 (不含 CUSTOM)


class ConductorHead(nn.Module):
    """
    MiniLM(冻结) → MLP → 16维想法向量

    结构: 384 → 128 → 64 → [16(thought) + 11(class_proj)]
    class_proj 只在蒸馏训练时用, 推理时丢掉

    参数量: 384*128 + 128*64 + 64*27 ≈ 67K
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 想法向量 (推理时用)
        self.thought_head = nn.Linear(64, N_DIMS)
        # 分类投影 (只在蒸馏训练时用, 不含 CUSTOM)
        self.class_proj = nn.Linear(64, N_INTENTS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, 384) — MiniLM 嵌入
        Returns:
            thought: (batch, 16) — 想法向量 (tanh)
            logits: (batch, 11) or None — 分类投影 (训练时)
        """
        h = self.shared(x)
        thought = torch.tanh(self.thought_head(h))  # [-1, 1] 避免坍缩
        logits = self.class_proj(h)  # 蒸馏用
        return thought, logits


class Conductor:
    """
    指挥家

    使用:
        c = Conductor()
        thought, logits = c.forward_emb(emb)  # 蒸馏训练
        thought = c.forward(state_text)        # 推理
    """

    def __init__(self, checkpoint: str = None, device: str = "cpu"):
        self.device = device
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.head = ConductorHead().to(device)

        if checkpoint and os.path.exists(checkpoint):
            sd = torch.load(checkpoint, map_location=device, weights_only=True)
            # 适配新旧intent数不同 (class_proj 可能尺寸不匹配)
            try:
                self.head.load_state_dict(sd)
            except Exception:
                self.head.load_state_dict(sd, strict=False)
                print(f"  ⚠️ Conductor 加载部分权重 (class_proj 已扩展)")
            else:
                print(f"  ✅ Conductor 已加载: {checkpoint}")

        self.head.eval()

    def forward(self, state_text: str) -> torch.Tensor:
        """推理: 状态文本 → 16维想法向量"""
        emb = self.encoder.encode(state_text, convert_to_tensor=True).clone().to(self.device)
        with torch.no_grad():
            thought, _ = self.head(emb.unsqueeze(0))
        return thought.squeeze(0)

    def forward_emb(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """训练: 嵌入 → (想法向量, 分类logits)"""
        return self.head(emb)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.head.state_dict(), path)

    def load(self, path: str):
        self.head.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.head.eval()
