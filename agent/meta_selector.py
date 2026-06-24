"""
元认知选择器 (MetaCognitiveSelector) — 三层 MODE 选择

人脑类比: 前额叶皮层 — 设定认知模式, 分配注意力

MODE:
  EXPLORE — 探索未知, 填事实缺口, 优先 CUSTOM/READ
  CREATE  — 从已知事实生成内容, 优先 WRITE/APPEND/GENERATE
  LEARN   — 验证假设, 修正世界模型, 优先训练 + 真实验证

设计: 规则为主 + 小 MLP 偏置
  - 前 100 步强制规则 (冷启动安全)
  - 每个 MODE 最小持有 30 步 (防振荡)
  - MLP 只在规则边界模糊时提供偏置
"""

import math
from typing import Optional

import torch
import torch.nn as nn


MODES = ["EXPLORE", "CREATE", "LEARN"]
N_MODES = 3

MODE_DESCRIPTIONS = {
    "EXPLORE": "探索模式 — 发现未知事实, 填事实缺口, 扩大知识覆盖",
    "CREATE": "创作模式 — 从已知事实生成内容, 产出报告/脚本/JSON",
    "LEARN":  "学习模式 — 验证假设, 修正模型, 跑训练+真实验证",
}

# 每个 MODE 偏好的意图 (用于意图选择 bias)
MODE_INTENT_BIAS = {
    "EXPLORE": {
        "CUSTOM": 2.0, "READ": 1.5, "SEARCH": 1.3,
        "INFO": 1.2, "EXPLORE": 1.5, "INSPECT": 1.0,
        "USB_DEVICES": 1.5, "DISK_USAGE": 1.0, "ARCH_INFO": 1.0,
        "LS_TMP": 1.0, "READ_ETC": 1.0, "LIST": 0.8, "COUNT": 0.8,
        "HELP": -2.0,
    },
    "CREATE": {
        "WRITE": 2.0, "APPEND": 1.5, "GENERATE": 2.0,
        "CUSTOM": 1.0, "READ": 0.5, "SEARCH": 0.5,
        "HELP": -2.0,
    },
    "LEARN": {
        "CUSTOM": 1.0, "READ": 1.0, "SEARCH": 1.2,
        "INFO": 1.0, "COUNT": 0.8,
        "HELP": -2.0,
    },
}


class MetaMLP(nn.Module):
    """
    小 MLP 偏置网络 (规则边界模糊时用)
    输入: [n_facts_norm, growth_rate, rnd_avg, schema_coverage, n_gaps_norm, wm_loss_norm, steps_since_switch]
    输出: [EXPLORE, CREATE, LEARN] logits
    """
    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, N_MODES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaCognitiveSelector:
    """
    元认知选择器

    用法:
      selector = MetaCognitiveSelector()
      mode = selector.select({
          "step": step,
          "n_facts": len(facts),
          "n_gaps": len(gaps),
          "schema_coverage": 0.5,
          "rnd_avg": 0.01,
          "wm_loss": 0.5,
          "recent_intents": [...],
          "in_chain": False,
          "has_active_goal": False,
      })
    """

    def __init__(self, min_hold_steps: int = 30, cold_start_steps: int = 100):
        self.current_mode: str = "EXPLORE"
        self.mode_start_step: int = 0
        self.mode_history: list[tuple[int, str, str]] = []  # (step, old_mode, new_mode, reason)
        self.min_hold_steps = min_hold_steps
        self.cold_start_steps = cold_start_steps

        # 小 MLP 偏置 (可选)
        self.mlp: Optional[MetaMLP] = None
        self.mlp_active = False  # 仅在规则边界模糊时启用

    def select(self, stats: dict) -> str:
        """
        选择当前 MODE

        stats 需要包含:
          - step: int
          - n_facts: int
          - n_gaps: int
          - schema_coverage: float (0~1)
          - rnd_avg: float
          - wm_loss: float
          - recent_intents: list[str]
          - in_chain: bool
          - has_active_goal: bool
          - fact_growth_rate: float (最近 10 步的新事实数)
        """
        step = stats.get("step", 0)
        n_facts = stats.get("n_facts", 0)
        n_gaps = stats.get("n_gaps", 0)
        schema_cov = stats.get("schema_coverage", 0.0)
        rnd_avg = stats.get("rnd_avg", 0.0)
        wm_loss = stats.get("wm_loss", 0.0)
        recent = stats.get("recent_intents", [])
        in_chain = stats.get("in_chain", False)
        has_goal = stats.get("has_active_goal", False)
        growth = stats.get("fact_growth_rate", 0.0)

        # ── 最小持有步数: 防止 MODE 振荡 ──
        steps_in_mode = step - self.mode_start_step
        if steps_in_mode < self.min_hold_steps and step >= self.cold_start_steps:
            return self.current_mode

        old_mode = self.current_mode

        # ── 规则优先 ──

        # R1: 冷启动: 前 100 步强制 EXPLORE
        if step < self.cold_start_steps:
            self.current_mode = "EXPLORE"
            self.mode_start_step = step
            return self.current_mode

        # R2: 有活跃链/目标 → 保持当前 MODE
        if in_chain or has_goal:
            return self.current_mode

        # R3: schema 覆盖度低 (< 40%) → EXPLORE
        if schema_cov < 0.4 and n_gaps > 0:
            self.current_mode = "EXPLORE"
            self.mode_start_step = step
            self.mode_history.append((step, old_mode, "EXPLORE", f"schema_cov={schema_cov:.2f} < 0.4"))
            return self.current_mode

        # R4: 事实充足 + 缺口少 → CREATE
        if n_facts >= 8 and n_gaps <= 2:
            # 检查是否最近做过创作
            recent_create = any(i in ("WRITE", "APPEND", "GENERATE") for i in recent[-10:])
            if not recent_create:
                self.current_mode = "CREATE"
                self.mode_start_step = step
                self.mode_history.append((step, old_mode, "CREATE", f"facts={n_facts} gaps={n_gaps}"))
                return self.current_mode

        # R5: WM loss 高 → LEARN (模型需要修正)
        if wm_loss > 1.0 and n_facts >= 5:
            self.current_mode = "LEARN"
            self.mode_start_step = step
            self.mode_history.append((step, old_mode, "LEARN", f"wm_loss={wm_loss:.2f} > 1.0"))
            return self.current_mode

        # R6: 新颖度高 (RND 高) → EXPLORE
        if rnd_avg > 0.05:
            self.current_mode = "EXPLORE"
            self.mode_start_step = step
            self.mode_history.append((step, old_mode, "EXPLORE", f"rnd={rnd_avg:.3f} > 0.05"))
            return self.current_mode

        # R7: 事实增长停滞 → 切换 MODE
        if growth < 0.1 and n_facts >= 5 and self.current_mode == "EXPLORE":
            self.current_mode = "CREATE"
            self.mode_start_step = step
            self.mode_history.append((step, old_mode, "CREATE", f"growth={growth:.2f} stalled"))
            return self.current_mode

        # ── MLP 偏置 (规则不明确时的 fallback) ──
        if self.mlp_active and self.mlp is not None:
            features = torch.tensor([
                min(1.0, n_facts / 40.0),       # n_facts_norm
                min(1.0, growth / 2.0),           # growth_rate
                min(1.0, rnd_avg / 0.1),           # rnd_avg
                schema_cov,                        # schema_coverage
                min(1.0, n_gaps / 10.0),           # n_gaps_norm
                min(1.0, wm_loss / 3.0),           # wm_loss_norm
                min(1.0, steps_in_mode / 100.0),   # steps_since_switch
            ]).unsqueeze(0)
            with torch.no_grad():
                logits = self.mlp(features)
                probs = torch.softmax(logits, dim=-1)
                mode_idx = probs.argmax().item()
                self.current_mode = MODES[mode_idx]
                if self.current_mode != old_mode:
                    self.mode_start_step = step
                    self.mode_history.append((
                        step, old_mode, self.current_mode,
                        f"mlp(wm_loss={wm_loss:.2f},rnd={rnd_avg:.3f})"
                    ))
                return self.current_mode

        # 默认: 保持当前 MODE
        return self.current_mode

    def get_intent_bias(self, mode: Optional[str] = None) -> dict[str, float]:
        """返回当前 MODE 的意图偏好偏置"""
        m = mode or self.current_mode
        return MODE_INTENT_BIAS.get(m, {})

    def get_mode_description(self, mode: Optional[str] = None) -> str:
        m = mode or self.current_mode
        return MODE_DESCRIPTIONS.get(m, "")

    def force_mode(self, mode: str, step: int, reason: str = "explicit"):
        """强制切换 MODE (外部调用)"""
        if mode not in MODES:
            return
        old = self.current_mode
        self.current_mode = mode
        self.mode_start_step = step
        self.mode_history.append((step, old, mode, reason))

    def stats(self) -> dict:
        return {
            "current_mode": self.current_mode,
            "steps_in_mode": 0,  # 由调用者填充
            "mode_history_count": len(self.mode_history),
            "mode_history": self.mode_history[-5:],
            "mlp_active": self.mlp_active,
        }

    def enable_mlp(self, checkpoint: Optional[str] = None):
        """启用 MLP 偏置"""
        self.mlp = MetaMLP()
        if checkpoint:
            try:
                self.mlp.load_state_dict(torch.load(checkpoint, weights_only=True))
            except Exception:
                pass
        self.mlp_active = True

    def save_mlp(self, path: str):
        if self.mlp:
            torch.save(self.mlp.state_dict(), path)

    def train_mlp(self, features: torch.Tensor, mode_labels: torch.Tensor):
        """训练 MLP 偏置 (从历史经验学习)"""
        if self.mlp is None:
            self.mlp = MetaMLP()
            self.mlp_active = True
        self.mlp.train()
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=1e-3)
        optimizer.zero_grad()
        logits = self.mlp(features)
        loss = nn.functional.cross_entropy(logits, mode_labels)
        loss.backward()
        optimizer.step()
        self.mlp.eval()
        return loss.item()
