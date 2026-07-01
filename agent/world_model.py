"""
世界模型 V3 — 思考 + 预测 + 直觉

架构:
  状态(384) + 思考向量(16) + 意图(N)
    ↓ 共享层
  ┌─ exit_head     → 成功/失败 (2)
  ├─ length_head   → 输出长度 (3)
  ├─ error_head    → 错误信号 (2)
  ├─ value_head    → 价值预测 (1)   ← 预测这次行动的奖励
  ├─ next_thought  → 下步思考 (16)  ← 预测行动后的思维状态
  └─ agreement_head→ 自洽性直觉 (2) ← 模型自我一致性

用法:
  wm = WorldModel()
  # 执行前: 心理模拟
  scores = wm.imagine_top_k(state_emb, thought, [0,1,2])
  best_intent = scores.argmax()
  # 执行后: 好奇心
  curiosity = wm.compute_curiosity(state_emb, thought, intent_idx, output, exit_code, reward)
  wm.update(state_emb, thought, intent_idx, output, exit_code, reward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WorldModelNetV3(nn.Module):
    """世界模型 V3 网络"""

    def __init__(self, embed_dim: int = 384, n_intents: int = 14,
                 thought_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        input_dim = embed_dim + thought_dim + n_intents
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # 输出属性头
        self.exit_head = nn.Linear(hidden_dim, 2)
        self.length_head = nn.Linear(hidden_dim, 3)
        self.error_head = nn.Linear(hidden_dim, 2)
        # 价值头 (回归: 预测 reward)
        self.value_head = nn.Linear(hidden_dim, 1)
        # 下步思考头 (回归: 预测 next thought vector)
        self.next_thought_head = nn.Linear(hidden_dim, thought_dim)
        # 直觉头 (自洽性: 这个组合合理吗?)
        self.agreement_head = nn.Linear(hidden_dim, 2)

    def forward(self, state_emb: torch.Tensor, thought: torch.Tensor,
                intent_onehot: torch.Tensor):
        x = torch.cat([state_emb, thought, intent_onehot], dim=-1)
        h = self.shared(x)
        return {
            "exit": self.exit_head(h),
            "length": self.length_head(h),
            "error": self.error_head(h),
            "value": self.value_head(h).squeeze(-1),
            "next_thought": self.next_thought_head(h),
            "agreement": self.agreement_head(h),
        }


class WorldModel:
    """
    世界模型 V3 — 集思考、预测、直觉
    
    Usage:
      wm = WorldModel()
      
      # 心理模拟: 遍历候选意图
      for intent_idx in candidates:
          pred = wm.simulate(state_emb, thought, intent_idx)
          score = pred["value"] + wm_curiosity * pred["uncertainty"]
      
      # 执行后学习
      curiosity = wm.compute_curiosity(state_emb, thought, intent_idx, ...)
      wm.update(state_emb, thought, intent_idx, ...)
    """

    def __init__(self, embed_dim: int = 384, n_intents: int = 14,
                 thought_dim: int = 16, lr: float = 3e-4, device: str = "cpu"):
        self.device = device
        self.n_intents = n_intents
        self.thought_dim = thought_dim
        self.predictor = WorldModelNetV3(
            embed_dim, n_intents=n_intents, thought_dim=thought_dim
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(), lr=lr, weight_decay=1e-4
        )
        # 误差追踪 (自适应好奇心归一化)
        self.running_errors: list[float] = []
        self.max_error = 1.0

    # P6.4: 扩展意图数
    def expand_intents(self, new_count: int):
        """P6.4: 扩展世界模型以容纳新意图
        Args:
            new_count: 新的总意图数 (含CUSTOM)
        """
        if new_count <= self.n_intents:
            return

        old_first = self.predictor.shared[0]
        old_in = old_first.in_features
        new_in = old_in + (new_count - self.n_intents)

        old_weight = old_first.weight.data   # (128, old_in)
        old_bias = old_first.bias.data       # (128,)

        new_first = nn.Linear(new_in, old_first.out_features).to(self.device)
        new_first.weight.data[:, :old_in] = old_weight
        new_first.bias.data = old_bias
        # 新意图对应的输入列: 零初始化 (不影响已有预测)
        nn.init.zeros_(new_first.weight.data[:, old_in:])

        self.predictor.shared[0] = new_first
        self.n_intents = new_count
        print(f"  [WORLD_MODEL] 意图数扩展: 输入 {old_in} → {new_in}, n_intents={new_count}")

    # ── 工具方法 ──

    def _intent_to_onehot(self, intent_idx: int) -> torch.Tensor:
        onehot = torch.zeros(1, self.n_intents, device=self.device)
        onehot[0, intent_idx] = 1.0
        return onehot

    def _ensure_2d(self, *tensors):
        return [t.unsqueeze(0) if t.dim() == 1 else t for t in tensors]

    def _extract_targets(self, output_text: str, exit_code: int,
                         reward: float = 0.0,
                         next_thought: Optional[torch.Tensor] = None,
                         agreement: int = 1) -> dict:
        """从实际经验提取训练目标"""
        # exit_code class
        exit_cls = 0 if exit_code == 0 else 1
        # length bucket: 0=短(<100), 1=中(100-1000), 2=长(>1000)
        n = len(output_text)
        length_cls = 0 if n < 100 else (1 if n < 1000 else 2)
        # error keywords
        lower = output_text.lower()
        has_error = 1 if any(kw in lower for kw in (
            "not found", "error", "denied", "no such file", "not installed",
            "timed out", "permission denied"
        )) else 0
        
        targets = {
            "exit": torch.tensor([exit_cls], device=self.device),
            "length": torch.tensor([length_cls], device=self.device),
            "error": torch.tensor([has_error], device=self.device),
            "value": torch.tensor([reward], device=self.device),
        }
        if next_thought is not None:
            targets["next_thought"] = next_thought.clone().detach()
        targets["agreement"] = torch.tensor([agreement], device=self.device)
        return targets

    def _compute_loss(self, pred: dict, targets: dict) -> torch.Tensor:
        """多任务损失"""
        loss = (
            F.cross_entropy(pred["exit"], targets["exit"])
            + F.cross_entropy(pred["length"], targets["length"])
            + F.cross_entropy(pred["error"], targets["error"])
        ) / 3.0
        
        if "value" in targets:
            loss += F.mse_loss(pred["value"], targets["value"])
        if "next_thought" in targets:
            loss += F.mse_loss(pred["next_thought"], targets["next_thought"]) * 0.1
        if "agreement" in targets:
            loss += F.cross_entropy(pred["agreement"], targets["agreement"]) * 0.3
        
        return loss

    # ── 核心 API ──

    @torch.no_grad()
    def simulate(self, state_emb: torch.Tensor, thought: torch.Tensor,
                 intent_idx: int) -> dict:
        """
        心理模拟: 给定状态和思考, 预测执行某意图的结果
        
        Returns:
          { "exit_probs", "value", "next_thought", "agreement_prob",
            "uncertainty" (预测熵), "score" (综合得分) }
        """
        self.predictor.eval()
        state_emb, thought = self._ensure_2d(state_emb, thought)
        intent_onehot = self._intent_to_onehot(intent_idx)
        
        pred = self.predictor(state_emb, thought, intent_onehot)
        
        exit_probs = F.softmax(pred["exit"], dim=-1)
        value = pred["value"].item()
        next_t = pred["next_thought"]
        agree_prob = F.softmax(pred["agreement"], dim=-1)[0, 1].item()
        
        # 不确定性 = 预测的熵 (越高 = 越不确定 = 越值得探索)
        exit_entropy = -(exit_probs * exit_probs.clamp(min=1e-8).log()).sum(dim=-1).item()
        uncertainty = exit_entropy / 0.693  # normalize by ln(2)
        
        # 综合得分 = 预测价值 + 不确定性bonus
        score = value + 0.2 * uncertainty
        
        return {
            "exit_probs": exit_probs[0].tolist(),
            "value": value,
            "next_thought": next_t,
            "agreement_prob": agree_prob,
            "uncertainty": uncertainty,
            "score": score,
        }

    @torch.no_grad()
    def imagine_top_k(self, state_emb: torch.Tensor, thought: torch.Tensor,
                       intent_candidates: list[int]) -> list[dict]:
        """对候选意图列表做心理模拟, 返回排序后的结果"""
        results = []
        for idx in intent_candidates:
            r = self.simulate(state_emb, thought, idx)
            r["intent_idx"] = idx
            results.append(r)
        results.sort(key=lambda r: -r["score"])
        return results

    @torch.no_grad()
    def compute_curiosity(self, state_emb: torch.Tensor,
                          thought: torch.Tensor,
                          intent_idx: int,
                          output_text: str = "",
                          exit_code: int = 0) -> float:
        """
        好奇心 = 属性预测误差 (仅 exit/length/error, 不含 value)
        世界模型猜对了 → 低好奇心
        世界模型猜错了 → 高好奇心
        """
        self.predictor.eval()
        state_emb, thought = self._ensure_2d(state_emb, thought)
        intent_onehot = self._intent_to_onehot(intent_idx)
        
        pred = self.predictor(state_emb, thought, intent_onehot)
        targets = self._extract_targets(output_text, exit_code)
        
        # 好奇心仅用属性预测误差 (不含 value/next_thought)
        error = (
            F.cross_entropy(pred["exit"], targets["exit"])
            + F.cross_entropy(pred["length"], targets["length"])
            + F.cross_entropy(pred["error"], targets["error"])
        ).item() / 3.0
        
        # 自适应归一化
        self.running_errors.append(error)
        if len(self.running_errors) > 50:
            self.running_errors.pop(0)
            self.max_error = max(self.running_errors) + 1e-8
        
        return min(error / self.max_error, 1.0)

    def update(self, state_emb: torch.Tensor, thought: torch.Tensor,
               intent_idx: int, output_text: str,
               exit_code: int, reward: float = 0.0,
               next_thought: Optional[torch.Tensor] = None) -> tuple[float, Optional[torch.Tensor]]:
        """从实际经验更新世界模型, 返回 (loss, 预测的next_thought)"""
        self.predictor.train()
        self.optimizer.zero_grad()

        state_emb, thought = self._ensure_2d(state_emb, thought)
        intent_onehot = self._intent_to_onehot(intent_idx)
        
        pred = self.predictor(state_emb, thought, intent_onehot)
        targets = self._extract_targets(output_text, exit_code, reward, next_thought)
        
        loss = self._compute_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        self.predictor.eval()
        
        # P4: 同时返回预测的 next_thought (用于跨步持久化)
        predicted_next = pred["next_thought"].squeeze(0).detach().clone()
        return loss.item(), predicted_next

    def train_on_buffer(self, samples: list) -> float:
        """从缓冲区批量训练"""
        if len(samples) < 4:
            return 0.0

        states, vecs, onehots = [], [], []
        exit_t, len_t, err_t, val_t = [], [], [], []
        next_t, agree_t = [], []

        for s in samples:
            states.append(s["state_emb"])
            vecs.append(s["thought"])
            oh = torch.zeros(self.n_intents)
            idx = s["intent_id"]
            oh[idx if idx < self.n_intents else 0] = 1.0
            onehots.append(oh)
            exit_t.append(s["exit_cls"])
            len_t.append(s["length_cls"])
            err_t.append(s["error_cls"])
            val_t.append(s.get("value", 0.0))
            if "next_thought" in s:
                next_t.append(s["next_thought"])
            agree_t.append(s.get("agreement", 1))

        s_t = torch.stack(states).to(self.device)
        v_t = torch.stack(vecs).to(self.device)
        o_t = torch.stack(onehots).to(self.device)

        self.predictor.train()
        self.optimizer.zero_grad()

        pred = self.predictor(s_t, v_t, o_t)

        loss = (
            F.cross_entropy(pred["exit"], torch.tensor(exit_t, device=self.device))
            + F.cross_entropy(pred["length"], torch.tensor(len_t, device=self.device))
            + F.cross_entropy(pred["error"], torch.tensor(err_t, device=self.device))
        ) / 3.0
        loss += F.mse_loss(pred["value"].squeeze(-1),
                           torch.tensor(val_t, device=self.device))
        if next_t:
            loss += F.mse_loss(pred["next_thought"],
                               torch.stack(next_t).to(self.device)) * 0.1
        loss += F.cross_entropy(pred["agreement"],
                                torch.tensor(agree_t, device=self.device)) * 0.3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        self.predictor.eval()
        return loss.item()

    # ── 持久化 (A) ──

    def save(self, path: str):
        """保存世界模型权重"""
        torch.save({
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "running_errors": self.running_errors,
            "max_error": self.max_error,
        }, path)

    def load(self, path: str):
        """加载世界模型权重 (P9.6: 自动扩展维度)"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        ckpt_sd = ckpt["predictor"]
        # P9.6: 检查第一层维度并自动扩展
        old_first = self.predictor.shared[0]
        ckpt_first_w = ckpt_sd.get('shared.0.weight')
        expanded = False
        if ckpt_first_w is not None and ckpt_first_w.size(1) < old_first.weight.size(1):
            old_in = ckpt_first_w.size(1)
            new_in = old_first.weight.size(1)
            # 复制旧权重, 新列零初始化
            old_first.weight.data[:, :old_in] = ckpt_first_w[:, :old_in]
            old_first.bias.data = ckpt_sd['shared.0.bias']
            # 删除已处理的键, 其余正常加载
            del ckpt_sd['shared.0.weight']
            del ckpt_sd['shared.0.bias']
            expanded = True
        self.predictor.load_state_dict(ckpt_sd, strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        # P9.6: 扩展优化器状态 (Adam momentum 缓冲区)
        if expanded:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p is old_first.weight:
                        state = self.optimizer.state.get(p)
                        if state:
                            for key in ('exp_avg', 'exp_avg_sq'):
                                old_t = state[key]
                                new_t = torch.zeros_like(p)
                                new_t[:, :old_in] = old_t[:, :old_in]
                                state[key] = new_t
        self.running_errors = ckpt.get("running_errors", [])
        self.max_error = ckpt.get("max_error", 1.0)
        self.predictor.eval()

    # ── 多步心理模拟 (B) ──

    @torch.no_grad()
    def rollout(self, state_emb: torch.Tensor, thought: torch.Tensor,
                intent_sequence: list[int], gamma: float = 0.9) -> dict:
        """
        多步心理模拟: 想象执行一系列动作后的累积价值

        Args:
            state_emb: 当前状态嵌入
            thought: 当前思考向量
            intent_sequence: 意图序列 [intent_1, intent_2, ...]
            gamma: 折扣因子

        Returns:
            {"total_value": 折扣累积价值,
             "step_values": 每一步的价值,
             "final_thought": 最后一步后的预测思考}
        """
        self.predictor.eval()
        s = state_emb.clone()
        t = thought.clone()
        total = 0.0
        step_values = []

        for step, intent_idx in enumerate(intent_sequence):
            s_2d, t_2d = self._ensure_2d(s, t)
            oh = self._intent_to_onehot(intent_idx)
            pred = self.predictor(s_2d, t_2d, oh)

            value = pred["value"].item()
            next_t = pred["next_thought"]

            discounted = value * (gamma ** step)
            total += discounted
            step_values.append({"step": step, "value": value, "discounted": discounted})

            # 把预测的 next_thought 作为下一步的输入
            t = next_t.squeeze(0)

        return {
            "total_value": total,
            "step_values": step_values,
            "final_thought": t,
        }

    @torch.no_grad()
    def rollout_top_k(self, state_emb: torch.Tensor, thought: torch.Tensor,
                       primary_candidates: list[int],
                       secondary_candidates: list[int] | None = None,
                       depth: int = 2, gamma: float = 0.9) -> list[dict]:
        """
        多步搜索: 对每个主候选意图, 尝试所有可能的第二步意图

        Args:
            primary_candidates: 第一步的候选意图
            secondary_candidates: 第二步的候选 (默认=primary)
            depth: 搜索深度 (1=单步, 2=两步)
            gamma: 折扣因子

        Returns:
            [(sequence, total_value, details), ...] 按总分降序
        """
        if secondary_candidates is None:
            secondary_candidates = primary_candidates

        results = []

        for p in primary_candidates:
            if depth >= 2:
                for s in secondary_candidates:
                    seq = [p, s]
                    r = self.rollout(state_emb, thought, seq, gamma)
                    results.append({
                        "sequence": seq,
                        "total_value": r["total_value"],
                        "step_values": r["step_values"],
                    })
            else:
                r = self.rollout(state_emb, thought, [p], gamma)
                results.append({
                    "sequence": [p],
                    "total_value": r["total_value"],
                    "step_values": r["step_values"],
                })

        results.sort(key=lambda r: -r["total_value"])
        return results

    def stats(self) -> dict:
        return {
            "running_errors_avg": (sum(self.running_errors) / len(self.running_errors)
                                   if self.running_errors else 0),
            "max_error": self.max_error,
            "n_samples": len(self.running_errors),
            "thought_dim": self.thought_dim,
            "n_intents": self.n_intents,
        }
