"""
Stream Core — 意识流循环核心

持续运行的循环神经网络，内置世界模型 + 好奇心门。

使用方式：
    core = StreamCore(vocab_size=128, hidden_dim=256, action_dim=16)
    h = core.init_state()
    for step in range(100):
        h, outputs = core.step(h, obs_token=obs, action=action)
        # outputs: next_obs_logits, action_logits, curiosity, thought_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamCore(nn.Module):
    """持续意识流核心"""

    def __init__(
        self,
        vocab_size: int = 128,       # 观察值词表大小
        embed_dim: int = 64,         # 嵌入维度
        hidden_dim: int = 256,       # 隐藏状态维度
        action_dim: int = 16,        # 动作空间大小
        thought_tokens: int = 4,     # 每次思考生成的 token 数
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.thought_tokens = thought_tokens
        self.vocab_size = vocab_size

        # 观察值嵌入
        self.obs_embed = nn.Embedding(vocab_size, embed_dim)

        # 动作嵌入
        self.action_embed = nn.Embedding(action_dim, embed_dim // 4)

        # 循环核心 — 使用 GRUCell 作为持续状态
        self.rnn = nn.GRUCell(
            input_size=embed_dim + embed_dim // 4,  # obs + action
            hidden_size=hidden_dim,
            bias=True,
        )

        # 世界模型头 — 预测下一个观察值
        self.world_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

        # 动作头 — 生成下一步动作
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # 好奇心门 — 预测误差的门控信号
        self.curiosity_gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 思考头 — 生成内心语言（可选）
        self.thought_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, vocab_size),
        )

        # 层归一化稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim)

    def step(
        self,
        h: torch.Tensor,
        obs_token: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        return_thoughts: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        单步前向传播

        参数:
            h: 上一时间步的隐藏状态 (batch, hidden_dim)
            obs_token: 当前观察值 token (batch,) 或 (batch, 1)
            prev_action: 上一时间步的动作 (batch,) 或 None
            return_thoughts: 是否生成内心语言

        返回:
            h_new: 更新后的隐藏状态
            outputs: {
                'next_obs_logits': (batch, vocab_size) 世界模型预测
                'action_logits':   (batch, action_dim)  动作概率
                'curiosity':       (batch, 1)            好奇心门值
                'thought_logits':  (batch, vocab_size)   思考内容
            }
        """
        # 确保输入的形状
        if obs_token.dim() == 0:
            obs_token = obs_token.unsqueeze(0)
        if obs_token.dim() == 1:
            obs_token = obs_token.unsqueeze(-1)
        if prev_action is not None and prev_action.dim() == 0:
            prev_action = prev_action.unsqueeze(0)

        batch = obs_token.size(0)

        # 观察值嵌入
        obs_emb = self.obs_embed(obs_token)  # (batch, 1, embed_dim)
        obs_emb = obs_emb.squeeze(1)  # (batch, embed_dim)

        # 动作嵌入（如果没有前一动作，用零向量）
        if prev_action is not None:
            act_emb = self.action_embed(prev_action)  # (batch, embed_dim//4)
        else:
            act_emb = torch.zeros(batch, self.action_embed.embedding_dim)

        # RNN 输入：观察值 + 动作
        rnn_input = torch.cat([obs_emb, act_emb], dim=-1)  # (batch, embed_dim + act_emb)

        # 更新隐藏状态
        h_new = self.rnn(rnn_input, h)
        h_new = self.layer_norm(h_new)

        # 世界模型预测下一个观察值
        next_obs_logits = self.world_head(h_new)  # (batch, vocab_size)

        # 动作生成
        action_logits = self.action_head(h_new)  # (batch, action_dim)

        # 好奇心门 — 基于当前的隐藏状态
        curiosity = self.curiosity_gate(h_new)  # (batch, 1)

        outputs = {
            "next_obs_logits": next_obs_logits,
            "action_logits": action_logits,
            "curiosity": curiosity,
        }

        # 思考生成（内心语言）
        if return_thoughts:
            thought_logits = self.thought_head(h_new)
            outputs["thought_logits"] = thought_logits

        return h_new, outputs


class CuriosityLoss:
    """好奇心驱动的损失函数

    核心思想：
    - world_model_loss: 模型想准确预测环境
    - curiosity_bonus: 模型被高预测误差的状态吸引（好奇心）
    - action_entropy: 模型保持行动多样性

    损失 = world_loss - α × curiosity_gate × world_loss.detach() + β × entropy
    """

    def __init__(
        self,
        curiosity_alpha: float = 0.1,   # 好奇心强度
        entropy_beta: float = 0.05,      # 多样性系数
        thought_gamma: float = 0.01,     # 思考损失权重
    ):
        self.alpha = curiosity_alpha
        self.beta = entropy_beta
        self.gamma = thought_gamma

    def __call__(
        self,
        outputs: dict,
        target_obs: torch.Tensor,
        target_action: torch.Tensor | None = None,
        target_thought: torch.Tensor | None = None,
    ) -> dict:
        """
        计算损失

        好奇心机制：
        - 世界模型损失：训练模型准确预测环境
        - 好奇心门损失：训练好奇心门预测预测误差（="意外程度"）
          好奇心门是一个&quot;惊讶检测器&quot;——它学会识别自己不理解的事务
        - 动作熵：保持行动多样性

        好奇心门 vs 好奇心奖励：
        好奇心门是独立训练的（目标 = sigmoid(预测误差)），
        不影响世界模型的学习。
        模型越准确 → 预测误差越低 → 好奇心门关闭
        遇到新奇事务 → 预测误差高 → 好奇心门打开 → 模型更新参数
        """

        # 1. 世界模型损失：预测准确性
        world_loss = F.cross_entropy(
            outputs["next_obs_logits"],
            target_obs,
            reduction="mean",
        )

        # 2. 好奇心门损失：训练好奇心门预测预测误差
        #    目标 = sigmoid(预测误差)，高误差 = 高好奇心
        with torch.no_grad():
            # 归一化预测误差到 0-1 范围
            curiosity_target = torch.sigmoid(world_loss.detach() - 2.0)
        curiosity_loss = F.mse_loss(
            outputs["curiosity"].squeeze(-1),
            curiosity_target.expand(outputs["curiosity"].size(0)),
        )

        # 3. 动作熵：保持多样性
        action_probs = F.softmax(outputs["action_logits"], dim=-1)
        action_entropy = -(
            action_probs * torch.log(action_probs + 1e-10)
        ).sum(dim=-1).mean()

        total = world_loss + self.alpha * curiosity_loss - self.beta * action_entropy

        return {
            "total": total,
            "world": world_loss,
            "curiosity_loss": curiosity_loss,
            "entropy": action_entropy,
        }
