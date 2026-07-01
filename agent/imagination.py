"""
Imagination Engine — P19 想象与梦境回放

用 WM5.1 从当前状态 rollout 想象轨迹，
把想象经验存入 IntuitionBuffer，让系统能"在脑中预演"。

流程:
    state_emb (384D) → Conductor → thought (16D)
    thought → IntuitionBuffer → intent
    WM5.1.step(state_emb, intent) → next_state_emb, reward, cont
    next_state_emb → Conductor → next_thought
    IntuitionBuffer.store(next_thought, intent, ..., imagined=True)
"""
import torch
import torch.nn.functional as F
from typing import Optional


class ImaginationEngine:
    """
    想象引擎: 从真实状态出发，用世界模型 roll out 未来。

    Args:
        world_model: WorldModelV51 实例
        conductor: Conductor 实例 (必须有 forward_emb 方法)
        buffer: IntuitionBuffer 实例
        max_steps: 每条想象轨迹最多几步
        device: "cpu"
    """
    def __init__(self, world_model, conductor, buffer,
                 max_steps: int = 5, device: str = "cpu"):
        self.wm = world_model
        self.conductor = conductor
        self.buffer = buffer
        self.max_steps = max_steps
        self.device = device
        self.stats = {
            "n_rollouts": 0,
            "total_steps": 0,
            "mean_reward": 0.0,
        }

    def _state_to_thought(self, state_emb: torch.Tensor) -> torch.Tensor:
        """384D 状态嵌入 → 16D thought 向量"""
        if self.conductor is None:
            return torch.zeros(16)
        with torch.no_grad():
            if state_emb.dim() == 1:
                state_emb = state_emb.unsqueeze(0)
            thought, _ = self.conductor.forward_emb(state_emb)
            return thought.squeeze(0)

    def _choose_intent(self, thought: torch.Tensor) -> int:
        """根据直觉缓冲区选意图，默认探索"""
        if self.buffer.size >= 10 and self.conductor is not None:
            probs = self.buffer.direction_prob(thought)
            # 加一点随机性，避免想象死循环
            p = torch.tensor(probs)
            p = p * 0.8 + torch.ones(3) * 0.2 / 3.0
            return int(torch.multinomial(p, 1).item())
        return int(torch.randint(0, 3, (1,)).item())

    def rollout(self, state_emb: torch.Tensor,
                first_intent: Optional[int] = None,
                force_steps: Optional[int] = None) -> dict:
        """
        从当前状态 rollout 一条想象轨迹。

        Args:
            state_emb: (384,) 或 (1,384) MiniLM 状态嵌入
            first_intent: 若指定，第一步用此意图；否则查 buffer
            force_steps: 覆盖 max_steps

        Returns:
            {
                "trajectory": [(thought, intent, reward, cont), ...],
                "n_steps": int,
                "total_reward": float,
                "mean_reward": float,
            }
        """
        if self.conductor is None:
            return {"trajectory": [], "n_steps": 0, "total_reward": 0.0, "mean_reward": 0.0}
        n_steps = force_steps or self.max_steps
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)

        trajectory = []
        total_reward = 0.0

        # WM5.1 在推理时隐藏状态应该独立，不污染真实运行
        hidden = None

        for _ in range(n_steps):
            thought = self._state_to_thought(state_emb)
            intent = first_intent if first_intent is not None and len(trajectory) == 0 else self._choose_intent(thought)

            # 动作编码
            action_emb = self.wm.encode_action(intent, state_emb)

            # 用 prior 预测下一步 (state_target=None)
            with torch.no_grad():
                result = self.wm.core(state_emb, action_emb, hidden, state_target=None)
                hidden = result["hidden"]
                next_state = result["next_state"]
                reward = result["reward"].squeeze().item()
                cont = result["cont"].squeeze().item()

            # 把想象经验存入直觉缓冲区
            next_thought = self._state_to_thought(next_state)
            success = reward > 0.0
            self.buffer.store(
                thought=next_thought,
                intent=intent,
                params={"imagined": True, "rollout_step": len(trajectory)},
                reward=reward,
                success=success,
                surprise=0.0,
                state_emb=next_state.squeeze(0) if next_state.dim() > 1 else next_state,
                imagined=True,
            )

            trajectory.append({
                "thought": next_thought,
                "intent": intent,
                "reward": reward,
                "success": success,
                "cont": cont,
            })

            total_reward += reward
            state_emb = next_state
            first_intent = None  # 只有第一步用给定意图

        self.stats["n_rollouts"] += 1
        self.stats["total_steps"] += len(trajectory)
        self.stats["mean_reward"] = (
            self.stats["mean_reward"] * (self.stats["n_rollouts"] - 1)
            + total_reward / max(len(trajectory), 1)
        ) / max(self.stats["n_rollouts"], 1)

        return {
            "trajectory": trajectory,
            "n_steps": len(trajectory),
            "total_reward": total_reward,
            "mean_reward": total_reward / max(len(trajectory), 1),
        }

    def report(self) -> dict:
        return self.stats.copy()
