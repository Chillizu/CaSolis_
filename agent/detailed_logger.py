"""
P9.0: 超级日志记录器 — 每步状态 + 训练指标 + 张量统计

记录到 run_logs/ 目录, 每个运行一个 JSONL 文件。
用于分析训练稳定性、模型坍缩、探索模式。
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Optional


class DetailedLogger:
    """
    每步详细日志记录

    记录内容:
     - 步骤编号、时间戳
     - 意图、参数、来源(conductor/classifier/probe/...)
     - 命令、输出(截断到200字符)、exit_code
     - 奖励、新颖度、多样性
     - A/B 统计
     - 工作栏事实数量
     - RND 状态
     - 训练 loss (如训练触发)
     - 检查点: 分类器、指挥家、世界模型 logits/thought 摘要
    """

    def __init__(self, log_dir: str = "run_logs"):
        os.makedirs(log_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"run_{run_id}.jsonl")
        self._file = open(self.path, "w", encoding="utf-8")
        self._write({"event": "init", "timestamp": time.time()})
        print(f"  [LOGGER] {self.path}")

    def _write(self, data: dict):
        try:
            self._file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except Exception:
            pass

    def log_step(self, *, step: int, intent: str, params: dict,
                 source: str, cmd_name: str = "", output: str = "",
                 exit_code: int, reward: float, novelty: float,
                 diversity: float, conductor_prob: float,
                 facts_before: int, facts_after: int,
                 ab_stats: dict, rnd_state: dict,
                 cond_logits_summary: Optional[dict] = None,
                 wm_pred_summary: Optional[dict] = None,
                 train_loss: Optional[float] = None,
                 recovery_action: str = ""):
        """记录一步的完整状态"""
        entry = {
            "event": "step",
            "step": step,
            "time": time.time(),
            "intent": intent,
            "params": {k: str(v)[:40] for k, v in params.items()},
            "source": source,
            "cmd": cmd_name[:80],
            "output_preview": output[:200].replace("\n", "\\n"),
            "exit_code": exit_code,
            "success": exit_code == 0 and bool(output),
            "reward": round(reward, 4),
            "novelty": round(novelty, 4),
            "diversity": round(diversity, 4),
            "p_conductor": round(conductor_prob, 4),
            "facts_before": facts_before,
            "facts_delta": facts_after - facts_before,
            "ab_stats": dict(ab_stats),
            "rnd": {
                "avg": round(rnd_state.get("running_errors_avg", 0), 6),
                "max": round(rnd_state.get("current_max_error", 0), 6),
            },
            "recovery": recovery_action,
        }
        if cond_logits_summary:
            entry["conductor"] = cond_logits_summary
        if wm_pred_summary:
            entry["world_model"] = wm_pred_summary
        if train_loss is not None:
            entry["train_loss"] = round(train_loss, 6)
        self._write(entry)

    def log_training(self, *, step: int, loss: float, lr: float,
                     intent_counts: dict, buffer_size: int,
                     ucb_weights: Optional[list] = None,
                     n_cond: int = 0, n_clf: int = 0):
        """记录训练事件"""
        entry = {
            "event": "train",
            "step": step,
            "time": time.time(),
            "loss": round(loss, 6),
            "lr": lr,
            "intent_counts": intent_counts,
            "buffer_size": buffer_size,
            "n_cond": n_cond,
            "n_clf": n_clf,
        }
        if ucb_weights:
            entry["ucb_weights"] = [round(w, 4) for w in ucb_weights[:14]]
        self._write(entry)

    def log_snapshot(self, *, step: int, success_rate: float,
                     intent_dist: dict, facts: list[str]):
        """检查点: 阶段性快照"""
        self._write({
            "event": "snapshot",
            "step": step,
            "time": time.time(),
            "success_rate": round(success_rate, 4),
            "intent_distribution": intent_dist,
            "fact_keys": facts,
        })

    def log_alert(self, *, step: int, level: str, message: str,
                  metrics: Optional[dict] = None):
        """异常告警: 坍缩、震荡、异常"""
        entry = {
            "event": "alert",
            "step": step,
            "time": time.time(),
            "level": level,
            "message": message,
        }
        if metrics:
            entry["metrics"] = metrics
        self._write(entry)

    def close(self):
        self._write({"event": "close", "timestamp": time.time()})
        self._file.close()
        print(f"  [LOGGER] 日志已保存: {self.path}")
