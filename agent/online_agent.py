"""
Online Agent - 闭环主循环

编码 → 大脑 → 手 → 执行 → 好奇心 → 学习
"""

import sys
import os
import json
import time
import random
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from sentence_transformers import SentenceTransformer

from agent.state_encoder import StateEncoder
from agent.rnd import RND
from agent.experience import Experience, ExperienceBuffer

# ── 全量执行日志 ──
import datetime
EXEC_LOG_PATH = "exec_log.jsonl"
def _log_execution(intent: str, params: dict, result, output: str, state_text: str, step: int = 0):
    """记录每一步执行的详细信息到 JSONL 文件"""
    try:
        with open(EXEC_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": datetime.datetime.now().isoformat(),
                "step": step,
                "intent": intent,
                "params": {k: str(v) for k, v in params.items()},
                "exit_code": result.exit_code if result else -1,
                "output_len": len(output),
                "output_preview": output[:200],
                "state": state_text[:260],
                "thought_label": state_text.split("思考:")[-1].split(" 事实:")[0].strip() if "思考:" in state_text else "",
            }, ensure_ascii=False) + "\n")
    except:
        pass
from agent.command_selector_v2 import HierarchicalSelector
from agent.command_clusterer import CommandClusterer
from agent.command_miner import CommandMiner
from agent.world_model import WorldModel
from agent.intent_discoverer import IntentDiscoverer
from agent.workbench import Workbench
from benchmark.param_extractor import ParameterExtractor
from benchmark.template_engine import TemplateEngine, ExecResult
from collections import deque


# 意图列表
INTENTS = ["READ", "LIST", "SEARCH", "INFO", "INSPECT", "COUNT", "EXPLORE", "HELP", "READ_ETC", "USB_DEVICES", "DISK_USAGE", "LS_TMP", "ARCH_INFO", "CUSTOM"]
N_INTENTS = 13  # Conductor/分类器输出维度 (不含 CUSTOM)


class IntentClassifier:
    """MiniLM + MLP 意图分类器 (11 类)"""

    def __init__(self, checkpoint: str = "checkpoints/intent_classifier/best_head.pt"):
        import torch.nn as nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        self.encoder.to(self.device)
        self.encoder.eval()

        class MLPHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(384),
                    nn.Linear(384, 128),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 13),
                )
            def forward(self, x):
                return self.net(x)

        self.head = MLPHead()
        try:
            sd = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.head.load_state_dict(sd, strict=False)  # 允许新旧intent数不同
        except Exception:
            print(f"  ⚠️ 分类器checkpoint加载部分失败, 新head随机初始化")
        self.head.to(self.device)
        self.head.eval()

    def predict(self, state_text: str) -> str:
        emb = self.encoder.encode(state_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            logits = self.head(emb)
            pred = logits.argmax().item()
        return INTENTS[pred]

    def predict_logits(self, state_text: str) -> torch.Tensor:
        """返回 logits (用于在线训练的梯度计算)"""
        emb = self.encoder.encode(state_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            logits = self.head(emb)
        return logits

    def get_embedding(self, state_text: str) -> torch.Tensor:
        """返回 MiniLM 嵌入 (用于 RND)"""
        emb_np = self.encoder.encode(state_text, convert_to_numpy=True)
        return torch.from_numpy(emb_np).float().to(self.device)


class OnlineAgent:
    """
    在线 Agent 主循环

    Usage:
      agent = OnlineAgent()
      agent.run(n_steps=100)
    """

    def __init__(
        self,
        classifier_checkpoint: str = "checkpoints/intent_classifier/best_head.pt",
        conductor_checkpoint: str = None,  # None=自动选最佳版本
        buffer_size: int = 5000,
        train_interval: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        novelty_weight: float = 0.3,
        explore_prob: float = 0.1,
        conductor_gate: float = 0.7,  # A/B 切换阈值 (经验证 0.7 最优)
        mode: str = "auto",  # stable | creative | auto
    ):
        # Conductor checkpoint 自动选择: 线上对齐版 > 原始版
        if conductor_checkpoint is None:
            aligned = "checkpoints/conductor/online_aligned.pt"
            original = "checkpoints/conductor/head.pt"
            conductor_checkpoint = aligned if os.path.exists(aligned) else original
        print("初始化 OnlineAgent...")

        # 沙箱 (Docker 容器, 所有命令在隔离环境执行)
        self.sandbox = None
        try:
            from agent.sandbox_executor import SandboxExecutor
            self.sandbox = SandboxExecutor()
            print(f"  ✅ Docker 沙箱就绪")
            # 安装常用工具
            self.sandbox.install_packages([
                "coreutils", "util-linux", "procps",
                "findutils", "grep", "diffutils",
            ])
        except Exception as e:
            print(f"  ⚠️ Docker 沙箱不可用: {e}")
        
        # 核心模块
        self.classifier = IntentClassifier(classifier_checkpoint)
        self.param_extractor = ParameterExtractor()
        self.engine = TemplateEngine(dry_run=False, sandbox=self.sandbox)
        # P4: 工作栏必须早于状态编码器
        self.workbench = Workbench()
        self.state_encoder = StateEncoder(workbench=self.workbench)
        # P5: 尝试从宿主持久化目录恢复工作栏
        loaded = self.workbench.load()
        if loaded > 0:
            print(f"  ✅ 工作栏已恢复: {loaded} 个事实")
        self.rnd = RND(embed_dim=384)
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        # 分层命令选择器 (替代 flat CommandSelector)
        self.clusterer = CommandClusterer()
        self.cmd_miner = CommandMiner(clusterer=self.clusterer, sandbox=self.sandbox)
        self.cmd_selector = HierarchicalSelector()
        
        # 注册所有 cluster
        for name, cmds in self.clusterer.clusters.items():
            if cmds:
                self.cmd_selector.register_cluster(name, cmds)
        self.world_model = WorldModel(embed_dim=384, n_intents=len(INTENTS), thought_dim=16)
        # 尝试加载已有世界模型权重
        wm_ckpt = "checkpoints/world_model/latest.pt"
        if os.path.exists(wm_ckpt):
            try:
                self.world_model.load(wm_ckpt)
                print(f"  ✅ 世界模型已加载: {wm_ckpt}")
            except Exception as e:
                print(f"  ⚠️ 世界模型加载失败: {e}")
        self.intent_discoverer = IntentDiscoverer(min_trajectories=30)

        # 训练配置
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.mode = mode  # stable | creative | auto
        self.novelty_weight = novelty_weight
        self.explore_prob = explore_prob
        self.conductor_gate = conductor_gate  # A/B 切换阈值

        # 创意模式: 降低确定性意图的奖励, 鼓励探索
        self.INTENT_REWARD_CREATIVE = {
            "READ":     0.2,   # 大幅降低 (太容易偷懒)
            "LIST":     0.3,
            "INFO":     0.5,
            "SEARCH":   0.8,
            "COUNT":    0.6,
            "INSPECT":  0.4,
            "EXPLORE":  1.2,   # 探索性意图高奖励
            "CUSTOM":   1.5,   # 自由命令最高奖励
            "USB_DEVICES":1.2,
            "READ_ETC": 0.3,
            "DISK_USAGE":0.6,
            "LS_TMP":   1.0,
            "ARCH_INFO":1.0,
            "HELP":    -0.8,
        }
        self.device = "cpu"

        # 指挥家 + 保姆 (A/B 路径)
        self.conductor_path_active = False
        try:
            from agent.nanny import Nanny
            self.nanny = Nanny(
                engine=self.engine,
                sandbox=self.sandbox,
                conductor_checkpoint=conductor_checkpoint,
            )
            self.conductor_path_active = True
            print(f"  ✅ 保姆就绪 (阈值={conductor_gate})")
        except Exception as e:
            print(f"  ⚠️ 保姆不可用: {e}")
        
        self.ab_stats = {"conductor": 0, "classifier": 0, "conductor_success": 0, "classifier_success": 0, "goal_driven": 0, "goal_driven_success": 0}
        self.multi_cmds_count = 0  # P1: 多命令步数
        self._last_cond_logits = None  # P2: 最近一次 Conductor logits
        self.ab_window = deque(maxlen=100)  # P1.5: 滑动窗口, 记录 (used_conductor, success)

        # 优化器 (只训练分类头)
        self.optimizer = torch.optim.AdamW(
            self.classifier.head.parameters(), lr=lr, weight_decay=1e-4
        )

        # Conductor 在线对齐训练
        if self.conductor_path_active:
            self.conductor_optimizer = torch.optim.AdamW(
                self.nanny.conductor.head.parameters(), lr=5e-6, weight_decay=1e-4
            )
            self.conductor_train_buffer = []  # 累积 (state_text, intent) 一致样本
            # 加载离线训练数据 (用于混合训练防止遗忘)
            self.offline_train_data = self._load_offline_conductor_data()
            print(f"  ✅ Conductor 在线对齐就绪 (离线样本={len(self.offline_train_data)})")
            # P2: REINFORCE 数据
            self.conductor_trajectories = []  # (state_text, intent, reward, logits)
            self.conductor_reward_baseline = {}  # intent → running mean reward
        else:
            self.conductor_train_buffer = []
            self.offline_train_data = []
            self.conductor_trajectories = []
            self.conductor_reward_baseline = {}

        # P4: 持久思考向量 (跨步传递)
        self.persistent_thought = torch.zeros(16)

        # 统计
        self.step_count = 0
        self.total_reward = 0.0
        self.intent_history: list[str] = []
        self.success_count = 0

        # 失败抑制: 同一 intent+params 失败后 N 步内不再尝试
        self._failure_log: dict[str, int] = {}  # key→step_number_last_tried
        self._failure_cooldown = 50  # 步数冷卻

        # P4: 工作栏驱动的目标池
        self.explore_targets = [
            "查看 /etc/hostname 的内容",
            "查看 CPU 信息",
            "查看内存信息",
            "查看磁盘使用情况",
            "列出 / 目录",
            "列出 /etc 目录",
            "统计 /etc/passwd 的行数",
            "在 /etc/passwd 中搜索 root",
            "检查 python3 是否安装",
            "探索 /tmp 目录",
            "获取当前时间",
            "查看当前用户",
            "查看系统信息",
        ]
        self.goal_idx = 0

    def _get_next_goal(self) -> str:
        """获取下一个探索目标 (工作栏 + RND 新颖度 + 轮转)"""
        # P4: 工作栏推荐优先
        if self.workbench:
            follow_up = self.workbench.get_follow_up()
            if follow_up:
                intent_name, _ = follow_up
                return f"验证发现: {intent_name}"
        
        # 如果有缓冲区, 找新颖度最高的目标方向
        if self.buffer.size > 5:
            novel_exps = self.buffer.sample_novel(3)
            for exp in novel_exps:
                if exp.novelty > 0.1:
                    # 只取输出摘要部分, 避免嵌套
                    short = exp.state_text.split("输出:")[-1].strip()[:30] if "输出:" in exp.state_text else exp.state_text[:30]
                    return f"探索系统: {short}"

        # 否则轮转预定目标
        goal = self.explore_targets[self.goal_idx % len(self.explore_targets)]
        self.goal_idx += 1
        return goal

    INTENT_REWARD_BASE = {
        # 有用: 获取系统信息
        "INFO":    1.5,
        "SEARCH":  1.5,
        "COUNT":   1.2,
        "READ":    1.0,
        "LIST":    1.0,
        # 中性
        "EXPLORE": 0.8,
        # 低价值 (容易偷懒, 严重惩罚)
        "INSPECT": 0.1,
        "HELP":   -0.8,
    }

    def _compute_reward(self, result: ExecResult, intent: str, novelty: float,
                        intent_diversity: float = 0.0, chain_bonus: float = 0.0) -> float:
        """计算奖励: 意图价值 + 成功信号 + 新颖度 - 重复惩罚"""
        # 双模式: 根据 recent 多样性自动切换
        if self.mode == "auto":
            recent = self.intent_history[-20:] if self.intent_history else []
            unique_ratio = len(set(recent)) / max(len(recent), 1)
            # 多样性 < 15% 时切创意, > 30% 时切稳定
            use_creative = unique_ratio < 0.15
        elif self.mode == "creative":
            use_creative = True
        else:
            use_creative = False

        reward = 0.0
        success = result.exit_code == 0

        # 意图基础价值 (双模式)
        if use_creative:
            base = self.INTENT_REWARD_CREATIVE.get(intent, 0.6)
        else:
            base = self.INTENT_REWARD_BASE.get(intent, 0.5)

        # 成功信号
        if success and result.stdout and len(result.stdout.strip()) > 3:
            reward += base
            # 丰富输出奖励
            n_lines = len(result.stdout.strip().splitlines())
            if n_lines > 5:
                reward += 0.5
        elif success and result.stderr:
            reward += base * 0.3
        # 新颖度奖励 (仅在真正新颖时)
        # 创意模式: 更低的新颖度阈值 + 更强的好奇心权重
        novelty_threshold = 0.01 if use_creative else 0.05
        novelty_mult = 1.0 if use_creative else self.novelty_weight
        if novelty > novelty_threshold:
            reward += novelty * novelty_mult

        # 创意模式: 强多样性奖励
        diversity_weight = 0.5 if use_creative else 0.3
        reward += intent_diversity * diversity_weight

        # 连续重复惩罚: 同一个意图连续出现 >3 次
        recent = self.intent_history[-4:] if len(self.intent_history) >= 4 else []
        if len(recent) >= 4 and len(set(recent)) == 1:
            if use_creative:
                reward *= 0.2  # 创意模式更狠
            else:
                reward *= 0.5

        # HELP 连续惩罚: 连续 HELP 越多, 惩罚越大
        if intent == "HELP":
            consecutive_help = 0
            for h in reversed(self.intent_history[-10:]):
                if h == "HELP":
                    consecutive_help += 1
                else:
                    break
            if consecutive_help >= 2:
                reward -= 0.5 * min(consecutive_help, 5)  # 累进惩罚

        # 持续非 HELP 奖励 (鼓励探索)
        if intent != "HELP":
            non_help_run = 0
            for h in reversed(self.intent_history[-10:]):
                if h != "HELP":
                    non_help_run += 1
                else:
                    break
            if non_help_run >= 3:
                reward += 0.5 * min(non_help_run / 5.0, 1.0)

        # P4.1: 链完成奖励 (发现→验证 1.5, 验证→扩展 3.0)
        reward += chain_bonus

        return reward

    def _load_offline_conductor_data(self) -> list:
        """加载离线训练数据, 用于 Conductor 混合训练防止遗忘"""
        import json
        data_path = "data/intent_train_v3.jsonl"
        offline = []
        try:
            with open(data_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["intent"] in INTENTS:
                        offline.append((r["state_text"], r["intent"]))
        except:
            pass
        return offline[:1000]  # 取前1000条, 够用

    def _collect_conductor_agreement(self, state_text: str, executed_intent: str):
        """收集 Conductor 和分类器一致的样本"""
        if not self.conductor_path_active:
            return
        if executed_intent in ("HELP", "CUSTOM"):
            return  # HELP/CUSTOM 不适合当训练信号

        try:
            _, cond_logits = self.nanny.think(state_text)
            cond_probs = torch.nn.functional.softmax(cond_logits, dim=-1)
            cond_intent = INTENTS[cond_logits.argmax().item()]
            cond_conf = cond_probs.max().item()

            # 如果 Conductor 和实际执行的 intent 一致, 且置信度还行 → 好样本
            if cond_intent == executed_intent and cond_conf > 0.3:
                self.conductor_train_buffer.append((state_text, executed_intent))
                # 控制缓冲区大小, 保留最近 500 条
                if len(self.conductor_train_buffer) > 500:
                    self.conductor_train_buffer = self.conductor_train_buffer[-500:]
        except Exception:
            pass

    def _train_conductor_online(self):
        """Conductor 在线训练: CE+对比+REINFORCE+entropy"""
        if not self.conductor_path_active:
            return 0.0

        import random
        n_agreement = len(self.conductor_train_buffer)
        n_traj = len(self.conductor_trajectories)
        if n_agreement < 4 and n_traj < 4:
            return 0.0

        self.nanny.conductor.head.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(3):  # 3 mini-batches
            self.conductor_optimizer.zero_grad()
            batch_loss = 0.0

            # ── 1. CE + 对比损失 (一致样本 + 离线混合) ──
            if n_agreement >= 4:
                batch = []
                n_online = min(16, n_agreement)
                batch.extend(random.sample(self.conductor_train_buffer, n_online))
                if self.offline_train_data:
                    n_offline = min(16, len(self.offline_train_data))
                    batch.extend(random.sample(self.offline_train_data, n_offline))

                if len(batch) >= 4:
                    texts = [b[0] for b in batch]
                    labels = [INTENTS.index(b[1]) if b[1] in INTENTS else 0 for b in batch]
                    embs_np = self.nanny.conductor.encoder.encode(texts, convert_to_numpy=True)
                    embs = torch.from_numpy(embs_np).float().to(self.device)
                    labels_t = torch.tensor(labels, device=self.device)
                    thought, logits = self.nanny.conductor.forward_emb(embs)

                    ce_loss = torch.nn.functional.cross_entropy(logits, labels_t)
                    thought_norm = torch.nn.functional.normalize(thought, dim=-1)
                    cos_sim = thought_norm @ thought_norm.T
                    targets_exp = labels_t.unsqueeze(1)
                    same_mask = (targets_exp == targets_exp.T).float()
                    pos_loss = (same_mask * torch.nn.functional.relu(0.8 - cos_sim)).sum() / (same_mask.sum() + 1)
                    neg_loss = ((1 - same_mask) * torch.nn.functional.relu(cos_sim + 0.2)).sum() / ((1 - same_mask).sum() + 1)
                    contrastive_loss = pos_loss + neg_loss
                    batch_loss += ce_loss + 0.05 * contrastive_loss

            # ── 2. REINFORCE + entropy (轨迹数据) ──
            if n_traj >= 4:
                sample = random.sample(self.conductor_trajectories, min(16, n_traj))
                # 过滤掉 CUSTOM (Conductor 不输出 CUSTOM)
                sample = [s for s in sample if s[1] != 'CUSTOM' and s[1] in INTENTS[:N_INTENTS]]
                if len(sample) < 2:
                    continue
                traj_texts = [s[0] for s in sample]
                traj_actions = [INTENTS.index(s[1]) for s in sample]
                traj_rewards = [s[2] for s in sample]

                traj_embs_np = self.nanny.conductor.encoder.encode(traj_texts, convert_to_numpy=True)
                traj_embs = torch.from_numpy(traj_embs_np).float().to(self.device)
                _, traj_logits = self.nanny.conductor.forward_emb(traj_embs)

                # V3: 世界模型作为 Critic — WM价值 + 实际奖励混合
                baselines = [self.conductor_reward_baseline.get(s[1], 0.0) for s in sample]
                # 获取WM价值预测 (从轨迹的状态嵌入)
                try:
                    wm_values = []
                    for s in sample:
                        s_emb = self.classifier.get_embedding(s[0]).clone()
                        i_idx = INTENTS.index(s[1]) if s[1] in INTENTS else 0
                        sim = self.world_model.simulate(s_emb, self.persistent_thought, i_idx)
                        wm_values.append(sim["value"])
                    wm_t = torch.tensor(wm_values)
                except Exception:
                    wm_t = torch.zeros(len(sample))
                
                # Actor-Critic: 混合奖励 (实际0.7 + WM预测0.3)
                advantages = (torch.tensor(traj_rewards) * 0.7 + wm_t * 0.3) - torch.tensor(baselines)
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # P3: Diversity bonus — 最近少用的意图得到额外优势
                diversity_bonus = torch.zeros(len(traj_actions))
                recent_wins = self.intent_history[-30:] if self.intent_history else []
                for i, intent_name in enumerate([s[1] for s in sample]):
                    freq = recent_wins.count(intent_name) / max(len(recent_wins), 1)
                    diversity_bonus[i] = 1.0 - freq  # 0~1: 越少用奖励越高

                # Policy gradient + diversity bonus
                log_probs = torch.nn.functional.log_softmax(traj_logits, dim=-1)
                selected = log_probs[range(len(traj_actions)), traj_actions]
                advantages = advantages + diversity_bonus * 0.3
                pg_loss = -(selected * advantages.detach()).mean()

                # Entropy bonus (鼓励探索)
                probs = torch.nn.functional.softmax(traj_logits, dim=-1)
                entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean()
                batch_loss += pg_loss - 0.01 * entropy

            if batch_loss.item() == 0.0:
                continue

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nanny.conductor.head.parameters(), 1.0)
            self.conductor_optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        self.nanny.conductor.head.eval()

        # 清空缓冲区
        self.conductor_train_buffer.clear()
        self.conductor_trajectories.clear()

        return total_loss / max(n_batches, 1)

    def _select_intent(self, state_text: str) -> str:
        """选择意图: 多样性优先 + 分类器 + 自适应探索"""
        # P5.1: 强多样性调度 — 检查全局意图覆盖
        if len(self.intent_history) >= 30:
            recent_all = self.intent_history[-30:]
            covered = set(recent_all)
            uncovered = [i for i in INTENTS if i not in covered and i not in ("HELP", "CUSTOM")]
            if uncovered:
                # 有从未使用的意图: 强制探索
                return random.choice(uncovered)
        
        # 最近20步单一意图超过35%: 强制转向
        recent = self.intent_history[-20:] if len(self.intent_history) >= 20 else None
        if recent:
            usage = {i: recent.count(i) for i in set(recent)}
            most_used = max(usage, key=usage.get)
            most_used_pct = usage[most_used] / len(recent)
            if most_used_pct > 0.35:
                alternatives = [i for i in INTENTS if i != most_used and i not in ("HELP",)]
                return random.choice(alternatives)
        
        # 偶尔选 CUSTOM (新颖度低时更频繁)
        rnd_stats = self.rnd.get_novelty_stats()
        rnd_avg = rnd_stats.get("running_errors_avg", 0)
        custom_prob = max(0.05, 0.15 - rnd_avg * 5)
        if random.random() < custom_prob:
            return "CUSTOM"
        
        # ε-贪心探索 (跳过 HELP)
        if random.random() < self.explore_prob:
            return random.choice([i for i in INTENTS if i not in ("HELP",)])
        
        intent = self.classifier.predict(state_text)
        
        # HELP 已被禁用: 分类器选到 HELP 时重定向到第二高置信度的非 HELP 意图
        if intent == "HELP":
            emb = self.classifier.get_embedding(state_text)
            with torch.no_grad():
                logits = self.classifier.head(emb)
            alternatives = [i for i in INTENTS if i not in ("HELP", "CUSTOM")]
            sorted_idx = logits.argsort(descending=True).flatten()
            for idx in sorted_idx.tolist():
                alt = INTENTS[idx]
                if alt in alternatives:
                    return alt
            return "INFO"  # 全 fail 兜底
        
        # INSPECT 也做低置信度保护 (保留)
        if intent == "INSPECT":
            emb = self.classifier.get_embedding(state_text)
            with torch.no_grad():
                logits = self.classifier.head(emb)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = probs.max().item()
            if confidence < 0.7:
                alternatives = [i for i in INTENTS if i not in ("HELP", "INSPECT", "CUSTOM")]
                sorted_idx = logits.argsort(descending=True).flatten()
                for idx in sorted_idx.tolist():
                    alt = INTENTS[idx]
                    if alt in alternatives:
                        return alt
        
        return intent

    def step(self) -> tuple[bool, float]:
        """执行一步: A/B 切换 + 指挥家/分类器 → 执行"""
        self.step_count += 1

        # 1. 编码当前状态
        # P4.1: 先无思考标签生成 state_text (给指挥家用)
        state_text = self.state_encoder.get_state_text(thought_label="")
        state_emb = self.classifier.get_embedding(state_text).detach().clone()

        # V3 + P4: 思考向量 — 跨步持续 + 当前指挥家新鲜混合
        thought_vector = self.persistent_thought.clone()
        thought_label = ""
        if self.conductor_path_active:
            try:
                fresh_thought, cond_logits = self.nanny.think(state_text)
                thought_vector = 0.7 * thought_vector + 0.3 * fresh_thought
                # 从 logits 生成可读思考标签
                cond_top = INTENTS[cond_logits.argmax().item()]
                discovery = self.workbench.get_current_discovery()
                if discovery:
                    val = self.workbench.get_fact(discovery) or ""
                    thought_label = f"{cond_top}@{discovery}={val[:20]}"
                else:
                    thought_label = cond_top
            except Exception:
                pass
        # 带思考标签重新生成 state_text
        state_text = self.state_encoder.get_state_text(thought_label=thought_label)
        state_emb = self.classifier.get_embedding(state_text).detach().clone()

        # 2. P4.2: 工作栏驱动目标 (优先于 A/B 切换)
        used_goal = False
        intent = None
        params = {}
        used_conductor = False
        if self.workbench and self.workbench.has_active_goal():
            fu = self.workbench.get_current_goal()
            if fu:
                intent, params = fu
                params = dict(params)
                used_goal = True
                self.ab_stats["goal_driven"] += 1
                cs = self.workbench.chain_step
                label = f"链{cs+1}/3" if cs > 0 else "新发现"
                print(f"  [GOAL] {intent} ({label})")
        elif self.workbench and self.step_count % 3 == 0:
            # P4.3: 好奇心探针 (每3步一次, 不设上限)
            if not hasattr(self, "_probe_find_count"):
                self._probe_find_count = 0
            probe = self.workbench.get_curiosity_probe(self.state_encoder.explored_paths)
            if probe:
                p_args = probe[1].get("custom_args", [])
                p_cmd = " ".join(str(a) for a in p_args) if isinstance(p_args, list) else str(p_args)
                # 仅限制 find 无限重复
                if "find" in p_cmd:
                    self._probe_find_count += 1
                    if self._probe_find_count > 20:
                        probe = None  # find 重复过多, 跳过
                if probe:
                    intent, params = probe
                    params = dict(params)
                    used_goal = True
                    self.ab_stats["goal_driven"] += 1
                    if self.step_count % 6 == 0:
                        print(f"  [PROBE] {params.get('custom_args', ['?'])}")

        if not used_goal:
            # Fallback: A/B 切换 + 指挥家/分类器
            used_conductor = False
            # P1.5: A/B 自适应采样率
            if self.conductor_path_active:
                cond_rate = self.ab_stats["conductor_success"] / max(self.ab_stats["conductor"], 1)
                clf_rate = self.ab_stats["classifier_success"] / max(self.ab_stats["classifier"], 1)
                p_conductor = 0.2 + 0.6 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))
                if self.mode in ("creative", "auto"):
                    recent = self.intent_history[-20:] if self.intent_history else []
                    if recent and len(set(recent)) / max(len(recent), 1) < 0.2:
                        p_conductor *= 0.3
                if random.random() < p_conductor:
                    try:
                        thought, logits = self.nanny.think(state_text)
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        best_prob = probs.max().item()
                        cond_intent = INTENTS[logits.argmax().item()]
                        if best_prob > self.conductor_gate:
                            raw_intent, nanny_params, _ = self.nanny.translate(thought, logits, state_text=state_text)
                            if raw_intent != "HELP":
                                intent = raw_intent
                                params = nanny_params
                                used_conductor = True
                                self.ab_stats["conductor"] += 1
                                self._last_cond_logits = logits.detach().clone()
                        elif cond_intent not in ("HELP", "CUSTOM"):
                            clf_intent = self.classifier.predict(state_text)
                            if cond_intent == clf_intent:
                                raw_intent, nanny_params, _ = self.nanny.translate(thought, logits, state_text=state_text)
                                intent = raw_intent
                                params = nanny_params
                                used_conductor = True
                                self.ab_stats["conductor"] += 1
                                self._last_cond_logits = logits.detach().clone()
                    except:
                        pass

            if not used_conductor:
                intent = self._select_intent(state_text)
                goal = self.state_encoder.current_goal
                if intent == "CUSTOM":
                    cluster, cmd_args = self.cmd_selector.select()
                    params = {"custom_args": cmd_args, "cluster": cluster}
                else:
                    params = self.param_extractor.extract(intent, goal)
                    if "path" not in params and intent in ("READ", "COUNT", "SEARCH"):
                        params["path"] = "/etc/hostname"
                    if "cmd" not in params and intent in ("INSPECT", "HELP"):
                        params["cmd"] = "python3" if intent == "INSPECT" else "ls"
                self.ab_stats["classifier"] += 1

        # V3: 世界模型心理模拟 (目标驱动时跳过, 避免丢失方向)
        if intent is not None and self.step_count > 5 and not used_goal:
            try:
                candidates = [INTENTS.index(intent)]
                for alt in ["CUSTOM", "EXPLORE", "INSPECT", "SEARCH", "LS_TMP", "LIST"]:
                    if alt in INTENTS and INTENTS.index(alt) not in candidates:
                        candidates.append(INTENTS.index(alt))
                
                # 多步搜索 (depth=2): 想象做两步之后的总价值
                rollout_results = self.world_model.rollout_top_k(
                    state_emb, thought_vector,
                    primary_candidates=candidates,
                    secondary_candidates=candidates[:4],
                    depth=2, gamma=0.9
                )
                best = rollout_results[0]
                best_seq = best["sequence"]
                best_name = INTENTS[best_seq[0]]
                
                # 多样性bonus: 最近少用的意图加分
                recent = self.intent_history[-30:] if self.intent_history else []
                for r in rollout_results:
                    seq_name = INTENTS[r["sequence"][0]]
                    freq = recent.count(seq_name) / max(len(recent), 1)
                    r["total_value"] += max(0, 1.0 - freq * 3) * 0.2
                rollout_results.sort(key=lambda r: -r["total_value"])
                best = rollout_results[0]
                best_name = INTENTS[best["sequence"][0]]
                
                # WM推荐的与原始不同且得分更高? 切换
                orig_single = self.world_model.rollout(
                    state_emb, thought_vector, [INTENTS.index(intent)], 0.9
                )
                if best["total_value"] > orig_single["total_value"] * 1.05:
                    intent = best_name
                    if intent == "CUSTOM":
                        cluster, cmd_args = self.cmd_selector.select()
                        params = {"custom_args": cmd_args, "cluster": cluster}
                    else:
                        params = {"depth": random.choice([1, 2, 3])}
            except Exception:
                pass

        # Conductor 一致样本收集 (用于在线对齐训练)
        self._collect_conductor_agreement(state_text, intent)

        # D: 固定概率随机深度 (不再依赖RND新颖度)
        if "depth" not in params:
            roll = random.random()
            if roll < 0.3:
                params["depth"] = 3
            elif roll < 0.6:
                params["depth"] = 2
            else:
                params["depth"] = 1

        # 4. 执行 (多命令组合 P1)
        depth = params.get("depth", 1)
        multi_results = None
        all_exit_ok = False

        if depth > 1 and intent not in ("CUSTOM", "HELP", "EXPLORE"):
            try:
                multi_results = self.engine.execute_multi(intent, params, depth)
            except Exception:
                multi_results = None

        if multi_results and len(multi_results) > 0 and multi_results[0].exit_code != -1:
            # 多命令: 合并输出, 用第一条为主结果
            output_parts = []
            # 成功判定: exit=0 或 intent适用1(未找到/无匹配) 都算有效执行
            ec = multi_results[0].exit_code
            if intent in ("INSPECT", "SEARCH", "EXPLORE", "LS_TMP"):
                all_exit_ok = ec in (0, 1, 127)  # not found / no match 也算
            else:
                all_exit_ok = ec == 0
            for i, r in enumerate(multi_results):
                body = (r.stdout or r.stderr or "").strip()
                if body:
                    output_parts.append(f"--- [{i+1}] ---\n{body}")
            output = "\n".join(output_parts)
            result = multi_results[0]
            self.multi_cmds_count += 1
        else:
            result = self.engine.execute(intent, params)
            output = (result.stdout or result.stderr or "")
        
        # P5.1: 命令名 (用于发现日志)
        if intent == "CUSTOM":
            cmd_name = " ".join(str(a) for a in params.get("custom_args", []))
        else:
            cmd_name = str(params.get("path", params.get("cmd", intent)))

        # 全量日志 (含正确步数)
        _log_execution(intent, params, result, output, state_text, step=self.step_count)
        
        # CUSTOM 回传结果 + 意图发现
        if intent == "CUSTOM":
            custom_args = params.get("custom_args", [])
            cluster_name = params.get("cluster", "UNKNOWN")
            cmd_name = custom_args[0] if custom_args else ""
            # RND 好奇心估计 (用于探索 bonus)
            se = self.classifier.get_embedding(state_text).clone()
            rnd_novelty_est = float(self.rnd.compute_novelty(se))
            self.cmd_selector.record_result(
                cluster=cluster_name,
                cmd=cmd_name,
                success=(result.exit_code == 0),
                novelty=rnd_novelty_est,
                reward=float(result.exit_code == 0),
            )
            # 如果是元命令, 挖掘新命令
            cmd_str = " ".join(custom_args)
            if self.cmd_selector.is_discovery_command(custom_args) and output:
                discovered = self.cmd_miner.mine(output, source=cmd_str)
                for d in discovered:
                    self.cmd_selector.add_command(d["cluster"], d["name"])
                if discovered:
                    print(f"\n  ⛏️ 发现 {len(discovered)} 个新命令!")
                    for d in discovered[:10]:
                        print(f"     {d['name']:20s} → {d['cluster']}")
                    if len(discovered) > 10:
                        print(f"     ... 还有 {len(discovered)-10} 个")
            
            # 记录轨迹用于自动意图发现
            if (result.exit_code == 0) and output:
                self.intent_discoverer.add_custom_trajectory(
                    state_text=state_text,
                    cmd_args=custom_args,
                    output=output,
                    success=True,
                )
                # 检查是否可以发现新意图
                if self.intent_discoverer.ready():
                    new_intents = self.intent_discoverer.discover()
                    if new_intents:
                        print(f"\n  🆕 发现 {len(new_intents)} 个候选意图!")
                        for ni in new_intents:
                            print(f"     {ni['name']:20s} ({ni['n_samples']}条, 例: {ni['cmd_base']})")

        # 5. 更新状态
        cmd_summary = f"{intent} depth={depth}" if depth > 1 else f"{intent} {params}"
        self.state_encoder.update(intent, cmd_summary, output)
        self.state_encoder.set_goal(self._get_next_goal())

        # P5.1: 发现日志 — 写入容器 /tmp (跨步骤可读)
        if self.sandbox and result.exit_code == 0 and output and len(output.strip()) > 10:
            try:
                safe_out = output[:80].replace("'", "").replace("\n", " | ")
                self.sandbox.execute(
                    f"printf '### 步{self.step_count} {intent}\\n{cmd_name}\\n{safe_out}\\n\\n' >> /tmp/discoveries.md"
                )
            except Exception:
                pass

        # P4: 从输出提取事实到工作栏
        cmd_name = str(params.get("custom_args", [intent]))
        if isinstance(cmd_name, list):
            cmd_name = " ".join(cmd_name)
        if result.exit_code == 0:
            self.workbench.extract_facts(intent, cmd_name, output, params, self.step_count)

        # 6. 世界模型好奇心 + RND 新颖度
        next_state_text = self.state_encoder.get_state_text(thought_label=thought_label)

        # V3: 世界模型 — 思考 + 预测 + 直觉
        intent_idx = INTENTS.index(intent) if intent in INTENTS else 0
        world_curiosity = self.world_model.compute_curiosity(
            state_emb, thought_vector, intent_idx, output, result.exit_code
        )
        # 世界模型 update 在 reward 计算之后 (见 step 9a)

        # 6b. RND 新颖度 (fallback, 权重降低)
        rnd_novelty = self.rnd.compute_novelty(state_emb)
        if intent != "CUSTOM":
            self.rnd.update(state_emb)

        # 7. 计算多样性
        recent_intents = self.intent_history[-5:] if self.intent_history else []
        unique_in_last_5 = len(set(recent_intents))
        intent_diversity = unique_in_last_5 / 5.0

        # 8. 综合好奇心 (世界模型为主, RND 为辅)
        # P3: 世界模型好奇心权重提升
        combined_curiosity = world_curiosity * 0.8 + rnd_novelty * 0.2

        # P4.1: 链检测 + 链奖励
        chain_bonus = 0.0
        if self.workbench.check_chain_completed(intent, params):
            chain_bonus = self.workbench.get_chain_bonus()
            if chain_bonus > 0:
                print(f"  [CHAIN] +{chain_bonus:.1f} (step={self.workbench.chain_step})")

        # 9. 计算奖励 (含链奖励)
        reward = self._compute_reward(result, intent, combined_curiosity, intent_diversity,
                                      chain_bonus=chain_bonus)

        # 9a. 更新世界模型 (含真实 reward), 获取预测的 next_thought
        _, predicted_next = self.world_model.update(
            state_emb, thought_vector, intent_idx,
            output, result.exit_code, reward
        )
        # P4: 持久思考 — 混合世界模型预测 + 当前思考
        self.persistent_thought = 0.8 * predicted_next + 0.2 * thought_vector

        # 10. 存储经验 (含嵌入用于世界模型训练)
        self.buffer.add(Experience(
            state_text=state_text,
            intent=intent,
            params=params,
            output=output,
            reward=reward,
            next_state_text=next_state_text,
            novelty=combined_curiosity,
            success=(result.exit_code == 0) and bool(output),
            exit_code=result.exit_code,
            thought=thought_vector.tolist() if thought_vector.numel() > 0 else [],
        ))

        # 9. 记录统计
        self.total_reward += reward
        self.intent_history.append(intent)
        # 多命令: 用 all_exit_ok, 单命令: exit_code==0
        step_success = all_exit_ok if multi_results else (result.exit_code == 0)
        step_success = step_success and bool(output)
        if step_success:
            self.success_count += 1
            if used_goal:
                self.ab_stats["goal_driven_success"] += 1
            elif used_conductor:
                self.ab_stats["conductor_success"] += 1
            else:
                self.ab_stats["classifier_success"] += 1
        else:
            # 失败抑制: 记录失败用于后续回避
            key_parts = [intent]
            for pk in ("cmd", "path", "pattern", "target"):
                if pk in params:
                    key_parts.append(str(params[pk]))
            fail_key = ":".join(key_parts)
            self._failure_log[fail_key] = self.step_count

        # P1.5: 滑动窗口记录
        self.ab_window.append((used_conductor, step_success))

        # P2: 记录 Conductor 轨迹 (用于 REINFORCE)
        if used_conductor and self.conductor_path_active and self._last_cond_logits is not None:
            self.conductor_trajectories.append((
                state_text, intent, reward, self._last_cond_logits.clone()
            ))
            if len(self.conductor_trajectories) > 500:
                self.conductor_trajectories = self.conductor_trajectories[-500:]
            # 更新 running baseline
            old_base = self.conductor_reward_baseline.get(intent, 0.0)
            self.conductor_reward_baseline[intent] = 0.9 * old_base + 0.1 * reward
            self._last_cond_logits = None

        # P5: 持久化工作栏状态 (每10步写一次, 降低IO)
        if self.step_count % 10 == 0:
            self.workbench.save()

        # P5.1: 自引用 — 每30步读自己的发现日志
        if self.sandbox and self.step_count > 0 and self.step_count % 30 == 0:
            try:
                r = self.sandbox.execute("tail -5 /tmp/discoveries.md 2>/dev/null || echo ''")
                if r.stdout and len(r.stdout) > 20:
                    # 把它当普通输出处理, 提取事实
                    self.workbench.extract_facts("SELF", "self-review", r.stdout, {}, self.step_count)
            except Exception:
                pass

        return step_success, reward

    def train_step(self):
        """在线训练: 从缓冲区采样, 微调分类头"""
        if self.buffer.size < self.batch_size:
            return 0.0

        # 分层采样: 按意图类别平衡
        buffer_list = list(self.buffer.buffer)
        by_intent = {intent: [] for intent in INTENTS}
        for e in buffer_list:
            if e.intent in by_intent:
                by_intent[e.intent].append(e)

        # 从每类采样, 排除 HELP (HELP 已经很稳定, 不需要在线训练)
        # 排除 HELP 和 CUSTOM (CUSTOM 是 9 类, 分类器只有 8 个输出)
        train_intents = [i for i in INTENTS if i not in ("HELP", "CUSTOM")]
        batch = []
        samples_per_class = max(1, self.batch_size // len(train_intents))
        for intent in train_intents:
            pool = by_intent.get(intent, [])
            if len(pool) >= samples_per_class:
                batch.extend(random.sample(pool, samples_per_class))
            else:
                batch.extend(pool)

        # 如果不够, 补充高奖励的非 HELP/CUSTOM 样本
        if len(batch) < self.batch_size:
            remaining = self.batch_size - len(batch)
            high_reward = self.buffer.sample_by_reward(remaining * 2, min_reward=0.5)
            # 过滤掉 HELP 和 CUSTOM
            high_reward = [e for e in high_reward if e.intent not in ("HELP", "CUSTOM")]
            batch.extend(high_reward[:remaining])

        random.shuffle(batch)
        batch = batch[:self.batch_size]

        # 准备数据
        texts = [e.state_text for e in batch]
        labels = [INTENTS.index(e.intent) if e.intent in INTENTS else 0 for e in batch]

        # 编码
        embs_np = self.classifier.encoder.encode(texts, convert_to_numpy=True)
        embs = torch.from_numpy(embs_np).float().to(self.device)

        # 训练
        self.classifier.head.train()
        self.optimizer.zero_grad()

        logits = self.classifier.head(embs)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor(labels, device=self.device)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.classifier.head.parameters(), 1.0)
        self.optimizer.step()

        self.classifier.head.eval()

        # V3: 世界模型训练 (思考 + 属性 + 价值)
        wm_samples = []
        for e in batch:
            s_emb = self.classifier.get_embedding(e.state_text)
            s_emb = s_emb.detach().clone() if s_emb.is_inference() else s_emb.clone()
            i_idx = INTENTS.index(e.intent) if e.intent in INTENTS else 0
            # 从输出提取分类目标
            out = e.output[:500] if e.output else ""
            ec = getattr(e, "exit_code", 0)
            exit_cls = 0 if ec == 0 else 1
            length_cls = 0 if len(out) < 100 else (1 if len(out) < 1000 else 2)
            lower = out.lower()
            error_cls = 1 if any(kw in lower for kw in (
                "not found", "error", "denied", "no such file"
            )) else 0
            # 思考向量
            thought_list = getattr(e, "thought", [])
            thought_t = torch.tensor(thought_list[:16] if thought_list else [0]*16)
            # 价值 = 奖励
            val = getattr(e, "reward", 0.0)
            wm_samples.append({
                "state_emb": s_emb,
                "thought": thought_t,
                "intent_id": i_idx,
                "exit_cls": exit_cls,
                "length_cls": length_cls,
                "error_cls": error_cls,
                "value": val,
                "agreement": 1,
            })
        wm_loss = self.world_model.train_on_buffer(wm_samples)

        return loss.item()

    def run(self, n_steps: int = 100, verbose: bool = True):
        """运行主循环"""
        print(f"\n开始闭环运行: {n_steps} 步")
        print(f"  ε-贪心探索: {self.explore_prob}")
        print(f"  新颖度权重: {self.novelty_weight}")
        print(f"  在线训练间隔: 每 {self.train_interval} 步\n")

        for i in range(n_steps):
            success, reward = self.step()

            # 在线训练
            if i > 0 and i % self.train_interval == 0:
                loss = self.train_step()
                cond_loss = self._train_conductor_online()
                if verbose:
                    rnd_stats = self.rnd.get_novelty_stats()
                    cond_info = f"  |  Conductor对齐={cond_loss:.4f}" if cond_loss > 0 else ""
                    print(f"  [{i:4d}] 训练 loss={loss:.4f}  |  "
                          f"新颖度 avg={rnd_stats['running_errors_avg']:.4f}  |  "
                          f"缓冲区 {self.buffer.size}{cond_info}")

            # C: RND 自动软重置 (新颖度持续过低时)
            if i > 0 and i % 100 == 0:
                rnd_stats = self.rnd.get_novelty_stats()
                if rnd_stats['running_errors_avg'] < 0.002:
                    self.rnd.soft_reset(factor=0.3)
                    if verbose:
                        print(f"  [RND] 新颖度过低({rnd_stats['running_errors_avg']:.4f}), 软重置")

            # 每隔 N 步显示统计
            if verbose and (i + 1) % 10 == 0:
                recent_intents = self.intent_history[-10:]
                intent_dist = {intent: recent_intents.count(intent) for intent in set(recent_intents)}
                print(f"  [{i+1:4d}] 奖励={reward:.2f}  "
                      f"累计成功={self.success_count}/{self.step_count}  "
                      f"intent_dist={dict(sorted(intent_dist.items()))}")

        # 最终统计
        return self.summarize()

    def save(self, path: str = "checkpoints/online_agent"):
        """保存分类器 + 经验 + 世界模型"""
        os.makedirs(path, exist_ok=True)

        # 保存分类头
        torch.save(self.classifier.head.state_dict(), f"{path}/classifier_head.pt")

        # 保存经验缓冲
        self.buffer.save(f"{path}/experience.jsonl")

        # V3: 保存世界模型
        os.makedirs("checkpoints/world_model", exist_ok=True)
        self.world_model.save("checkpoints/world_model/latest.pt")

        # 保存统计
        stats = {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "success_count": self.success_count,
            "intent_history": self.intent_history[-500:],
        }
        with open(f"{path}/stats.json", "w") as f:
            json.dump(stats, f)

        print(f"  ✅ 已保存: {path}/ + 世界模型")

    def load(self, path: str = "checkpoints/online_agent"):
        """加载之前训练的模型和经验"""
        ckpt = f"{path}/classifier_head.pt"
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location=self.device, weights_only=True)
            self.classifier.head.load_state_dict(sd)
            self.classifier.head.eval()
            print(f"  ✅ 已加载分类器: {ckpt}")

        exp_path = f"{path}/experience.jsonl"
        if os.path.exists(exp_path):
            self.buffer.load(exp_path)
            print(f"  ✅ 已加载经验: {self.buffer.size} 条")

    def summarize(self) -> dict:
        """返回总统计"""
        intent_dist = {}
        for intent in self.intent_history:
            intent_dist[intent] = intent_dist.get(intent, 0) + 1

        result = {
            "steps": self.step_count,
            "success": self.success_count,
            "success_rate": self.success_count / max(self.step_count, 1),
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(self.step_count, 1),
            "buffer_size": self.buffer.size,
            "intent_distribution": {
                k: v / max(self.step_count, 1) for k, v in intent_dist.items()
            },
            "rnd_stats": self.rnd.get_novelty_stats(),
            "multi_cmds": self.multi_cmds_count,
        }

        print(f"\n{'=' * 45}")
        print(f"  闭环运行完成")
        print(f"  步数: {result['steps']}")
        print(f"  成功率: {result['success_rate']:.1%} ({result['success']}/{result['steps']})")
        print(f"  总奖励: {result['total_reward']:.2f}")
        print(f"  经验缓冲: {result['buffer_size']}")
        print(f"  意图分布: {result['intent_distribution']}")
        rnd_s = result['rnd_stats']
        print(f"  RND 新颖度均值: {rnd_s['running_errors_avg']:.4f}")
        a = self.ab_stats
        cond_rate = a['conductor_success'] / max(a['conductor'], 1) * 100
        clf_rate = a['classifier_success'] / max(a['classifier'], 1) * 100
        print(f"  A/B: 指挥家={a['conductor']}次 ({cond_rate:.0f}%)  分类器={a['classifier']}次 ({clf_rate:.0f}%)")
        if a.get('goal_driven', 0) > 0:
            print(f"  目标驱动: {a['goal_driven']}次 ({a['goal_driven']/max(result['steps'],1)*100:.0f}%)")
        if self.multi_cmds_count > 0:
            print(f"  多命令步数: {self.multi_cmds_count}/{result['steps']} ({self.multi_cmds_count/result['steps']*100:.0f}%)")

        # P1.5: 自适应采样率
        cond_rate = self.ab_stats["conductor_success"] / max(self.ab_stats["conductor"], 1)
        clf_rate = self.ab_stats["classifier_success"] / max(self.ab_stats["classifier"], 1)
        p_cond = 0.2 + 0.6 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))
        print(f"  A/B 自适应: p_conductor={p_cond:.0%}  (Conductor胜率={cond_rate:.0%} vs 分类器胜率={clf_rate:.0%})")
        print(f"{'=' * 45}")

        return result
