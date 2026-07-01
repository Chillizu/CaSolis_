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
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from agent.state_encoder import StateEncoder
from agent.rnd import RND
from agent.detailed_logger import DetailedLogger
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
from agent.meta_learner import MetaLearner
from agent.intent_discoverer import IntentDiscoverer
from agent.error_recovery import ErrorRecovery
from agent.workbench import Workbench
from agent.meta_selector import MetaCognitiveSelector
from agent.goal_generator import GoalGenerator
from agent.world_model_v4 import GrowingWorldModel
from agent.episodic_memory import EpisodicMemory
from agent.creative_writer import CreativeWriter
from agent.code_archive import CodeArchive
from agent.imagination import ImaginationEngine  # P19: 想象引擎
from benchmark.param_extractor import ParameterExtractor
from benchmark.template_engine import TemplateEngine, ExecResult
from collections import deque


# 意图列表
INTENTS = ["OBSERVE", "CREATE", "TRY"]
N_INTENTS = 3  # 3种执行模式, 具体做什么由 GoalGenerator + 命令推荐动态决定


class IntentClassifier:
    """MiniLM + MLP 意图分类器 (动态意图数)"""

    N_OUT = 17  # 默认, 会被 expand_intents 更新

    def __init__(self, checkpoint: str = "checkpoints/intent_classifier/best_head.pt"):
        import torch.nn as nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        self.encoder.to(self.device)
        self.encoder.eval()

        # P10: 使用动态意图数
        n_out = N_INTENTS
        IntentClassifier.N_OUT = n_out

        class MLPHead(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(384),
                    nn.Linear(384, 128),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, n_classes),
                )
            def forward(self, x):
                return self.net(x)

        self.head = MLPHead(n_out)
        try:
            sd = torch.load(checkpoint, map_location=self.device, weights_only=True)
            # P10: 自动适配维度 (扩展或收缩)
            last_key = 'net.7.weight'
            if last_key in sd:
                ckpt_n = sd[last_key].size(0)
                model_n = self.head.net[-1].weight.size(0)
                if ckpt_n != model_n:
                    old_w = sd.pop('net.7.weight')
                    old_b = sd.pop('net.7.bias')
                    self.head.load_state_dict(sd, strict=False)
                    copy_n = min(ckpt_n, model_n)
                    self.head.net[-1].weight.data[:copy_n] = old_w[:copy_n]
                    self.head.net[-1].bias.data[:copy_n] = old_b[:copy_n]
                    print(f"  \U0001f7e6 分类头: {ckpt_n}\u2192{model_n} (自动适配)")
                else:
                    self.head.load_state_dict(sd, strict=False)
            else:
                self.head.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"  ⚠️ 分类器checkpoint加载部分失败: {e}")
        self.head.to(self.device)
        self.head.eval()
        self.checkpoint_path = checkpoint

    def save(self, path: str = None):
        import torch
        torch.save(self.head.state_dict(), path or self.checkpoint_path)

    def expand_intents(self, new_n: int):
        """P6.4/P10: 扩展分类头输出层以容纳新意图"""
        import torch.nn as nn
        old_head = self.head.net[-1]
        old_n = old_head.out_features
        if new_n <= old_n:
            return

        old_weight = old_head.weight.data  # (old_n, 128)
        old_bias = old_head.bias.data      # (old_n,)
        new_head = nn.Linear(128, new_n).to(self.device)
        new_head.weight.data[:old_n] = old_weight
        new_head.bias.data[:old_n] = old_bias
        nn.init.normal_(new_head.weight.data[old_n:], std=0.01)
        nn.init.zeros_(new_head.bias.data[old_n:])
        self.head.net[-1] = new_head
        IntentClassifier.N_OUT = new_n
        print(f"  [Classifier] 分类头扩展: {old_n} → {new_n} 个输出")

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
        api_backend: str = "",
        api_key: str = "",
        model: str = "qwen3.5:0.8b",
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
        # P5.4: 元学习器
        self.meta = MetaLearner()
        meta_stats = self.meta.get_stats()
        if meta_stats["total_behaviors"] > 0:
            print(f"  \u2705 元学习器已恢复: {meta_stats['total_behaviors']} 个行为")
        # P4: 工作栏必须早于状态编码器
        self.workbench = Workbench(meta_learner=self.meta)
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
        # P8.0: 失败恢复模块
        self.error_recovery = ErrorRecovery(
            sandbox=self.sandbox,
            workbench=self.workbench,
        )

        # P10: 层级架构 — 元认知 + 目标生成 + 增长型 WM V4
        self.meta_selector = MetaCognitiveSelector()
        # P17: SelfModel (自我意识统计)
        from agent.self_model import SelfModel
        self.self_model = SelfModel()
        # 挂到 workbench 上供 CreativeWriter 自省 prompt 使用
        self.workbench.self_model = self.self_model

        self.creative_writer = None
        try:
            cw_backend = api_backend or os.environ.get("DEEPSEEK_BACKEND", "ollama")
            cw_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            self.creative_writer = CreativeWriter(
                model=model,
                timeout=120.0,
                max_tokens=2048,
                api_backend=cw_backend,
                api_key=cw_key,
            )
            if self.creative_writer and self.creative_writer.health_check():
                print(f"  ✅ CreativeWriter 就绪 ({self.creative_writer.model})")
            else:
                self.creative_writer = None
                print(f"  ⚠️ CreativeWriter 不可用 (Ollama?)")
        except Exception as e:
            print(f"  ⚠️ CreativeWriter 初始化失败: {e}")

        self.goal_generator = GoalGenerator(creative_writer=self.creative_writer)
        # P19: CodeArchive — 持久化代码, 追踪成长
        self.code_archive = CodeArchive()
        self.goal_generator.code_archive = self.code_archive
        if self.creative_writer:
            self.creative_writer.enable_async()
            self._last_llm_step = -100
            try:
                self.creative_writer.generate_async(self.workbench, "report")
            except Exception:
                pass
        else:
            self._last_llm_step = -100
        # V4 在所有意图上初始化叶
        self.world_model_v4 = GrowingWorldModel(embed_dim=384, thought_dim=16)
        for name in INTENTS:
            if name != "HELP":
                self.world_model_v4.add_intent(name)

        # P18: WorldModel V5.1 — RSSM Lite (随机隐变量 + KL 惊奇度)
        from agent.world_model_v5_1 import WorldModelV51
        self.world_model_v5 = WorldModelV51()
        self._wm5_buffer: list[dict] = []
        self._wm5_train_interval = 15
        self._wm5_batch_size = 32
        self._wm5_checkpoint = "data/persistent/world_model_v5.pt"
        if os.path.exists(self._wm5_checkpoint):
            try:
                self.world_model_v5.load(self._wm5_checkpoint)
                print(f"  ✅ WorldModel V5.1 已加载 ({self.world_model_v5._param_count:,} params)")
            except Exception as e:
                print(f"  ⚠️ WorldModel V5.1 加载失败, 从头开始: {e}")
        else:
            print(f"  ℹ️ WorldModel V5.1 新创建 ({self.world_model_v5._param_count:,} params)")
        self._prev_fact_categories = {}

        # P18: IntuitionBuffer — 余弦相似度直觉缓冲
        from agent.intuition_buffer import IntuitionBuffer
        self.intuition_buffer = IntuitionBuffer(capacity=1024)
        print(f"  ✅ IntuitionBuffer 就绪 (cap=1024, 0 参数)")

        # P19: 睡眠巩固 — 用经验回放微调 Conductor
        self._sleep_interval = 100
        self._sleep_batch_size = 32
        self._sleep_lr = 1e-5
        self._sleep_steps = 3
        self._sleep_optimizer = None
        # P19: 默认模式/无聊度 — 内生冲动驱动
        self._boredom = 0.0
        self._boredom_decay = 0.05
        self._boredom_threshold = 0.6
        self._spontaneous_source = False
        print(f"  ✅ 增长型WM V4: {len(self.world_model_v4.leaves)} 个意图叶")
        # P10: 情景记忆 (初始禁用, V4 ready 后启用)
        self.episodic_memory = EpisodicMemory()
        print("  \u2705 情景记忆就绪 (初始禁用)")

        # 尝试加载 V4 checkpoint
        wm_v4_ckpt = "checkpoints/world_model/v4_latest.pt"
        if os.path.exists(wm_v4_ckpt):
            try:
                self.world_model_v4.load(wm_v4_ckpt)
                print(f"  \u2705 WM V4 已加载: {wm_v4_ckpt}")
            except Exception as e:
                print(f"  \u26a0\ufe0f WM V4 加载失败: {e}")

        # 训练配置
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.mode = mode  # stable | creative | auto
        self.novelty_weight = novelty_weight
        self.explore_prob = explore_prob
        self.conductor_gate = conductor_gate  # A/B 切换阈值

        # 自适应奖励: 从经验中学习每个意图的价值, 不再手写
        # 初始化时用均匀保守值, 运行时逐步替换为经验均值
        self._reward_tracker: dict[str, list[float]] = {}
        for name in INTENTS:
            if name != "HELP":
                self._reward_tracker[name] = []
        self._adapted_rewards: dict[str, float] = {}  # intent → running mean reward
        self.device = self.classifier.device
        self._tried_custom_cmds: set[str] = set()  # P5.6: 追踪已试过的 CUSTOM 命令
        # P7.2: 概率门控 — 让 A/B 决策拿回核心循环的主导权
        self.probe_rate = 0.5       # 探针占用概率 (原100%)
        self.imagination_rate = 0.8 # P9.7: 想象力占用概率 (原60%→80%)
        self._total_create_content = 0  # LLM 创作累计字节

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

        # P19: ImaginationEngine — 用 WM5.1 做想象回放
        # 必须在 Nanny 初始化后，因为需要 conductor
        self.imagination_engine = ImaginationEngine(
            world_model=self.world_model_v5,
            conductor=self.nanny.conductor if self.conductor_path_active else None,
            buffer=self.intuition_buffer,
            max_steps=5,
            device=self.device,
        )
        print(f"  ✅ ImaginationEngine 就绪 (max_steps=5)")

        # P19: 初始化 sleep optimizer（在 nanny 就绪后）
        if self.conductor_path_active:
            self._sleep_optimizer = torch.optim.AdamW(
                self.nanny.conductor.head.parameters(),
                lr=self._sleep_lr, weight_decay=1e-4,
            )
            print(f"  ✅ Sleep consolidation 就绪 (lr={self._sleep_lr})")
        else:
            self._sleep_optimizer = None

        # P11: PersistentStore
        from agent.persistent_store import PersistentStore
        self.pstore = PersistentStore()
        if self.pstore._read_version() is not None:
            self.pstore.load_all(self)
        else:
            print(f"  ℹ️ PersistentStore: 无历史数据, 从零开始")

        # P16 R1: Transition recording (因果推理数据收集)
        self._transitions: list[dict] = []
        self._transition_file = "data/persistent/transitions.jsonl"
        self._transition_flush_size = 50
        os.makedirs("data/persistent", exist_ok=True)
        # 尝试从文件恢复最近 transition
        self._load_recent_transitions()

        # P16 R2: 因果挖掘 + 假设生成
        from agent.transition_miner import TransitionMiner
        from agent.hypothesis_engine import HypothesisEngine
        self.transition_miner = TransitionMiner()
        self.hypothesis_engine = HypothesisEngine(top_k=5)
        # P16 R3: 实验规划 + 验证
        from agent.experiment_planner import ExperimentPlanner
        from agent.verdict import Verdict
        self.experiment_planner = ExperimentPlanner()
        self.verdict = Verdict(lr=0.3)
        self._last_experiment_step = -10
        self._latest_hypotheses: list[dict] = []

        self.ab_stats = {"conductor": 0, "classifier": 0, "conductor_success": 0, "classifier_success": 0, "goal_driven": 0, "goal_driven_success": 0, "imagined": 0, "imagined_success": 0}
        self.multi_cmds_count = 0  # P1: 多命令步数
        self._last_cond_logits = None  # P2: 最近一次 Conductor logits
        self.ab_window = deque(maxlen=100)  # P1.5: 滑动窗口, 记录 (used_conductor, success)

        # 优化器 (只训练分类头)
        self.optimizer = torch.optim.AdamW(
            self.classifier.head.parameters(), lr=lr, weight_decay=1e-4
        )
        # P8.3: LR 调度 — 损失停滞时降学习率
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
            threshold=0.01, min_lr=1e-6
        )
        self._train_losses: list[float] = []

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

        # P11: 持久化运行信息
        import time
        self._run_id = f"run_{int(time.time())}"
        self._started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # 统计
        self.step_count = 0
        self.total_reward = 0.0
        self.intent_history: list[str] = []
        self.success_count = 0
        self._last_action_source = "classifier"  # P5.4: 'probe'|'goal'|'imagination'|'conductor'|'classifier'
        self._last_was_imagined = False  # P5.2: 当前步是否来自想象力
        self._discovered_commands: set[str] = set()  # P8.5d: 从沙箱扫到的新命令
        self._discovered_cmd_last_scan: int = 0
        # P17: 自省方向 (LLM输出→实际行为)
        self._self_direction: str = ""      # 当前方向描述
        self._self_remaining: int = 0       # 剩余偏置步数
        self._self_plan_steps: int = 0      # 当前规划的第几步
        self.logger = DetailedLogger()  # P9: 超级日志

        # 失败抑制: 同一 intent+params 失败后 N 步内不再尝试
        self._failure_log: dict[str, int] = {}  # key→step_number_last_tried
        self._failure_cooldown = 50  # 步数冷卻

        # P12: KnowledgeMapper 知识拓展
        from agent.knowledge_mapper import KnowledgeMapper
        self.knowledge_mapper = KnowledgeMapper(sandbox=self.sandbox, workbench=self.workbench)
        self.workbench._knowledge_mapper = self.knowledge_mapper
        # 立即跑 Phase A (静态清单, 全快命令)
        n_a = self.knowledge_mapper.run_phase("A", 0)
        if n_a > 0:
            print(f"  ✅ 知识拓展 Phase A: {n_a} 个新事实")
        # 尽早扫描可用命令, 让 try_command 和 probe 有候选
        if hasattr(self.knowledge_mapper, 'scan_available_commands'):
            self.knowledge_mapper.scan_available_commands()
            n_avail = len(getattr(self.knowledge_mapper, '_all_available_commands', []))
            if n_avail > 0:
                print(f"  📋 扫描到 {n_avail} 个可用命令")

        # P13: ToolFactory + ToolRegistry
        from agent.tool_factory import ToolFactory
        from agent.tool_registry import ToolRegistry
        self.tool_factory = ToolFactory()
        self.tool_registry = ToolRegistry()
        # 从418命令池自造初始工具
        if hasattr(self, 'knowledge_mapper'):
            first_tools = self.tool_factory.discover_new_tools(self.knowledge_mapper, n=10)
            for fname in first_tools:
                self.tool_registry.register(fname, description=f"tool from 418 pool",
                                            tool_type="utility")
            if first_tools:
                print(f"  ✅ 工具工厂: {len(first_tools)} 个初始工具 (418池)")

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

    # ── P16 R1: Transition recording ──

    def _load_recent_transitions(self, max_entries: int = 500):
        """从 JSONL 加载最近的 transition 记录"""
        if not os.path.exists(self._transition_file):
            return
        try:
            with open(self._transition_file) as f:
                lines = f.readlines()
            # 只保留最近 max_entries 条
            recent = lines[-max_entries:]
            for line in recent:
                line = line.strip()
                if line:
                    try:
                        self._transitions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if self._transitions:
                print(f"  ✅ 已加载 {len(self._transitions)} 条 transition 记录")
        except (OSError, IOError):
            pass

    def _flush_transitions(self):
        """将 transition 缓冲区写入 JSONL 文件"""
        if not self._transitions:
            return
        try:
            with open(self._transition_file, "a") as f:
                for t in self._transitions[-self._transition_flush_size:]:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            # 清空已刷新的条目 (保留最近未写入的)
            self._transitions = []
            # 限制文件大小: 只保留最近 600 条 (500 + buffer)
            self._trim_transition_file()
        except (OSError, IOError) as e:
            if random.random() < 0.01:
                print(f"  [TRANSITION] 写入失败: {e}")

    def _trim_transition_file(self, keep: int = 600):
        """限制 transition 文件行数"""
        if not os.path.exists(self._transition_file):
            return
        try:
            with open(self._transition_file) as f:
                lines = f.readlines()
            if len(lines) > keep:
                with open(self._transition_file, "w") as f:
                    f.writelines(lines[-keep:])
        except (OSError, IOError):
            pass

    # P9.2: 创作意图基础奖励 (低于 INFO 但高于 LS_TMP, 确保被选中但不泛滥)
    INTENT_REWARD_CREATE = {
        "WRITE":   0.5,
        "APPEND":  0.4,
        "GENERATE": 0.6,
    }

    def _get_adaptive_base_reward(self, intent: str) -> float:
        """
        从经验中学习每个意图的基础奖励
        返回 running mean, 初始为 0.5, 随实际执行逐步调整
        """
        if intent == "HELP":
            return -0.8
        # 如果有经验数据, 用 running mean
        if hasattr(self, '_reward_tracker') and intent in self._reward_tracker:
            vals = self._reward_tracker[intent]
            if len(vals) >= 3:
                return sum(vals[-20:]) / min(len(vals), 20)
        # 有已学习的自适应值?
        if hasattr(self, '_adapted_rewards') and intent in self._adapted_rewards:
            return self._adapted_rewards[intent]
        # 默认 0.5
        return 0.5

    def _update_reward_knowledge(self, intent: str, actual_reward: float):
        """每步更新对某个意图的价值认知"""
        if not hasattr(self, '_reward_tracker'):
            return
        if intent not in self._reward_tracker:
            self._reward_tracker[intent] = []
        self._reward_tracker[intent].append(actual_reward)
        # 保留最近 50 步
        if len(self._reward_tracker[intent]) > 50:
            self._reward_tracker[intent] = self._reward_tracker[intent][-50:]
        # 更新自适应值
        vals = self._reward_tracker[intent]
        if len(vals) >= 3:
            self._adapted_rewards[intent] = sum(vals) / len(vals)

    def _compute_reward(self, result: ExecResult, intent: str, novelty: float,
                        intent_diversity: float = 0.0, chain_bonus: float = 0.0,
                        params: dict = None, facts_before: int = 0) -> float:
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

        # 自适应基础奖励 (从经验学习, 不再手写)
        base = self._get_adaptive_base_reward(intent)

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

        # P9.2: 创作奖励 — 区分模板/LLM/代码
        if intent == "CREATE" and success:
            written_bytes = self._get_written_file_size(result, params)
            source = params.get("source", "") if isinstance(params, dict) else ""
            # 代码执行奖励 (通过 sync LLM 路径, 已在 _llm_code 中单独处理)
            # 这里只处理 GoalGenerator 来的 CREATE
            if source in ("auto_report", "auto_analysis", "auto_experiment"):
                # 模板内容: 基础奖励
                reward += 0.2
                reward += min(0.3, written_bytes / 200.0 * 0.1)
            elif source == "llm_create":
                # LLM 生成内容: 中等奖励
                reward += 0.5
                reward += min(0.6, written_bytes / 100.0 * 0.1)
            elif source == "auto_script":
                # 脚本模板: 尝试执行
                script_path = params.get("path", "")
                if script_path:
                    r = self.sandbox.execute(f"bash {script_path} 2>&1 | head -5", timeout=10)
                    executed_ok = r and r.exit_code == 0
                    if executed_ok:
                        reward += 1.5
                        reward += min(0.5, len((r.stdout or "")) / 100.0 * 0.1)
                        print(f"  [SCRIPT] {script_path} 执行成功 +{reward:.1f}")
                    else:
                        reward += 0.3  # 尝试就有奖励
            else:
                reward += 0.3
                reward += min(0.5, written_bytes / 100.0 * 0.1)
            # 新路径惩罚
            write_key = f"write:{params.get('path','')}"
            if getattr(self, "_recent_writes", {}).get(write_key, 0) > self.step_count - 30:
                reward *= 0.3
            if not hasattr(self, "_recent_writes"):
                self._recent_writes = {}
            self._recent_writes[write_key] = self.step_count
            # 创作来源多样性奖励
            if not hasattr(self, "_last_create_source"):
                self._last_create_source = ""
            if source and source != self._last_create_source:
                reward += 0.3  # 换类型有额外奖励
            self._last_create_source = source
        # 创意模式: 强多样性奖励
        diversity_weight = 0.5 if use_creative else 0.3
        reward += intent_diversity * diversity_weight

        # P9.4: 新事实发现奖励 — 激励探索未知文件/命令
        if hasattr(self, "workbench") and self.workbench:
            facts_now = len(self.workbench.facts)
            if facts_now > facts_before:
                reward += min(0.5, (facts_now - facts_before) * 0.15)
                if intent == "CUSTOM":
                    reward += 0.3  # CUSTOM 新事实额外奖励

        # P5.6: 命令级新颖性奖励 — CUSTOM 命令第一次被尝试
        if intent == "CUSTOM" and success and hasattr(self, "_tried_custom_cmds") and params:
            cmd_str = str(params.get("custom_args", ""))[:40]
            if cmd_str not in self._tried_custom_cmds:
                reward += 1.0
                self._tried_custom_cmds.add(cmd_str)

        # P5.5: 强重复惩罚: 同一个意图连续 >2 次就递减
        recent = self.intent_history[-5:] if len(self.intent_history) >= 5 else []
        if recent:
            seq_count = 0
            for h in reversed(recent):
                if h == intent:
                    seq_count += 1
                else:
                    break
            if seq_count >= 3:
                reward *= 0.2  # 连续3次以上, 几乎归零
            elif seq_count == 2:
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
        if executed_intent == "HELP":
            return  # HELP 不适合当训练信号
        # P7.0: CUSTOM 现在是正常意图, 允许收集一致性样本

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
                # P7.0: CUSTOM 是正常意图, 保留
                sample = [s for s in sample if s[1] in INTENTS[:IntentClassifier.N_OUT]]
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

    def _adaptive_should(self, action: str, base_rate: float = 0.5) -> bool:
        """根据系统状态自适应决定是否执行某动作 — 替代硬编码 step%N
        
        Args:
            action: 动作名 ("make_tool", "make_script", "run_llm", etc.)
            base_rate: 基础概率 (0~1)
            
        Returns:
            True/False
        """
        import random
        
        # 获取系统状态
        n_facts = len(self.workbench.graph.nodes) if hasattr(self.workbench, 'graph') else 100
        km = getattr(self, 'knowledge_mapper', None)
        n_commands = len(getattr(km, '_all_available_commands', [])) if km else 0
        n_tools = len(getattr(self.tool_registry, '_index', {})) if hasattr(self, 'tool_registry') else 10
        n_infs = sum(1 for n in self.workbench.graph.nodes.values() 
                     if n.category == 'inference') if hasattr(self.workbench, 'graph') else 0
        
        # 饱和度: 0=饱和, 1=有很多空间
        saturation = max(0.0, 1.0 - n_facts / 400.0)
        cmd_left = max(0.0, 1.0 - n_tools / max(n_commands, 1))
        
        probs = {
            "make_tool":    0.3 * saturation + 0.3 * cmd_left,
            "make_script":  0.2 * saturation + 0.2 * (1.0 - n_infs / 20.0),
            "run_llm":      0.15 * (1.0 - n_infs / 15.0) + 0.05 * (1.0 - saturation),
            "run_tool":     0.2 * (1.0 - n_tools / max(n_commands, 1)),
            "self_review":  0.1 * (1.0 - saturation),
            "infer":        0.05 * (1.0 - n_infs / 20.0),
            "discover_cmd": 0.1 * cmd_left,
            "mode_select":  0.15,
            "self_goal":    0.1 * (1.0 - n_infs / 15.0) + 0.05 * (1.0 - saturation),
        }
        
        prob = probs.get(action, base_rate)
        return random.random() < prob

    def _create_and_run_script(self):
        """
        P5.3c: 基于工作栏事实生成 shell 脚本并执行
        脚本存于 /workspace/scripts/ (P1: 可写空间)
        """
        if not self.sandbox:
            return
        script, combo_key = self.workbench.generate_script() or (None, "")
        if not script:
            return
        try:
            # 确保目录存在
            self.sandbox.execute("mkdir -p /workspace/scripts", timeout=5)
            script_name = f"discover_{self.step_count}.sh"
            # 写脚本 (用 base64 编码避免 shell 特殊字符)
            import base64 as _b64
            encoded = _b64.b64encode(script.encode()).decode()
            self.sandbox.execute(
                f"echo '{encoded}' | base64 -d > /workspace/scripts/{script_name}", timeout=10
            )
            self.sandbox.execute(f"chmod +x /workspace/scripts/{script_name}", timeout=5)
            # 执行脚本
            r = self.sandbox.execute(f"/workspace/scripts/{script_name} 2>&1", timeout=15)
            if r.exit_code == 0 and r.stdout and len(r.stdout) > 20:
                out = r.stdout[:500]
                self.workbench.extract_facts("SCRIPT", script_name, out, {}, self.step_count)
                # P5.4: 记录脚本效用
                script_facts_before = len(self.workbench.facts)
                combo = self.workbench.get_last_script_combo()
                self.meta.register(f"script_{self.step_count}", "script_output",
                                  {"script": script_name, "size": len(r.stdout),
                                   "combo": combo}, self.step_count)
                self.meta.record(f"script_{self.step_count}",
                                min(1.0, len(r.stdout) / 500), self.step_count)
                print(f"  [SCRIPT] {script_name} -> {len(r.stdout)} bytes (combo: {combo})")
                # 记到发现日志
                try:
                    safe = r.stdout[:80].replace("'", "").replace("\n", " | ")
                    self.sandbox.execute(
                        f"printf '### 脚本{self.step_count} {script_name}\\n{safe}\\n\\n' >> /tmp/discoveries.md"
                    )
                except Exception:
                    pass
                # 自改进: 为新输出模式添加规则
                self._maybe_add_discovery_rule(r.stdout, script_name)
            elif r.exit_code != 0:
                print(f"  [SCRIPT] {script_name} FAIL (exit={r.exit_code})")
        except Exception as e:
            pass

    def _maybe_add_discovery_rule(self, output: str, script_name: str):
        """
        检查脚本输出中是否有可提取的新模式
        如果有, 追加到用户自定义规则
        """
        lines = output.splitlines()
        # 找 "key=value" 模式的输出
        import re
        for line in lines:
            line = line.strip()
            m = re.match(r'^\w+=[\w\./\-:]+$', line)
            if m:
                key, val = line.split("=", 1)
                self.workbench.add_user_rule(
                    trigger_type="output_contains",
                    trigger_pattern=key,
                    key=key,
                    category="script",
                )
                print(f"  \U0001f9f0 新规则: {key}={val} (来自 {script_name})")

    def _scan_new_commands(self) -> list[str]:
        """
        P8.5d: 从沙箱 /usr/bin 发掘新命令
        每50步扫描一次, 扩充 CUSTOM 探索空间
        """
        if not self.sandbox:
            return []
        if self.step_count - self._discovered_cmd_last_scan < 50:
            return []
        self._discovered_cmd_last_scan = self.step_count
        try:
            # 用 compgen -c (bash 内置) 列出所有可用命令, 随机取 10 个
            r = self.sandbox.execute(
                "compgen -c 2>/dev/null | shuf -n 10 || "
                "ls /usr/bin /bin /usr/local/bin 2>/dev/null | sort -u | shuf -n 10",
                timeout=5
            )
            new_cmds = []
            if r and r.stdout:
                for line in r.stdout.strip().split("\n"):
                    cmd = line.strip()
                    if (cmd and len(cmd) > 1 and len(cmd) < 30
                        and cmd not in self._tried_custom_cmds
                        and cmd not in self._discovered_commands):
                        self._discovered_commands.add(cmd)
                        new_cmds.append(cmd)
            if new_cmds and random.random() < 0.01:
                print(f"  [SCAN] 发现 {len(new_cmds)} 个新命令: {', '.join(new_cmds[:5])}...")
            return new_cmds
        except Exception:
            return []

    def _get_untried_custom_cmds(self) -> list[str]:
        """P8.5d: 获取未尝试过的自定义命令 (含动态发掘)"""
        untried = []
        # 1. 从 selector 历史中找未试过的
        if hasattr(self.cmd_selector, "history"):
            for cmd, meta in self.cmd_selector.history.items():
                if meta.get("tries", 0) == 0 and cmd not in self._tried_custom_cmds:
                    untried.append(cmd)
        # 2. 从 clusterer 的 mined 命令中找
        if hasattr(self.clusterer, "clusters"):
            for cluster_name, cmds in self.clusterer.clusters.items():
                for cmd in cmds:
                    if cmd not in self._tried_custom_cmds and cmd not in untried:
                        untried.append(cmd)
        # 3. P8.5d: 从动态发现的命令中找
        new_from_scan = [c for c in self._discovered_commands
                         if c not in self._tried_custom_cmds and c not in untried]
        untried.extend(new_from_scan)
        # P9.4: 过滤带空格的命令名 (不是合法单命令)
        untried = [c for c in untried if ' ' not in c]
        return untried[:30]  # 限制30个 (原20)

    def _create_script_from_commands(self):
        """从418命令池组合随机命令生成shell脚本并执行"""
        import random, base64
        all_cmds = getattr(self.knowledge_mapper, '_all_available_commands', [])
        if len(all_cmds) < 3:
            return
        chosen = random.sample(all_cmds, min(random.randint(3, 5), len(all_cmds)))
        lines = ["#!/bin/bash", "# Auto-generated from 418 command pool", "set -e", ""]
        for cmd in chosen:
            lines.append(f"echo '=== {cmd} ==='")
            lines.append(f"{cmd} 2>&1 || echo '(exit: $?)'")
            lines.append("")
        script = "\n".join(lines)
        name = f"cmd_{self.step_count}.sh"
        encoded = base64.b64encode(script.encode()).decode()
        self.sandbox.execute("mkdir -p /workspace/scripts")
        self.sandbox.execute(f"echo '{encoded}' | base64 -d > /workspace/scripts/{name}")
        self.sandbox.execute(f"chmod +x /workspace/scripts/{name}")
        r = self.sandbox.execute(f"/workspace/scripts/{name} 2>&1", timeout=30)
        if r and r.stdout and len(r.stdout) > 20:
            self.workbench.extract_facts("CMDPOOL", name, r.stdout[:500], {}, self.step_count)
            print(f"  [CMDPOOL] {name} ({len(chosen)} cmds) -> {len(r.stdout)}B")
        elif r:
            print(f"  [CMDPOOL] {name} -> {len(r.stdout or '')}B (short)")

    def _validate_custom(self, args: list) -> list:
        """
        P9.4: 验证 CUSTOM 命令参数
        - 如果 args 是带空格的字符串, 切成正确列表
        - 检查首命令是否存在, 不存在则 fallback 到 echo
        - 确保命令有合理参数
        """
        if not args:
            return ["echo", "(empty_custom)"]

        # 标准化为列表
        if isinstance(args, str):
            import ast
            try:
                # 尝试解析 Python repr 格式: "['sh', '-c', 'cmd']"
                parsed = ast.literal_eval(args)
                if isinstance(parsed, list):
                    args = parsed
                else:
                    import shlex
                    args = shlex.split(args)
            except (ValueError, SyntaxError):
                import shlex
                args = shlex.split(args)

        # P9.4: 展开带空格的参数
        # sh -c 脚本必须保留为整体, 不展开
        if len(args) >= 2 and args[0] == "sh" and args[1] == "-c":
            pass  # 保持 sh -c 脚本参数完整
        else:
            expanded = []
            for a in args:
                if isinstance(a, str) and ' ' in a:
                    import shlex
                    expanded.extend(shlex.split(a))
                else:
                    expanded.append(a)
            args = expanded

        cmd = args[0]
        if len(args) == 1:
            # 裸命令: 检查是否需要参数
            _bare_ok = {"hostname", "uname", "id", "pwd", "arch",
                        "env", "whoami", "groups", "lsmod", "mount",
                        "locale", "timedatectl", "pstree", "lspci"}
            if cmd in _bare_ok:
                return args
            # 需要至少 --help 参数
            args.append("--help")

        # 检查命令是否在沙箱中存在 (只查裸名, 不改 cmd_selector)
        if self.sandbox and ' ' not in cmd:
            try:
                check = self.sandbox.execute(f"command -v {cmd} >/dev/null 2>&1 && echo yes || echo no")
                if check.stdout and 'no' in check.stdout.strip():
                    # 不存在: fallback 到 echo
                    return ["echo", f"(command '{cmd}' not available in sandbox)"]
            except Exception:
                pass

        return args

    def _get_written_file_size(self, result, params: dict) -> int:
        """P9.2: 获取刚写入文件的字节数"""
        # 从 stdout 获取 (write 模板输出字节数)
        if hasattr(result, 'stdout') and result.stdout and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
        # 从沙箱 stat
        if self.sandbox and params:
            path = params.get("path", "")
            if path:
                try:
                    r = self.sandbox.execute(f"stat -c %s {path} 2>/dev/null || echo 0", timeout=5)
                    if r.exit_code == 0 and r.stdout and r.stdout.strip().isdigit():
                        return int(r.stdout.strip())
                except Exception:
                    pass
        return 0

    def _rescue_params(self, intent: str, params: dict) -> dict:
        """
        P8.5b: 参数预校验 — 执行前替换无效参数

        模式化的参数错误:
          - path="" (空字符串) → 默认路径
          - path 是意图名而不是文件路径 → 默认路径
          - pattern="" → 默认 "root"
          - cmd="" → 默认 "ls"
        """
        import re
        fixed = dict(params)

        # 已知的路径类意图
        path_intents = {"READ", "LIST", "COUNT", "SEARCH", "READ_ETC",
                       "DISK_USAGE", "EXPLORE"}
        if intent in path_intents:
            p = fixed.get("path", "")
            if not p or not isinstance(p, str) or len(p.strip()) < 2:
                fixed["path"] = "/etc/hostname"
            elif p.upper() in {"READ", "LIST", "SEARCH", "COUNT", "INFO",
                               "CUSTOM", "HELP", "EXPLORE", "INSPECT",
                               "READ_ETC", "LS_TMP", "ARCH_INFO",
                               "USB_DEVICES", "DISK_USAGE"}:
                # path 值恰好是意图名 → 显然是参数错误
                fixed["path"] = "/etc/hostname"
            elif not p.startswith("/"):
                fixed["path"] = "/etc/hostname"
            # P9.1: 路径必须是已知合法前缀, 否则是事实值被误用
            _valid_path_prefixes = ("/proc/", "/etc/", "/tmp/", "/sys/",
                                    "/var/", "/dev/", "/home/", "/usr/",
                                    "/workspace/", "/persistent/",
                                    "/bin/", "/sbin/", "/lib/", "/opt/")
            if not any(p.startswith(prefix) for prefix in _valid_path_prefixes):
                fixed["path"] = "/etc/hostname"

        # 搜索类意图: 固定用可靠 pattern+path
        if intent == "SEARCH":
            # 始终用 root (从事实推断的模式可能过期)
            fixed["pattern"] = "root"
            if not fixed.get("path", "") or fixed.get("path") in ("/etc/hostname", "/22Gi"):
                fixed["path"] = "/etc/passwd"

        # INSPECT: cmd 不能空
        if intent == "INSPECT":
            if not fixed.get("cmd", ""):
                fixed["cmd"] = "ls"

        # P0: 写操作路径白名单 + P3: 内容生成器
        if intent in ("WRITE", "APPEND", "GENERATE"):
            p = fixed.get("path", "")
            _safe_write_prefixes = ("/tmp/", "/workspace/", "/persistent/scripts/")
            if not any(p.startswith(prefix) for prefix in _safe_write_prefixes):
                fixed["path"] = "/tmp/folunar_output.txt"
            # P3/P9.6: 从工作栏事实生成有价值的内容
            if hasattr(self, "workbench") and self.workbench:
                if intent == "GENERATE":
                    content_info = self.workbench.build_generate_content()
                else:
                    content_info = self.workbench.build_write_content()
                fixed["path"] = content_info["path"]
                fixed["content"] = content_info["content"]
            else:
                if not fixed.get("content", ""):
                    fixed["content"] = "# generated by Folunar\n"
            # P9.4: 确保目录存在 (写入前创建)
            parent = os.path.dirname(fixed.get("path", "/tmp/folunar_output.txt"))
            if parent and self.sandbox:
                self.sandbox.execute(f"mkdir -p {parent}", timeout=3)

        return fixed

    def _imagine_intent(self, state_text: str, temperature: float = 1.5) -> str | None:
        """
        P8.4d: 基于世界模型的意图想象 — 替代 P5.2 的伪逆解码

        核心思想:
          世界模型已经能预测每个意图的 next_thought。
          与其通过伪逆转码恢复意图 logits (数学幻象),
          不如直接用世界模型的预测信号来识别最有前景的意图。

        评分公式:
          score[i] = -
            P(agreement)[i] * 0.3      # 低 agreement = 反直觉组合 → 探索奖励
            + value[i] * 0.4            # 高价值 = 有回报的路径
            + thought_dist[i] * 0.3     # 思考距离 = 新颖的认知状态
        """
        if not self.conductor_path_active:
            return None
        try:
            # 1. 编码当前状态
            emb = self.classifier.get_embedding(state_text).to(self.world_model.device)
            cond_emb = self.nanny.conductor.encoder.encode(
                state_text, convert_to_tensor=True
            ).clone().to(self.world_model.device)
            thought, _ = self.nanny.conductor.head(cond_emb.unsqueeze(0))
            thought = thought.squeeze(0)

            # 2. 批量模拟所有意图, 收集世界模型预测
            # P10: V4 ready 后用 V4
            if self._v4_ready():
                candidates_v4 = [n for n in INTENTS if n != "HELP"]
                best_name, best_val = self.world_model_v4.get_best_intent(
                    emb, cond_emb.squeeze(0), candidates_v4
                )
                intent_name = best_name
                if intent_name and random.random() < 0.1:
                    print(f"  [IMAGINE-V4] best={intent_name} value={best_val:.3f}")
                return intent_name

            n = self.world_model.n_intents
            oh = torch.zeros(n, n, device=self.world_model.device)
            oh[torch.arange(n), torch.arange(n)] = 1.0
            s_batch = emb.unsqueeze(0).repeat(n, 1)
            t_batch = thought.unsqueeze(0).repeat(n, 1)

            self.world_model.predictor.eval()
            with torch.no_grad():
                pred = self.world_model.predictor(s_batch, t_batch, oh)

            # 3. 计算每个意图的评分
            # agreement_prob: softmax 取 prob(class=1) = 合理
            agreement_probs = torch.softmax(pred["agreement"], dim=-1)[:, 1]  # (n,)
            # value: 预期奖励
            values = pred["value"].squeeze()  # (n,)
            # next_thought 与当前 thought 的距离 (余弦距离)
            nt = pred["next_thought"]  # (n, thought_dim)
            thought_dist = 1.0 - torch.cosine_similarity(nt, thought.unsqueeze(0), dim=-1)  # (n,)

            # 归一化到 0~1
            agreement_norm = agreement_probs.squeeze()
            value_norm = torch.sigmoid(values)
            dist_norm = torch.clamp(thought_dist / max(thought_dist.max().item(), 0.01), 0, 1)

            # 评分: 低agreement(惊奇) + 高价值(回报) + 大距离(多样性)
            scores = (1.0 - agreement_norm) * 0.3 + value_norm * 0.4 + dist_norm * 0.3

            # P9.7: 新颖度奖励 — 给可能发现新事实的意图加分
            if hasattr(self, "workbench") and self.workbench:
                fact_count = len(self.workbench.facts)
                # 创作类意图 (WRITE/APPEND/GENERATE) 在事实多时更有价值
                for idx, name in enumerate(INTENTS):
                    if name in ("WRITE", "APPEND", "GENERATE") and fact_count >= 5:
                        scores[idx] += 0.05 * min(1.0, fact_count / 20.0)
                    # 探索类意图从边界探索获益
                    if name in ("EXPLORE", "CUSTOM", "READ"):
                        scores[idx] += 0.03  # 探索奖励

            # 4. 温度噪声: 鼓励多样性
            noise = torch.randn(n, device=self.world_model.device) * temperature * 0.15
            scores = scores + noise

            # 5. 选取最高分意图
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()

            # 如果所有意图得分都太低, 放弃想象
            if best_score < 0.15:
                return None

            intent_name = INTENTS[best_idx] if best_idx < len(INTENTS) else None
            if intent_name and random.random() < 0.1:
                top5 = scores.topk(min(5, n)).indices.tolist()
                top5_str = ", ".join(f"{INTENTS[i]}={scores[i]:.2f}" for i in top5)
                print(f"  [IMAGINE] scores: {top5_str}")

            return intent_name
        except Exception as e:
            return None

    def _select_intent(self, state_text: str, mode: str | None = None) -> str:
        """选择意图: MODE偏置 + 多样性 + 分类器 (P10)"""
        recent = self.intent_history[-20:] if len(self.intent_history) >= 20 else None
        if recent:
            usage = {i: recent.count(i) for i in set(recent)}
            most_used = max(usage, key=usage.get)
            most_used_pct = usage[most_used] / len(recent)
            if most_used_pct > 0.35:
                alternatives = [i for i in INTENTS if i != most_used and i != "HELP"]
                return random.choice(alternatives)

        if random.random() < self.explore_prob:
            return random.choice([i for i in INTENTS if i != "HELP"])

        emb = self.classifier.get_embedding(state_text)
        with torch.no_grad():
            logits = self.classifier.head(emb)

        # 元学习效用偏置
        utility_bias = torch.zeros(IntentClassifier.N_OUT, device=logits.device)
        if self.meta and len(self.meta.data) > 0:
            for bid, b in self.meta.data.items():
                if b.get("type") == "intent_choice":
                    iname = b.get("params", {}).get("intent", "")
                    if not iname or iname not in INTENTS or iname == "HELP":
                        continue
                    u = b.get("utility", 0.0)
                    n = b.get("n", 0)
                    if n < 3:
                        u *= n / 3.0
                    idx = INTENTS.index(iname)
                    if idx < IntentClassifier.N_OUT:
                        utility_bias[idx] += u * 0.3

        # P10: MODE 偏置
        mode_bias = torch.zeros(IntentClassifier.N_OUT, device=logits.device)
        if mode:
            bias_dict = self.meta_selector.get_intent_bias(mode)
            for iname, w in bias_dict.items():
                if iname in INTENTS:
                    mode_bias[INTENTS.index(iname)] = w

        biased_logits = logits + utility_bias + mode_bias * 0.5  # MODE偏置缩放到不影响分类器
        intent = INTENTS[biased_logits.argmax().item()]

        if intent == "HELP":
            alternatives = [i for i in INTENTS if i != "HELP"]
            sorted_idx = biased_logits.argsort(descending=True).flatten()
            for idx in sorted_idx.tolist():
                alt = INTENTS[idx]
                if alt in alternatives:
                    return alt
            return "INFO"

        return intent

    # ── P10: MODE 统计 ──

    def _v4_ready(self, intent_name: str | None = None) -> bool:
        """V4 readiness gate: per-leaf 或全局"""
        leaf_stats = self.world_model_v4.get_leaf_stats()
        if intent_name:
            st = leaf_stats.get(intent_name, {})
            return st.get("n_samples", 0) >= 20
        ready_count = sum(1 for st in leaf_stats.values() if st.get("n_samples", 0) >= 20)
        return ready_count >= max(1, len(self.world_model_v4.leaves) // 3)

    def _compute_belief_confidence(self) -> float:
        """计算信念置信度: miner 写入的边的平均 weight"""
        if not hasattr(self, 'workbench') or not hasattr(self.workbench, 'graph'):
            return 0.5
        g = self.workbench.graph
        if not g or not g.edges:
            return 0.5
        miner_weights = []
        for edges in g.edges.values():
            for e in edges:
                if e.get("hypothesis_key") == "transition_miner":
                    miner_weights.append(e.get("weight", 0))
        if not miner_weights:
            return 0.5
        return round(sum(miner_weights) / len(miner_weights), 3)

    def _compute_mode_stats(self) -> dict:
        """计算 MODE 选择器需要的统计量"""
        graph_st = self.workbench.graph.stats() if hasattr(self.workbench, "graph") else {}
        rnd_st = self.rnd.get_novelty_stats()
        if hasattr(self, "_facts_history") and len(self._facts_history) >= 2:
            growth = (self._facts_history[-1] - self._facts_history[0]) / max(len(self._facts_history), 1)
        else:
            growth = 0.0
        wm_loss = (sum(self._wm_loss_history[-5:]) / max(len(self._wm_loss_history), 1)
                   if hasattr(self, "_wm_loss_history") and self._wm_loss_history else 0.0)
        return {
            "step": self.step_count,
            "n_facts": graph_st.get("n_nodes", len(self.workbench.facts)),
            "n_gaps": graph_st.get("n_gaps", 0),
            "schema_coverage": graph_st.get("schema_coverage", 0.0),
            "rnd_avg": rnd_st.get("running_errors_avg", 0.0),
            "wm_loss": wm_loss,
            # P15: 模型置信度
            "wm_confidence": self.world_model_v4.get_confidence(),
            # P16 R4: 推理信念
            "belief_confidence": self._compute_belief_confidence(),
            "hypothesis_count": len(getattr(self, '_latest_hypotheses', [])),
            "wm_error": wm_loss,
        }

    def _compute_fact_diff(self) -> torch.Tensor:
        """Compute binary vector of fact categories that gained new nodes this step"""
        n_cats = 20
        result = torch.zeros(n_cats)
        try:
            wb = self.workbench
            if not hasattr(wb, 'graph') or not wb.graph:
                return result
            # Get current category counts
            cur = {}
            for n in wb.graph.nodes.values():
                cur[n.category] = cur.get(n.category, 0) + 1
            prev = getattr(self, '_prev_fact_categories', {})
            # Which categories increased?
            for cat, count in cur.items():
                old = prev.get(cat, 0)
                if count > old:
                    idx = hash(cat) % n_cats
                    result[idx] = 1.0
            # Store current as prev for next step
            self._prev_fact_categories = cur
        except Exception:
            pass
        return result


    # ── P17: 自省方向解析与应用 ──

    def _parse_and_apply_direction(self, intention: str):
        """解析 LLM 自省意图, 设置方向偏置"""
        text = intention.lower()
        direction = ""
        cluster_bias = {}

        # 关键词 → 方向 + cluster 偏置
        kw_dirs = {
            "filesystem": ("filesystem", {"fs": 2.0, "system": 0.5}),
            "file": ("filesystem", {"fs": 2.0}),
            "directory": ("filesystem", {"fs": 2.0}),
            "ls": ("filesystem", {"fs": 2.0}),
            "disk": ("filesystem", {"disk": 1.5}),
            "storage": ("storage", {"disk": 2.0}),
            "network": ("network", {"network": 2.0}),
            "connection": ("network", {"network": 2.0}),
            "socket": ("network", {"network": 2.0}),
            "ip": ("network", {"network": 2.0}),
            "script": ("scripting", {"script": 2.0}),
            "python": ("scripting", {"script": 2.0}),
            "code": ("scripting", {"script": 2.0}),
            "write": ("create", {"write": 1.5}),
            "report": ("create", {"content": 1.5}),
            "command": ("explore", {"system": 1.0}),
            "explore": ("explore", {"system": 1.5}),
            "process": ("process", {"process": 2.0}),
            "memory": ("memory", {"memory": 2.0, "system": 0.5}),
            "cpu": ("cpu", {"cpu": 2.0, "system": 0.5}),
            "hardware": ("hardware", {"system": 1.0, "cpu": 1.0}),
        }

        import re
        for kw, (dir_name, c_bias) in kw_dirs.items():
            if re.search(r'\b' + kw + r'\b', text):
                direction = dir_name
                cluster_bias = c_bias
                break

        if not direction:
            # 默认: 探索
            direction = "explore"
            cluster_bias = {"system": 1.0}

        # 设置方向
        self._self_direction = direction
        self._self_remaining = 5  # 接下来 5 步按方向走
        self._self_plan_steps = 0

        # 应用 cluster 偏置
        if hasattr(self, 'cmd_selector'):
            self.cmd_selector.cluster_bias = cluster_bias

        if random.random() < 0.3:
            print(f"  [SELF-DIR] {direction}: {intention[:60]}...")


    def _stats_based_direction(self):
        """直觉方向: 从 IntuitionBuffer 经验中采样, 不再用 if/else"""
        buf = getattr(self, 'intuition_buffer', None)
        signals = {}

        # P19: 如果无聊度很高，触发内生冲动
        boredom = getattr(self, '_boredom', 0.0)
        if boredom > getattr(self, '_boredom_threshold', 0.6):
            self._spontaneous_impulse()
            return

        # 1. IntuitionBuffer 查询 — 从经验中学习方向
        if buf and buf.size >= 10:
            result = buf.query(self.persistent_thought)
            signals["familiarity"] = round(result["familiarity"], 4)
            signals["n_exp"] = result["n_entries"]
            probs = list(result["direction_probs"])  # [p_obs, p_create, p_try]
        else:
            probs = [0.4, 0.2, 0.4]
            signals["n_exp"] = buf.size if buf else 0

        # 2. 惊奇度 z-score 修正
        rs = getattr(self, '_recent_surprise', [])
        if len(rs) >= 10:
            avg_sur = sum(rs[-10:]) / 10
            all_avg = sum(rs) / len(rs)
            std = max((sum((s - all_avg)**2 for s in rs) / len(rs))**0.5, 1e-8)
            z = (avg_sur - all_avg) / std
            signals["z"] = round(z, 2)
            if z > 1.5:
                probs[0] *= 1.5   # 高惊奇 → 观察
                probs[2] *= 1.2   # 尝试验证
            elif z < -1.0:
                probs[1] *= 2.0   # 低惊奇 → 创作

        # 3. KL 不确定度修正
        kl = getattr(self, '_last_kl', 0.0)
        signals["kl"] = round(kl, 3)
        if kl > 1.0:
            probs[0] *= 1.5
            probs[2] *= 1.2

        # 4. 归一化 → 采样方向
        p_t = torch.tensor(probs, dtype=torch.float)
        p_t = torch.softmax(p_t, dim=0)
        direction_map = {0: "explore", 1: "create", 2: "scripting"}
        chosen = int(p_t.argmax().item())
        direction = direction_map[chosen]

        self._self_direction = direction
        self._self_remaining = 5
        self._spontaneous_source = False
        if random.random() < 0.3:
            sig_parts = [f"{k}={v}" for k, v in signals.items()]
            sig_str = " ".join(sig_parts)
            print(f"  [DIR] {direction} ({sig_str}) "
                  f"pobs={p_t[0]:.2f} pcre={p_t[1]:.2f} ptry={p_t[2]:.2f}")

    def _spontaneous_impulse(self):
        """
        P19: 默认模式/内生冲动。
        当无聊度超过阈值时，从 IntuitionBuffer 中采样历史高光经验，
        复现其意图作为当前方向。
        """
        buf = getattr(self, 'intuition_buffer', None)
        if not buf or buf.size < 5:
            return

        # 采样高奖励成功经验
        batch = buf.sample_batch(batch_size=16, prefer_success=True)
        best = max(batch, key=lambda x: x["reward"] + (1.0 if x["success"] else 0.0))
        intent = best["intent"]
        direction_map = {0: "explore", 1: "create", 2: "scripting"}
        direction = direction_map.get(intent, "explore")

        self._self_direction = direction
        self._self_remaining = 5
        self._spontaneous_source = True
        self._boredom = 0.0
        print(f"  [SPONT] {direction} (from historical reward={best['reward']:.2f} "
              f"success={best['success']}) boredom reset")

    def _sleep_consolidation(self):
        """
        P19: 睡眠巩固 — 用 IntuitionBuffer 中的经验回放微调 Conductor。
        目标: Conductor(state_emb)  closer to stored thought。
        """
        if not self.conductor_path_active or not self._sleep_optimizer:
            return

        buf = getattr(self, 'intuition_buffer', None)
        if not buf or buf.size < 32:
            return

        batch = buf.sample_batch(batch_size=self._sleep_batch_size, prefer_success=True)
        # 只保留有 state_emb 的条目
        batch = [b for b in batch if b["state_emb"] is not None]
        if len(batch) < 8:
            return

        self.nanny.conductor.head.train()
        total_loss = 0.0
        n_steps = 0

        for _ in range(self._sleep_steps):
            random.shuffle(batch)
            states = torch.stack([b["state_emb"] for b in batch[:16]])
            targets = torch.stack([b["thought"] for b in batch[:16]])

            pred, _ = self.nanny.conductor.forward_emb(states)
            loss = F.mse_loss(pred, targets)

            self._sleep_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.nanny.conductor.head.parameters(), 1.0)
            self._sleep_optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        self.nanny.conductor.head.eval()
        avg_loss = total_loss / max(n_steps, 1)
        print(f"  [SLEEP] Conductor consolidation loss={avg_loss:.4f} "
              f"n={len(batch)}")

    def _update_boredom(self, reward: float, surprise: float):
        """P19: 更新无聊度。低奖励+低惊奇 -> 无聊度上升。"""
        interesting = max(0.0, reward) + surprise * 10.0
        if interesting < 0.05:
            self._boredom += 0.08
        else:
            self._boredom = max(0.0, self._boredom - self._boredom_decay)
        self._boredom = min(1.0, self._boredom)

    def _direction_step(self) -> tuple[str | None, dict | None]:
        """按当前方向执行一步"""
        self._self_remaining -= 1
        self._self_plan_steps += 1
        dir = self._self_direction
        step_n = self._self_plan_steps

        if dir == "filesystem":
            # 步1-2: OBSERVE 文件系统, 步3+: 总结
            if step_n <= 2:
                return "OBSERVE", {"path": "/"}
            else:
                return "CREATE", {"path": f"/tmp/fs_summary_{self.step_count}.md",
                                  "content": f"# Filesystem overview (step {self.step_count})\n"}
        elif dir == "network":
            return "TRY", {"custom_args": ["ip", "addr"]}
        elif dir == "scripting":
            if step_n <= 1:
                return "OBSERVE", {"path": "/workspace/scripts"}
            else:
                return "CREATE", {"path": f"/tmp/script_{self.step_count}.py",
                                  "content": "#!/usr/bin/env python3\nprint('script by agent')\n"}
        elif dir == "create":
            return "CREATE", {"path": f"/tmp/creation_{self.step_count}.md",
                              "content": f"# Creation step {self.step_count}\n"}
        elif dir == "explore":
            cluster, cmd_args = self.cmd_selector.select()
            return "TRY", {"custom_args": cmd_args, "cluster": cluster}
        elif dir == "process":
            return "TRY", {"custom_args": ["ps", "aux"]}
        elif dir == "memory":
            return "OBSERVE", {"path": "/proc/meminfo"}
        elif dir == "cpu":
            return "OBSERVE", {"path": "/proc/cpuinfo"}
        else:
            return "TRY", {"custom_args": ["uname", "-a"]}

    def step(self) -> tuple[bool, float]:
        """执行一步: P10 MODE/GOAL/ACTION路由"""
        self.step_count += 1
        self._last_was_imagined = False

        # 1. 更新状态编码器上下文 (动态 FactGraph 驱动)
        self.state_encoder.set_step(self.step_count)

        # 1. 编码当前状态
        state_text = self.state_encoder.get_state_text(thought_label="")
        state_emb = self.classifier.get_embedding(state_text).detach().clone()

        thought_vector = self.persistent_thought.clone()
        thought_label = ""
        if self.conductor_path_active:
            try:
                fresh_thought, cond_logits = self.nanny.think(state_text)
                thought_vector = 0.7 * thought_vector + 0.3 * fresh_thought
                cond_top = INTENTS[cond_logits.argmax().item()]
                discovery = self.workbench.get_current_discovery()
                if discovery:
                    val = self.workbench.get_fact(discovery) or ""
                    thought_label = f"{cond_top}@{discovery}={val[:20]}"
                else:
                    thought_label = cond_top
            except Exception:
                pass
        # MODE 选择 (在 state_text 里注入当前模式)
        if not hasattr(self, "_wm_loss_history"):
            self._wm_loss_history = []
        if not hasattr(self, "_facts_history"):
            self._facts_history = []
        mode_stats = self._compute_mode_stats()
        self.current_mode = self.meta_selector.select(mode_stats)
        self.state_encoder.set_mode(self.current_mode)
        thought_label = f"{self.current_mode}:{thought_label}" if thought_label else self.current_mode
        state_text = self.state_encoder.get_state_text(thought_label=thought_label)
        state_emb = self.classifier.get_embedding(state_text).detach().clone()

        # 2. 推理引擎: 自适应频率
        if self.step_count > 10 and self._adaptive_should("infer", 0.05):
            if hasattr(self, 'workbench') and hasattr(self.workbench, 'graph'):
                from agent.inference_engine import InferenceEngine
                ie = InferenceEngine(self.workbench.graph)
                n_inf = ie.infer_all(self.step_count)
                if n_inf > 0:
                    print(f"  [INFER] {n_inf} 个推断")

        # P16 R2: TransitionMiner 因果挖掘 (低频率)
        if (self.step_count > 50 and self._adaptive_should("mine_causal", 0.02)
                and hasattr(self, 'transition_miner')):
            try:
                candidates = self.transition_miner.mine(
                    path=self._transition_file,
                    graph=self.workbench.graph if hasattr(self.workbench, 'graph') else None,
                    window=500,
                )
                if candidates:
                    # 写入 FactGraph
                    self.transition_miner.apply_to_graph(
                        candidates, self.workbench.graph, step=self.step_count)
                    # 生成假设
                    hyps = self.hypothesis_engine.generate(
                        candidates, self.workbench.graph)
                    if hyps:
                        self._latest_hypotheses = hyps
                        print(f"  [HYPOTHESIS] {len(hyps)} 个假设 (来自 {len(candidates)} 候选边)")
                        for h in hyps[:3]:
                            print(f"    {h['if_node']} → {h['then_node']} "
                                  f"(priority={h['priority']:.3f})")
            except Exception as e:
                if random.random() < 0.05:
                    print(f"  [MINER-ERR] {e}")

        # P18: DoCalculusEngine — 干预评分 (补充 TransitionMiner 的统计关联)
        if (hasattr(self, 'workbench') and hasattr(self.workbench, 'graph')
                and hasattr(self.workbench.graph, 'edges')
                and self.step_count % 50 == 0):
            try:
                from agent.do_calculus import DoCalculusEngine
                dc = DoCalculusEngine()
                n_edges = dc.add_from_factgraph(self.workbench.graph)
                if n_edges > 0 and hasattr(self, '_transitions'):
                    # 对 FactGraph 中每个因果边计算干预评分
                    for src, edge_list in self.workbench.graph.edges.items():
                        for e in edge_list:
                            dst = e.get("to", "")
                            rel = e.get("rel", "")
                            if rel in ("causes", "predicts") and len(self._transitions) > 5:
                                cs = dc.causal_score(src, dst, self._transitions)
                                # 将 causal_score 存到边 metadata
                                e["causal_score"] = round(cs, 3)
            except Exception:
                pass

        if hasattr(self, 'knowledge_mapper') and self.step_count > 10:
            any_remaining = any(
                not self.knowledge_mapper.is_phase_done(p)
                for p in ["B", "C", "D", "E"]
            )
            if any_remaining:
                # 固定 phase 模式 (BFS探索)
                phase_intervals = {"B": 15, "C": 20, "D": 25, "E": 30}
                for phase in ["B", "C", "D", "E"]:
                    if not self.knowledge_mapper.is_phase_done(phase):
                        phase_prob = 1.0 / phase_intervals.get(phase, 15)  # 转换间隔为概率
                        if random.random() < phase_prob:
                            n_new = self.knowledge_mapper.run_phase(phase, self.step_count)
                            if n_new > 0:
                                print(f"  [KNOWLEDGE] Phase {phase}: {n_new} 个新事实")
                            break
            else:
                # 自发现模式: 用 RND 驱动
                # 确保命令列表已扫描
                if not self.knowledge_mapper._all_available_commands:
                    self.knowledge_mapper.scan_available_commands()
                    n_cmds = len(self.knowledge_mapper._all_available_commands)
                    if n_cmds > 0:
                        print(f"  [DISCOVER] 扫描到 {n_cmds} 个可用命令")

                # 自适应发现新命令
                if self.step_count > 30 and self._adaptive_should("discover_cmd", 0.1):
                    n_new = self.knowledge_mapper.discover_next(
                        self.step_count, rnd=self.rnd
                    )
                    if n_new > 0:
                        ks = self.knowledge_mapper.get_exploration_stats()
                        print(f"  [DISCOVER] 探索: {ks['explored']}/{ks['total_available']} 命令")

                # P13++: 从418命令池自造新工具 (自适应)
                if hasattr(self, 'tool_factory') and hasattr(self, 'knowledge_mapper') and self._adaptive_should("make_tool", 0.1):
                    new_tools = self.tool_factory.discover_new_tools(
                        self.knowledge_mapper, n=1)
                    if new_tools:
                        for fname in new_tools:
                            self.tool_registry.register(
                                fname, description=f"418-pool: {fname}",
                                tool_type="utility")
                        print(f"  [TOOL_AUTO] 新生成 {len(new_tools)} 个工具: {new_tools}")

        # P13: 自适应运行工具
        if hasattr(self, 'tool_registry') and self.step_count > 20 and self._adaptive_should("run_tool", 0.02):
            tool = self.tool_registry.get_best_tool()
            if tool:
                env = {
                    "workbench": self.workbench,
                    "sandbox": self.sandbox,
                    "state_text": state_text,
                }
                result = self.tool_registry.run_tool(tool["name"], env)
                if result.get("success"):
                    self.tool_registry.log_use(tool["name"], self.step_count, True,
                                                bytes_created=len(str(result.get("data", {}))))
                    # 如果工具有用数据, 注入 FactGraph
                    data = result.get("data", {}) or result.get("packages", []) or result.get("processes", []) or result.get("profile", "")
                    summary = result.get("summary", "")
                    if data or summary:
                        tool_key = f"tool_result_{tool['name']}"
                        tg_value = summary[:100]
                        wb = self.workbench
                        if hasattr(wb, 'graph'):
                            wb.graph.add_node(tool_key, tg_value, category="tool_result",
                                              confidence=0.7, step=self.step_count,
                                              source_cmd=f"tool:{tool['name']}")
                        elif hasattr(wb, 'facts'):
                            wb.facts[tool_key] = {
                                "value": tg_value, "category": "tool_result",
                                "confidence": 0.7, "step": self.step_count,
                                "source_cmd": f"tool:{tool['name']}",
                            }
                        print(f"  [TOOL] {tool['name']}: {summary}")
                else:
                    self.tool_registry.log_use(tool["name"], self.step_count, False)

        # 3. P10: GoalGenerator (MODE驱动目标)
        used_goal = False
        intent = None
        params = {}
        used_conductor = False

        # 跟踪事实数 (增长率用)
        n_facts_now = self.workbench.graph.node_count() if hasattr(self.workbench, "graph") else len(self.workbench.facts)
        self._facts_history.append(n_facts_now)

        # GoalGenerator 产生目标
        # P1: 每40步强制创作 (让 LLM CreativeWriter 有机会运行)
        # P2: 自适应 LLM 频率 (异步模式下每 20 步触发一次预生成)
        force_create = False
        if self.step_count > 30 and self.creative_writer:
            interval = 100 if self.creative_writer.is_thermal_ok() else 200
            if self.step_count - self._last_llm_step >= interval:
                force_create = True
                self._last_llm_step = self.step_count

        goal = self.goal_generator.generate(
            mode=self.current_mode,
            workbench=self.workbench,
            rnd_avg=mode_stats["rnd_avg"],
            step=self.step_count,
            recent_intents=list(self.intent_history[-20:]),
            force_create=force_create,
            knowledge_mapper=self.knowledge_mapper if hasattr(self, 'knowledge_mapper') else None,
            tool_registry=self.tool_registry if hasattr(self, 'tool_registry') else None,
            hypothesis_engine=self.hypothesis_engine if hasattr(self, 'hypothesis_engine') else None,
            fact_graph=self.workbench.graph if hasattr(self.workbench, 'graph') else None,
        )
        if goal:
            intent = goal.intent
            params = dict(goal.params)
            used_goal = True
            self._last_action_source = "goal_driven"
            self.ab_stats["goal_driven"] = self.ab_stats.get("goal_driven", 0) + 1
            if random.random() < 0.15:
                print(f"  [GOAL] ({goal.source}) {intent}")
            # P16 R3: 假设验证实验
            if hasattr(goal, 'type') and goal.type == "hypothesis_test":
                h_key = params.get("hypothesis_key", "")
                if (h_key and hasattr(self, 'experiment_planner')
                        and self.sandbox
                        and self.step_count - self._last_experiment_step >= 10):
                    try:
                        hypothesis = params.get("hypothesis", {})
                        plan = self.experiment_planner.plan(hypothesis)
                        if plan:
                            plan["_step"] = self.step_count
                            result = self.experiment_planner.execute_plan(plan, self.sandbox)
                            verdict = self.verdict.evaluate(
                                plan, result, self.workbench.graph)
                            self._last_experiment_step = self.step_count
                            print(f"  [EXPERIMENT] {h_key}: {verdict['verdict']} "
                                  f"(score={verdict['score']:.2f})")
                            if verdict["edge_removed"]:
                                print(f"  [EXPERIMENT] Edge removed: "
                                      f"weight={verdict['old_weight']} < -0.5")
                            # P16 R4: Feed verdict to WorldModel
                            if hasattr(self, 'world_model_v4') and plan.get("cmd"):
                                try:
                                    wm_input = f"{h_key} {verdict['verdict']} score={verdict['score']}"
                                    wm_emb = self.classifier.get_embedding(wm_input).detach().clone()
                                    self.world_model_v4.update(
                                        wm_emb, thought_vector.clone(),
                                        INTENTS.index("TRY"),
                                        f"experiment: {verdict['verdict']}",
                                        verdict["n_support"] if verdict["verdict"] == "support" else 1,
                                        verdict["score"],
                                    )
                                except Exception:
                                    pass
                            # P16 R4: Auto schema — verified edge → schema
                            if (verdict["verdict"] == "support"
                                    and verdict["score"] > 0.5
                                    and hasattr(self.workbench, 'graph')):
                                try:
                                    g = self.workbench.graph
                                    parts = h_key.split(":", 2)
                                    if len(parts) == 3:
                                        sk, _, dk = parts
                                        if sk in g.nodes and dk in g.nodes:
                                            cat_src = g.nodes[sk].category
                                            cat_dst = g.nodes[dk].category
                                            if cat_src == cat_dst:
                                                schema = g.schemas.setdefault(cat_src, [])
                                                if dk not in schema:
                                                    schema.append(dk)
                                                    print(f"  [SCHEMA] Added {dk} to {cat_src} schema")
                                except Exception:
                                    pass
                            intent = "TRY"
                            params = {"custom_args": plan["cmd"].split()}
                    except Exception as e:
                        if random.random() < 0.1:
                            print(f"  [EXPERIMENT-ERR] {e}")
            # P10: 即使有目标, 也检查多样性 (防止 GoalGenerator 返回同类型目标)
            if len(self.intent_history) >= 8:
                recent10 = self.intent_history[-8:]
                covered = set(recent10)
                if len(covered) <= 4 and random.random() < 0.4:
                    uncovered = [i for i in INTENTS if i not in covered and i != "HELP"]
                    if uncovered:
                        forced = random.choice(uncovered)
                        if forced == "CUSTOM":
                            cluster, cmd_args = self.cmd_selector.select()
                            params = {"custom_args": cmd_args, "cluster": cluster}
                        else:
                            params = self.param_extractor.extract(
                                forced, self.state_encoder.current_goal or "",
                                workbench=self.workbench,
                                known_files=self.state_encoder.explored_paths if hasattr(
                                    self.state_encoder, 'explored_paths') else None,
                            )
                        intent = forced
                        used_goal = True
                        self._last_action_source = "diversity"
                        if random.random() < 0.1:
                            print(f"  [DIVERSITY] 强制 {forced} (GoalGenerator 多样性调整)")

        # P17: 保持 LLM 异步生成管道运行 — 小模型决定意图, DeepSeek 执行
        if (hasattr(self, 'creative_writer')
                and self.creative_writer._async_enabled
                and getattr(self.creative_writer, '_async_result', None) is None
                and self.step_count % 5 == 0):
            try:
                # Feed WM5 prediction error as surprise signal to GoalGenerator
                self.goal_generator._recent_surprise = getattr(self, '_recent_surprise', [])
                self.goal_generator._last_surprise = getattr(self, '_last_wm_surprise', 0.0)
                self.goal_generator._last_kl = getattr(self, '_last_kl', 0.0)
                intention = self.goal_generator.decide_creative_intention(
                    self.workbench, self.step_count)
                self.creative_writer.generate_async(
                    self.workbench,
                    style=intention.get("style", "create"),
                    intention=intention["intention"],
                )
            except Exception:
                pass
        # P17: 自我方向 — 多信号决策方向 (完全不依赖 LLM)
        if (hasattr(self, 'self_model')
                and self.step_count > 30
                and self._self_remaining <= 0
                and self._adaptive_should("set_direction", 0.04)):
            self._stats_based_direction()
        # P17: 方向持久期 — 每步按方向选择
        if self._self_remaining > 0 and not used_goal:
            dir_intent, dir_params = self._direction_step()
            if dir_intent:
                intent = dir_intent
                params = dir_params
                used_goal = True
                self._last_action_source = "self_reflect"

        # 3. Fallback: 全局多样性 (15步未覆盖意图)
        if not used_goal and len(self.intent_history) >= 15:
            recent_all = self.intent_history[-15:]
            covered = set(recent_all)
            uncovered = [i for i in INTENTS if i not in covered and i != "HELP"]
            if uncovered:
                forced = random.choice(uncovered)
                if forced == "CUSTOM":
                    cluster, cmd_args = self.cmd_selector.select()
                    params = {"custom_args": cmd_args, "cluster": cluster}
                else:
                    params = self.param_extractor.extract(
                        forced, self.state_encoder.current_goal or "",
                        workbench=self.workbench,
                        known_files=self.state_encoder.explored_paths if hasattr(
                            self.state_encoder, 'explored_paths') else None,
                    )
                intent = forced
                used_goal = True
                self._last_action_source = "diversity"
                self.ab_stats["goal_driven"] = self.ab_stats.get("goal_driven", 0) + 1
                if random.random() < 0.1:
                    n_total = len(INTENTS) - 1
                    print(f"  [DIVERSITY] 强制 {forced} ({n_total-len(covered)}种未覆盖)")

        # 4. Fallback: 工作栏链式目标
        if not used_goal and self.workbench and self.workbench.has_active_goal():
            fu = self.workbench.get_current_goal()
            if fu:
                intent, params = fu
                params = dict(params)
                used_goal = True
                self._last_action_source = "goal_driven"
                self.ab_stats["goal_driven"] += 1
                cs = self.workbench.chain_step
                print(f"  [CHAIN] {intent} (链{cs+1}/3)")

        # 5. Fallback: 自生成目标
        if not used_goal and self.workbench and self.step_count > 10 and self._adaptive_should("self_goal", 0.15):
            self_goal = self.workbench.generate_self_goal()
            if self_goal:
                intent, params = self_goal
                params = dict(params)
                used_goal = True
                self._last_action_source = "goal_driven"
                self.ab_stats["goal_driven"] += 1
                if random.random() < 0.1:
                    print(f"  [SELF-GOAL] {intent} (事实缺口)")

        # 6. Fallback: 探针 (概率门控)
        if not used_goal and self.workbench and random.random() < self.probe_rate:
            if not hasattr(self, "_probe_find_count"):
                self._probe_find_count = 0
            probe = self.workbench.get_curiosity_probe(self.state_encoder.explored_paths)
            if probe:
                p_args = probe[1].get("custom_args", [])
                p_cmd = " ".join(str(a) for a in p_args) if isinstance(p_args, list) else str(p_args)
                if "find" in p_cmd:
                    self._probe_find_count += 1
                    if self._probe_find_count > 20:
                        probe = None
                if probe:
                    intent, params = probe
                    params = dict(params)
                    used_goal = True
                    self._last_action_source = "probe"
                    self.ab_stats["goal_driven"] += 1
                    if random.random() < 0.15:
                        print(f"  [PROBE] {params.get('custom_args', ['?'])}")

        # 7. Fallback: P9.7 想象力 (概率门控)
        if not used_goal and self.step_count > 3 and random.random() < self.imagination_rate:
            rnd_stats = self.rnd.get_novelty_stats()
            rnd_avg = rnd_stats.get('running_errors_avg', 0)
            curiosity_mode = rnd_avg < 0.01 and random.random() < 0.4
            if curiosity_mode:
                # 好奇心不足: 强制 CUSTOM 探索
                intent = "CUSTOM"
                self._scan_new_commands()
                untried = self._get_untried_custom_cmds()
                if untried:
                    cmd = random.choice(untried)
                    params = {"custom_args": [cmd], "cluster": "CURIOSITY"}
                else:
                    cluster, cmd_args = self.cmd_selector.select()
                    params = {"custom_args": cmd_args, "cluster": cluster}
                if random.random() < 0.05:
                    print(f"  [CURIOSITY] RND新颖度低({rnd_avg:.4f}), 强制CUSTOM探索")
                used_goal = True
                self._last_action_source = "imagination"
                self.ab_stats["imagined"] = self.ab_stats.get("imagined", 0) + 1
                self.ab_stats["goal_driven"] += 1
            else:
                temp = max(0.5, 2.0 - self.step_count * 0.003)
                imagined = self._imagine_intent(state_text, temperature=temp)
                if imagined and imagined not in ("HELP",):
                    intent = imagined
                    params = self.param_extractor.extract(
                        intent, self.state_encoder.current_goal or "",
                        workbench=self.workbench,
                        known_files=self.state_encoder.explored_paths if hasattr(self.state_encoder, 'explored_paths') else None,
                    )
                    if intent == "CUSTOM":
                        cluster, cmd_args = self.cmd_selector.select()
                        params = {"custom_args": cmd_args, "cluster": cluster}
                    used_goal = True
                    self._last_was_imagined = True
                    self._last_action_source = "imagination"
                    self.ab_stats["imagined"] = self.ab_stats.get("imagined", 0) + 1
                    self.ab_stats["goal_driven"] += 1
                if random.random() < 0.25:
                    print(f"  [IMAGINE] {intent}")

        if not used_goal:
            # Fallback: A/B 切换 + 指挥家/分类器
            used_conductor = False
            # P5.5: A/B 自适应采样率 (Wilson 下限, 小样本抑制)
            if self.conductor_path_active:
                n_cond = self.ab_stats["conductor"]
                n_clf = self.ab_stats["classifier"]
                cond_rate = self.ab_stats["conductor_success"] / max(n_cond, 1)
                clf_rate = self.ab_stats["classifier_success"] / max(n_clf, 1)
                # P7.2: 小样本抑制降低 n<3, 让 Conductor 早点上场
                if n_cond < 3:
                    p_conductor = 0.25  # 原0.1, 给更多试炼机会
                else:
                    p_conductor = 0.2 + 0.5 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))
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
                        # P5.5: gate 0.5, 非 0.7
                        if best_prob > 0.5:
                            raw_intent, nanny_params, _ = self.nanny.translate(thought, logits, state_text=state_text)
                            if raw_intent != "HELP":
                                intent = raw_intent
                                params = nanny_params
                                # P7.0: CUSTOM 由 Conductor 正常输出时, 补充命令
                                if intent == "CUSTOM" and "custom_args" not in params:
                                    cluster, cmd_args = self.cmd_selector.select()
                                    params = {"custom_args": cmd_args, "cluster": cluster}
                                used_conductor = True
                                self._last_action_source = "conductor"
                                self.ab_stats["conductor"] += 1
                                self._last_cond_logits = logits.detach().clone()
                        elif cond_intent not in ("HELP",):
                            clf_intent = self.classifier.predict(state_text)
                            if cond_intent == clf_intent:
                                raw_intent, nanny_params, _ = self.nanny.translate(thought, logits, state_text=state_text)
                                intent = raw_intent
                                params = nanny_params
                                # P7.0: CUSTOM 补充命令
                                if intent == "CUSTOM" and "custom_args" not in params:
                                    cluster, cmd_args = self.cmd_selector.select()
                                    params = {"custom_args": cmd_args, "cluster": cluster}
                                used_conductor = True
                                self.ab_stats["conductor"] += 1
                                self._last_cond_logits = logits.detach().clone()
                    except Exception as e:
                        if random.random() < 0.1:
                            print(f"  [CONDUCTOR-ERR] {e}")

            if not used_conductor:
                intent = self._select_intent(state_text)
                goal = self.state_encoder.current_goal
                if intent == "CUSTOM":
                    # P8.5d: 每50步扫描新命令
                    self._scan_new_commands()
                    # P5.6: 偶尔试未用过的命令
                    untried = self._get_untried_custom_cmds()
                    if untried and random.random() < 0.3:
                        cmd = random.choice(untried)
                        params = {"custom_args": [cmd], "cluster": "NOVEL"}
                    else:
                        cluster, cmd_args = self.cmd_selector.select()
                        params = {"custom_args": cmd_args, "cluster": cluster}
                else:
                    params = self.param_extractor.extract(
                        intent, goal,
                        workbench=self.workbench,
                        known_files=self.state_encoder.explored_paths if hasattr(self.state_encoder, 'explored_paths') else None,
                    )
                    # 3-intent 默认回退
                    if not params.get("path") and not params.get("custom_args") and not params.get("content"):
                        if intent == "OBSERVE":
                            params = {"path": "/etc/hostname"}
                        elif intent == "TRY":
                            cluster, cmd_args = self.cmd_selector.select()
                            params = {"custom_args": cmd_args, "cluster": cluster}
                        elif intent == "CREATE":
                            params = {"path": "/tmp/out.txt", "content": "generated"}
                self.ab_stats["classifier"] += 1

        # V3: P8.5c: 世界模型心理模拟 — 对Conductor/分类器选择做二次验证
        # 跳过目标驱动路径 (避免打断链式目标)
        if intent is not None and not used_goal and self.step_count > 3:
            try:
                # P8.5c: 扩展候选集, 让WM有更多选项
                candidates = [INTENTS.index(intent)]
                all_alts = ["OBSERVE", "CREATE", "TRY"]
                for alt in all_alts:
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
                # P8.5c: 更激进的覆盖 (2%优势就切换, 原5%)
                if best["total_value"] > orig_single["total_value"] * 1.02:
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

        # 命令推荐: 对 OBSERVE/TRY 检查已发现命令
        if intent in ("OBSERVE", "TRY") and hasattr(self, 'knowledge_mapper'):
            km = self.knowledge_mapper
            cmd_map = km.get_intent_command_map() if hasattr(km, 'get_intent_command_map') else {}
            if intent in cmd_map and cmd_map[intent]:
                if not hasattr(self, '_recommended_cmds'):
                    self._recommended_cmds: set[str] = set()
                # 随机选一个命令
                cmd = random.choice(cmd_map[intent])
                if cmd not in self._recommended_cmds:
                    self._recommended_cmds.add(cmd)
                    intent = "TRY"
                    params = {"custom_args": [cmd], "cluster": "SYSTEM"}
                    print(f"  [RECOMMEND] {cmd} → TRY")

        # P8.5b: 参数预校验 — 执行前替换无效参数
        params = self._rescue_params(intent, params)

        # 追踪已试过的 TRY 命令
        if intent == "TRY":
            try_cmd = str(params.get("custom_args", ""))[:40]
            self._tried_custom_cmds.add(try_cmd)
        
        # TRY 预验证 — 检查命令存在、参数合法性
        if intent == "TRY":
            params["custom_args"] = self._validate_custom(params.get("custom_args", []))

        # P18: WorldModel V5 — pre-action prediction for surprise computation
        _wm5_pred = None
        if hasattr(self, 'world_model_v5'):
            try:
                _i_idx = INTENTS.index(intent) if intent in INTENTS else 0
                _wm5_pred = self.world_model_v5.step(state_emb, _i_idx)
            except Exception:
                pass

        # 4. 执行 (多命令组合 P1)
        depth = params.get("depth", 1)
        multi_results = None
        all_exit_ok = False

        # P9.2: WRITE/APPEND 跳过 multi-command (使用安全写入模板)
        if depth > 1 and intent not in ("TRY", "EXPLORE", "CREATE"):
            try:
                multi_results = self.engine.execute_multi(intent, params, depth)
            except Exception:
                multi_results = None

        # P16 R1: Ensure _pre_state defined for all paths
        _pre_state = {}
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
            # P16 R1: Capture pre_state before execution
            _pre_state = {}
            wb = self.workbench
            if hasattr(wb, 'graph') and wb.graph:
                _pre_state = {k: n.value for k, n in wb.graph.nodes.items()}
            elif hasattr(wb, 'facts'):
                _pre_state = {k: v.get("value", "") for k, v in wb.facts.items()}
            result = self.engine.execute(intent, params)
            output = (result.stdout or result.stderr or "")
        
        # P5.1: 命令名 (用于发现日志)
        if intent == "CUSTOM":
            cmd_name = " ".join(str(a) for a in params.get("custom_args", []))
        else:
            cmd_name = str(params.get("path", params.get("cmd", intent)))
        
        # P8.0: 失败后自动恢复
        if result.exit_code != 0 and hasattr(self, 'error_recovery'):
            new_result, recovery_info = self.error_recovery.recover(
                result, intent, params, cmd_name
            )
            if new_result and new_result.exit_code == 0:
                result = new_result
                output = (result.stdout or result.stderr or "")
                if random.random() < 0.2:
                    print(f"  [RECOVER] {recovery_info.get('action', '?')} -> OK")

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
        # P5.5: 提取前记录事实数, 用于 success 判断
        facts_before = len(self.workbench.facts)
        discovery_before = self.workbench.get_current_discovery()
        if result.exit_code == 0:
            self.workbench.extract_facts(intent, cmd_name, output, params, self.step_count)

        # P16 R1: Capture post_state after fact extraction
        _post_state = {}
        _wb = self.workbench
        if hasattr(_wb, 'graph') and _wb.graph:
            _post_state = {k: n.value for k, n in _wb.graph.nodes.items()}
        elif hasattr(_wb, 'facts'):
            _post_state = {k: v.get("value", "") for k, v in _wb.facts.items()}

        # 6. 世界模型好奇心 + RND 新颖度
        next_state_text = self.state_encoder.get_state_text(thought_label=thought_label)

        # P18: WM5 surprise — prediction error as intrinsic motivation signal
        _wm5_surprise = 0.0
        if hasattr(self, 'world_model_v5') and _wm5_pred is not None:
            try:
                _next_emb = self.classifier.get_embedding(next_state_text).detach()
                _wm5_surprise = ((_wm5_pred["next_state"].detach() - _next_emb) ** 2).mean().item()
                if not hasattr(self, '_recent_surprise'):
                    self._recent_surprise = []
                self._recent_surprise.append(_wm5_surprise)
                if len(self._recent_surprise) > 200:
                    self._recent_surprise = self._recent_surprise[-200:]
                self._last_wm_surprise = _wm5_surprise
                self._last_post_emb = _next_emb
            except Exception:
                pass

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
        
        # RND 自适应重置 (基于新颖度状态)
        if self.step_count > 0:
            rnd_stats = self.rnd.get_novelty_stats()
            # 高新颖度: 偶尔重置, 低新颖度: 频繁重置
            reset_prob = max(0.01, min(0.1, 0.05 + (0.006 - rnd_stats['running_errors_avg']) * 10))
            if random.random() < reset_prob:
                self.rnd.interest_reset(fraction=0.3)
            # 软重置: 预测误差持续偏低时
            if rnd_stats['running_errors_avg'] < 0.006:
                self.rnd.soft_reset(factor=0.4)
                if random.random() < 0.01:
                    print(f"  [RND] 软重置 (avg={rnd_stats['running_errors_avg']:.4f})")

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

        # P5.5: 重复惩罚 key
        repeat_key = f"{intent}:{str(params)[:40]}"
        recent_repeats = sum(1 for k in self.intent_history[-10:] if k == intent)
        is_repeat = (repeat_key in getattr(self, "_recent_actions", set()))
        if not hasattr(self, "_recent_actions"):
            self._recent_actions = set()
        self._recent_actions.add(repeat_key)
        if len(self._recent_actions) > 50:
            self._recent_actions = set(list(self._recent_actions)[-30:])

        # 9. 计算奖励 (含链奖励 + 重复惩罚)
        reward = self._compute_reward(result, intent, combined_curiosity, intent_diversity,
                                      chain_bonus=chain_bonus, params=params,
                                      facts_before=facts_before)
        self.state_encoder.set_reward(reward)
        if is_repeat and recent_repeats > 2:
            reward *= 0.3  # 重复严重惩罚

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
        # P5.5: step_success 基础 = exit==0 + 有输出
        facts_after = len(self.workbench.facts)
        new_fact_this_step = facts_after > facts_before
        new_discovery = self.workbench.get_current_discovery() and \
                        self.workbench.get_current_discovery() != discovery_before
        step_success = (result.exit_code == 0) and bool(output)
        if step_success:
            self.success_count += 1
        # P17: SelfModel 记录
        if hasattr(self, 'self_model'):
            try:
                n_facts_new = max(0, len(self.workbench.facts) - facts_before)
                self.self_model.record(
                    intent=intent, success=step_success, reward=reward,
                    step=self.step_count,
                    output_len=len(output),
                    n_facts=n_facts_new,
                    path=params.get("path", ""),
                    content_len=len(params.get("content", "")),
                )
            except Exception:
                pass

        # P15: WM 自我反思 (预测 vs 实际)
        if hasattr(self, 'world_model_v4') and thought_vector is not None and intent:
            try:
                sim = self.world_model_v4.simulate(state_emb, thought_vector.unsqueeze(0), intent)
                self.world_model_v4.self_reflect(intent, sim, {
                    "exit_code": result.exit_code,
                    "reward": reward,
                    "success": step_success,
                })
            except Exception:
                pass
            # 去人为: 更新自适应奖励知识
            try:
                self._update_reward_knowledge(intent, reward)
            except Exception:
                pass
            # P7.1: 修复归因优先级 — imagination 独立于 used_goal
            if self._last_was_imagined:
                self.ab_stats["imagined_success"] += 1
            elif used_goal:
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

        # P9: 超级日志 — 每步记录
        try:
            self.logger.log_step(
                step=self.step_count, intent=intent, params=params,
                source=self._last_action_source,
                cmd_name=cmd_name if 'cmd_name' in dir() else '',
                output=output,
                exit_code=result.exit_code,
                reward=reward, novelty=combined_curiosity,
                diversity=intent_diversity,
                conductor_prob=p_conductor if 'p_conductor' in dir() else 0.0,
                facts_before=facts_before, facts_after=len(self.workbench.facts),
                ab_stats=self.ab_stats,
                rnd_state=self.rnd.get_novelty_stats(),
            )
        except Exception:
            pass

        # P11: PersistentStore 统一保存 (每10步)
        if self.step_count % 10 == 0:
            stats = {
                "run_id": self._run_id if hasattr(self, '_run_id') and self._run_id else f"run_{self.step_count}",
                "started_at": getattr(self, '_started_at', ''),
                "n_steps": self.step_count,
                "success_rate": self.success_count / max(self.step_count, 1),
                "n_intents_covered": len(set(self.intent_history)),
                "total_reward": self.total_reward,
                "fact_graph_nodes": len(self.workbench.graph.nodes) if hasattr(self.workbench, 'graph') else 0,
                "fact_graph_edges": sum(len(e) for e in self.workbench.graph.edges.values()) if hasattr(self.workbench, 'graph') else 0,
                "schema_coverage": self.workbench.graph.stats()['schema_coverage'] if hasattr(self.workbench, 'graph') else 0,
                "llm_calls": getattr(self.creative_writer, 'stats', {}).get('total_calls', 0) if self.creative_writer else 0,
                "llm_success": getattr(self.creative_writer, 'stats', {}).get('llm_success', 0) if self.creative_writer else 0,
                "llm_fallback": getattr(self.creative_writer, 'stats', {}).get('fallback', 0) if self.creative_writer else 0,
            }
            if not hasattr(self, '_run_id'):
                self._run_id = f"run_{self.step_count}"
            self.pstore.save_all(self, run_stats=stats)
            if hasattr(self, 'self_model'):
                self.self_model.save()

        # P5.1: 自引用 — 自适应频率
        if self.sandbox and self.step_count > 0 and self._adaptive_should("self_review", 0.03):
            try:
                r = self.sandbox.execute("tail -5 /tmp/discoveries.md 2>/dev/null || echo ''")
                if r.stdout and len(r.stdout) > 20:
                    self.workbench._add_fact("self_discovery", r.stdout.strip()[:80],
                        "system", "self_monitor", self.step_count, category="meta")
            except Exception:
                pass

        # LLM sync generation (自适应频率, 并行报告 + 可执行代码 + 自动重试)
        if self.creative_writer and self.step_count > 50 and self._adaptive_should("run_llm", 0.01):
            import threading
            def _llm_report():
                try:
                    import base64 as _b64
                    p = self.creative_writer.build_prompt(self.workbench, "report")
                    t = self.creative_writer.generate(p, timeout=120.0)
                    if t and len(t) > 100:
                        path = f"/tmp/llm_sync_{self.step_count}.md"
                        enc = _b64.b64encode(t.encode()).decode()
                        self.sandbox.execute(f"echo '{enc}' | base64 -d > {path}")
                        self.workbench._add_fact(f"llm_sync_{self.step_count}", t[:80], "LLM",
                            f"sync:{self.step_count}", self.step_count, category="content")
                        self._total_create_content += len(t)
                        print(f"  [LLM] report {len(t)}B -> {path}")
                except Exception as e:
                    print(f"  [LLM] report error: {e}")
            def _llm_code():
                try:
                    import base64 as _b64
                    p = self.creative_writer.build_prompt(self.workbench, "code")
                    t = self.creative_writer.generate(p, timeout=120.0)
                    if t and len(t) > 80:
                        path = f"/tmp/llm_code_{self.step_count}.py"
                        enc = _b64.b64encode(t.encode()).decode()
                        self.sandbox.execute(f"echo '{enc}' | base64 -d > {path}")
                        self.sandbox.execute(f"chmod +x {path}")
                        r = self.sandbox.execute(f"python3 {path} 2>&1", timeout=15)
                        out = (r.stdout or r.stderr or "").strip()
                        exit_ok = r and r.exit_code == 0
                        if exit_ok:
                            self._total_create_content += len(t)
                            self.workbench.extract_facts("LLM_CODE", path, out, {}, self.step_count)
                            self.total_reward += 2.0
                            print(f"  [LLM] code {len(t)}B -> {path} OK +2.0!")
                        else:
                            error_preview = out[:400]
                            retry_prompt = (
                                f"The Python script below failed with error:\n"
                                f"```\n{error_preview}\n```\n\n"
                                f"Fix the bug. Output ONLY the corrected Python code, no explanation.\n"
                                f"```python\n{t}\n"
                            )
                            t2 = self.creative_writer.generate(retry_prompt, timeout=120.0)
                            if t2 and len(t2) > 80:
                                path2 = f"/tmp/llm_code_{self.step_count}_retry.py"
                                # 剥离 retry 输出的围栏
                                if t2.startswith('```'):
                                    nli = t2.find('\n')
                                    if nli > 0: t2 = t2[nli+1:]
                                    if t2.rstrip().endswith('```'):
                                        t2 = t2[:t2.rfind('```')].rstrip()
                                t2 = t2.strip()
                                if not t2.startswith('#!'):
                                    t2 = '#!/usr/bin/env python3\n' + t2
                                enc2 = _b64.b64encode(t2.encode()).decode()
                                self.sandbox.execute(f"echo '{enc2}' | base64 -d > {path2}")
                                self.sandbox.execute(f"chmod +x {path2}")
                                r2 = self.sandbox.execute(f"python3 {path2} 2>&1", timeout=15)
                                out2 = (r2.stdout or r2.stderr or "").strip()
                                if r2 and r2.exit_code == 0:
                                    self._total_create_content += len(t2)
                                    self.workbench.extract_facts("LLM_CODE", path2, out2, {}, self.step_count)
                                    self.total_reward += 1.5
                                    print(f"  [LLM] code retry {len(t2)}B -> {path2} OK +1.5!")
                                else:
                                    self.workbench.extract_facts("LLM_CODE", path,
                                        t + "\n---\n" + error_preview,
                                        f"failed: {out2[:200]}", self.step_count)
                                    print(f"  [LLM] code {len(t)}B FAIL (retry also failed)")
                            else:
                                print(f"  [LLM] code {len(t)}B FAIL ({error_preview[:80]})")
                except Exception as e:
                    print(f"  [LLM] code error: {e}")
            threading.Thread(target=_llm_report, daemon=True).start()
            threading.Thread(target=_llm_code, daemon=True).start()
        # LLM async result check every 5 steps
        if self.creative_writer and self.step_count % 5 == 0:
            async_result = self.creative_writer.check_async_result()
            if async_result and async_result.get("source", "") == "llm":
                content = async_result.get("content", "")
                path = async_result.get("path", f"/tmp/llm_output_{self.step_count}.md")
                if content and self.sandbox and len(content) > 50:
                    # 剥离 Markdown 代码围栏 (```python ... ```)
                    if content.startswith('```'):
                        # 移除开头的 ``` 行 (可能 ```python, ```py, 或 ```)
                        first_newline = content.find('\n')
                        if first_newline > 0:
                            content = content[first_newline+1:]
                        # 移除结尾的 ``` 行
                        if content.rstrip().endswith('```'):
                            last_fence = content.rfind('```')
                            content = content[:last_fence].rstrip()
                    content = content.strip()
                    if path.endswith('.py') and not content.startswith('#!'):
                        content = '#!/usr/bin/env python3\n' + content
                    import base64
                    encoded = base64.b64encode(content.encode()).decode()
                    self.sandbox.execute(f"echo '{encoded}' | base64 -d > {path}")
                    key = f"llm_{self.step_count}"
                    self.workbench._add_fact(key, content[:80], "LLM",
                        f"async:{async_result.get('desc','content')}",
                        self.step_count, category="content")
                    self._total_create_content += len(content)
                    # P19: CodeArchive — 持久化代码
                    if hasattr(self, 'code_archive') and self.code_archive:
                        try:
                            from pathlib import Path
                            self.code_archive.save(
                                filename=Path(path).name,
                                content=content,
                                step=self.step_count,
                                idea=async_result.get('desc', '') or content[:200],
                                success=True,
                            )
                        except Exception:
                            pass

        # P5.4: 元学习 — 记录步效用 + 定期淘汰
        asrc = self._last_action_source
        
        # 记录探针效用 (Kimi: 成功=提取到新事实, 非 exit=0)
        if asrc == "probe":
            probe_cmd = params.get("custom_args", [intent])
            probe_path = " ".join(str(a) for a in probe_cmd)[:50]
            probe_id = f"probe_{probe_path[:40].replace(' ', '_')}"
            self.meta.register(probe_id, "probe_path",
                              {"path": probe_path}, self.step_count)
            # 提取到新事实 = 成功, 否则 = 弱失败
            probe_utility = 0.5 if (step_success and new_fact_this_step) else (-0.2 if step_success else -0.5)
            self.meta.record(probe_id, probe_utility, self.step_count)
        # 记录意图效用 (所有来源: classifier + conductor + imagination)
        if asrc in ("classifier", "conductor", "imagination"):
            self.meta.register(f"intent_{intent}_{asrc}", "intent_choice",
                              {"intent": intent, "source": asrc}, self.step_count)
            # 成功且产生新事实 = 高;
            # 成功但无新事实 = 边际;
            # 失败 = 负
            delta = 0.5 if (step_success and new_fact_this_step) else \
                    (0.1 if step_success else -0.4)
            self.meta.record(f"intent_{intent}_{asrc}", delta, self.step_count)
        # 自适应淘汰 + 保存 + 打印摘要
        meta_stats = self.meta.get_stats()
        meta_prob = 0.02 * (1.0 + meta_stats['total_behaviors'] / 50.0)
        if self.step_count > 0 and random.random() < meta_prob:
            pruned = self.meta.prune(min_trials=5, threshold=-0.2,
                                     max_age=300, current_step=self.step_count)
            self.meta.save()
            if pruned > 0:
                print(f"  [META] 淘汰 {pruned} 个低效行为. "
                      f"现存: {meta_stats['total_behaviors']} 个")

        # P16 R1: Record transition
        _transition = {
            "step": self.step_count,
            "pre_state": _pre_state if '_pre_state' in dir() else {},
            "action": intent,
            "params": str(params)[:200],
            "cmd": cmd_name if 'cmd_name' in dir() else intent,
            "post_state": _post_state if '_post_state' in dir() else {},
            "exit_code": result.exit_code,
            "output_len": len(output) if 'output' in dir() else 0,
            "had_new_facts": new_fact_this_step if 'new_fact_this_step' in dir() else False,
            "reward": reward,
        }
        self._transitions.append(_transition)
        if len(self._transitions) >= self._transition_flush_size:
            self._flush_transitions()

        # P18: WorldModel V5 — 数据收集 + 定时训练
        if hasattr(self, 'world_model_v5') and state_emb is not None:
            try:
                self._wm5_buffer.append({
                    "pre_emb": state_emb.detach().clone(),
                    "intent": INTENTS.index(intent) if intent in INTENTS else 0,
                    "reward": reward,
                    "continue": 1.0,
                    "post_emb": self._last_post_emb.detach().clone()
                        if hasattr(self, '_last_post_emb')
                        else state_emb.detach().clone(),
                    # P18 Phase 3: fact category diff (which categories got new facts)
                    "fact_cats": self._compute_fact_diff() if hasattr(self, 'workbench') else torch.zeros(20),
                })
                # P18 Phase 3: IntuitionBuffer 记录 (含成功/惊奇度)
                if hasattr(self, 'intuition_buffer'):
                    self.intuition_buffer.store(
                        thought=self.persistent_thought,
                        intent=INTENTS.index(intent) if intent in INTENTS else 0,
                        params=params,
                        reward=reward,
                        success=step_success,
                        surprise=getattr(self, '_last_wm_surprise', 0.0),
                        state_emb=state_emb.detach().clone(),
                        imagined=False,
                    )
                # P19: 想象回放 — 用 WM5.1 从当前状态 rollout 未来
                if (hasattr(self, 'imagination_engine')
                        and self.step_count % 10 == 0
                        and self.intuition_buffer.size >= 20):
                    try:
                        img_result = self.imagination_engine.rollout(
                            state_emb, first_intent=INTENTS.index(intent) if intent in INTENTS else 0)
                        if img_result.get('n_steps', 0) > 0:
                            print(f"  [IMAGINE] rolled {img_result['n_steps']} steps "
                                  f"mean_reward={img_result['mean_reward']:.3f}")
                    except Exception as e:
                        print(f"  [IMAGINE] error: {e}")
                # P19: 睡眠巩固 — 每 _sleep_interval 步微调 Conductor
                if (self.step_count > 0
                        and self.step_count % self._sleep_interval == 0):
                    try:
                        self._sleep_consolidation()
                    except Exception as e:
                        print(f"  [SLEEP] error: {e}")
                # P19: 更新无聊度
                self._update_boredom(reward, getattr(self, '_last_wm_surprise', 0.0))
                if len(self._wm5_buffer) > 600:
                    self._wm5_buffer = self._wm5_buffer[-600:]
                if (len(self._wm5_buffer) >= 20
                        and self.step_count % self._wm5_train_interval == 0):
                    buf = self._wm5_buffer
                    T = min(len(buf) - 1, self._wm5_batch_size)
                    states = torch.stack([b["pre_emb"] for b in buf[:T]])
                    actions = []
                    for i in range(T):
                        a_emb = self.world_model_v5.encode_action(
                            buf[i]["intent"], states[i:i+1])
                        actions.append(a_emb)
                    actions_t = torch.cat(actions, dim=0)
                    next_states = torch.stack([b["post_emb"] for b in buf[:T]])
                    rewards = torch.tensor([[b["reward"]] for b in buf[:T]])
                    cont = torch.tensor([[b["continue"]] for b in buf[:T]])
                    fact_t = torch.stack([b.get("fact_cats", torch.zeros(20)) for b in buf[:T]])
                    loss = self.world_model_v5.train_step(
                        states, actions_t, next_states, rewards, cont,
                        fact_targets=fact_t,
                        chunk_size=16,
                    )
                    # V5.1: track KL divergence as uncertainty signal
                    self._last_kl = self.world_model_v5.last_kl_value
                    if self.step_count % 50 == 0:
                        self.world_model_v5.save(self._wm5_checkpoint)
                    if self.step_count % 100 == 0:
                        print(f"  [WM5] train loss={loss:.4f}")
            except Exception as e:
                print(f"  [WM5] train error: {e}")

        return step_success, reward

    def train_step(self):
        """P8.3: 在线训练 — UCB采样 + LR调度"""
        if self.buffer.size < self.batch_size:
            return 0.0

        buffer_list = list(self.buffer.buffer)
        by_intent = {intent: [] for intent in INTENTS}
        for e in buffer_list:
            if e.intent in by_intent:
                by_intent[e.intent].append(e)

        train_intents = [i for i in INTENTS if i != "HELP"]
        
        # P8.3: UCB采样 — 少样本意图获得更多权重
        # P10: 约束 UCB ≤ 3.0, 防噪声样本权重过大导致 Loss 发散
        total = len(buffer_list)
        n_intents = len(train_intents)
        base_per_class = max(1, self.batch_size // n_intents)
        
        # 计算每个意图的 UCB 分数: 样本数越少, UCB 越高
        intent_scores = {}
        UCB_MAX = 3.0  # P10: 硬上限, 防止罕见意图噪声放大
        for intent in train_intents:
            n_samples = len(by_intent.get(intent, []))
            if n_samples == 0:
                intent_scores[intent] = UCB_MAX  # 无样本 → 最高优先级
            else:
                ratio = n_samples / max(total, 1)
                # UCB: 越少的类权重越大, 但上限 UCB_MAX
                intent_scores[intent] = min(1.0 / max(ratio, 0.01), UCB_MAX)
        
        # 归一化权重 → 样本分配数
        total_score = sum(intent_scores.values())
        batch = []
        for intent in train_intents:
            weight = intent_scores[intent] / total_score
            target_n = max(1, int(self.batch_size * weight))
            pool = by_intent.get(intent, [])
            if len(pool) >= target_n:
                batch.extend(random.sample(pool, target_n))
            else:
                batch.extend(pool)

        # 补充高奖励样本
        if len(batch) < self.batch_size:
            remaining = self.batch_size - len(batch)
            high_reward = self.buffer.sample_by_reward(remaining * 2, min_reward=0.5)
            high_reward = [e for e in high_reward if e.intent != "HELP"]
            batch.extend(high_reward[:remaining])

        random.shuffle(batch)
        batch = batch[:self.batch_size]

        texts = [e.state_text for e in batch]
        labels = [INTENTS.index(e.intent) if e.intent in INTENTS else 0 for e in batch]

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

        # P8.3: LR 调度 + 损失追踪
        self._train_losses.append(loss.item())
        if len(self._train_losses) >= 10:
            avg_loss = sum(self._train_losses[-10:]) / 10
            self.lr_scheduler.step(avg_loss)

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

        # P10: V4 双轨训练
        v4_samples = []
        for e in batch:
            s_emb = self.classifier.get_embedding(e.state_text)
            s_emb = (s_emb.detach().clone() if s_emb.is_inference() else s_emb.clone())
            out = e.output[:500] if e.output else ""
            ec = getattr(e, "exit_code", 0)
            thought_list = getattr(e, "thought", [])
            thought_t = torch.tensor(thought_list[:16] if thought_list else [0]*16)
            iname = e.intent if e.intent in self.world_model_v4.leaves else "CUSTOM"
            lower = out.lower()
            v4_samples.append({
                "state_emb": s_emb,
                "thought": thought_t,
                "intent_name": iname,
                "exit_cls": 0 if ec == 0 else 1,
                "length_cls": 0 if len(out) < 100 else (1 if len(out) < 1000 else 2),
                "error_cls": 1 if any(kw in lower for kw in ("not found","error","denied","no such file")) else 0,
                "value": getattr(e, "reward", 0.0),
            })
        v4_loss = self.world_model_v4.train_on_buffer(v4_samples)
        self._wm_loss_history.append(v4_loss if v4_loss > 0 else 0.0)
        # 自适应保存 V4 checkpoint
        _save_prob = 0.02 if len(self._wm_loss_history) < 5 else 0.02 + 0.03 * min(1.0, abs(v4_loss - self._wm_loss_history[-1]) / 0.1)
        if random.random() < _save_prob:
            import os as _os
            _os.makedirs("checkpoints/world_model", exist_ok=True)
            self.world_model_v4.save("checkpoints/world_model/v4_latest.pt")
            if self._v4_ready() and not self.episodic_memory.is_enabled():
                self.episodic_memory.enable()
                print(f"  \u2705 情景记忆已启用 (V4 ready)")

        # P9: 训练日志
        try:
            intent_counts = {intent: len(by_intent.get(intent, []))
                           for intent in INTENTS if intent != "HELP"}
            self.logger.log_training(
                step=self.step_count if hasattr(self, 'step_count') else 0,
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr'],
                intent_counts=intent_counts,
                buffer_size=self.buffer.size,
                n_cond=self.ab_stats["conductor"],
                n_clf=self.ab_stats["classifier"],
            )
        except Exception:
            pass

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
                # P9: 坍缩检测
                if loss > 3.0:
                    self.logger.log_alert(
                        step=i, level="warning",
                        message=f"训练loss异常: {loss:.4f}",
                        metrics={"loss": loss, "cond_loss": cond_loss}
                    )
                if verbose:
                    rnd_stats = self.rnd.get_novelty_stats()
                    cond_info = f"  |  Conductor对齐={cond_loss:.4f}" if cond_loss > 0 else ""
                    print(f"  [{i:4d}] 训练 loss={loss:.4f}  |  "
                          f"新颖度 avg={rnd_stats['running_errors_avg']:.4f}  |  "
                          f"缓冲区 {self.buffer.size}{cond_info}")

            # C: RND 自适应软重置 (基于新颖度状态)
            if i > 0 and random.random() < 0.01:
                rnd_stats = self.rnd.get_novelty_stats()
                if rnd_stats['running_errors_avg'] < 0.008:
                    self.rnd.soft_reset(factor=0.3)
                    if verbose:
                        print(f"  [RND] 新颖度略低({rnd_stats['running_errors_avg']:.4f}), 软重置")
            # P6.3: 自适应全量重置
            if i > 0 and random.random() < 0.005:
                self.rnd.reset()
                if verbose:
                    print(f"  [RND] 自适应全量重置")

            # 自适应显示统计
            if verbose and random.random() < 0.1:
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

    def _expand_intents(self, stable_new_intents: list[dict]):
        """P6.4: 将稳定新意图接入系统 (INTENTS + 分类头 + WM + Conductor + 训练)"""
        old_effective = len([i for i in INTENTS if i != "CUSTOM"])

        # 1. 追加到 INTENTS (末尾, 不改变已有索引)
        new_names = []
        for ni in stable_new_intents:
            name = ni['name']
            if name not in INTENTS:
                INTENTS.append(name)
                import agent.nanny as nanny
                if name not in nanny.INTENTS:
                    nanny.INTENTS.append(name)
                new_names.append(name)

        if not new_names:
            return

        # P10: 更新 N_INTENTS 全局常量
        import agent.conductor as conductor_mod
        import agent.nanny as nanny_mod
        new_n = len(INTENTS)
        # 更新模块级常量
        global N_INTENTS
        N_INTENTS = new_n
        conductor_mod.N_INTENTS = new_n
        nanny_mod.N_INTENTS = new_n
        # 更新 IntentClassifier.N_INTENTS (类属性)
        IntentClassifier.N_OUT = new_n

        # 2. 扩展 Conductor 分类头 (通过 nanny 访问, 仅在可用时)
        effective = len(INTENTS)  # P10: 使用完整意图数 (含 CUSTOM)
        if hasattr(self, 'nanny') and self.nanny is not None:
            self.nanny.conductor.expand_intents(effective)

        # 3. 扩展世界模型 (含 CUSTOM)
        total_intents = len(INTENTS)
        self.world_model.expand_intents(total_intents)

        # 4. 扩展 IntentClassifier 分类头
        self.classifier.expand_intents(effective)

        # 5. 为新意图注册自适应奖励 (初始值 0.5)
        for name in new_names:
            if hasattr(self, '_reward_tracker'):
                self._reward_tracker[name] = []
            if hasattr(self, '_adapted_rewards'):
                self._adapted_rewards[name] = 0.5

        # 6. 用 discoverer 的真实轨迹生成训练数据, 增量训练
        training_data = self._generate_training_for_new_intents(new_names)
        if training_data:
            self._quick_train_new_intents(training_data)

        # 7. 保存所有扩展后的 checkpoint
        if hasattr(self, 'nanny') and self.nanny is not None:
            self.nanny.conductor.save(
            os.path.join("checkpoints", "conductor", "online_aligned.pt"))
        self.world_model.save(
            os.path.join("checkpoints", "world_model", "latest.pt"))
        self.classifier.save(
            os.path.join("checkpoints", "intent_classifier", "best_head.pt"))

        print(f"  [SYSTEM] 接入 {len(new_names)} 个新意图: {', '.join(new_names)}")
        print(f"  [SYSTEM] 有效意图: {old_effective} → {effective}, 总意图: {total_intents}")

    def _generate_training_for_new_intents(self, new_names: list[str]) -> list:
        """P6.4: 从 discoverer 的真实轨迹生成新意图训练样本"""
        if not hasattr(self.intent_discoverer, 'trajectories'):
            return []
        samples = []
        # 新意图对应的命令基名 (从 discoverer 的 cmd_intent_map 反查)
        rev = {v: k for k, v in {
            "cat": "READ_ETC", "file": "CHECK_TYPE", "stat": "FILE_STAT",
            "du": "DISK_USAGE", "ps": "PROCESS_LIST", "mount": "MOUNT_INFO",
            "lspci": "PCI_DEVICES", "lsusb": "USB_DEVICES", "lsmod": "KERNEL_MODULES",
            "dns": "DNS_CONFIG", "env": "ENV_VARS", "timedatectl": "TIME_INFO",
        }.items()}
        for name in new_names:
            cmd_base = rev.get(name, "")
            for t in self.intent_discoverer.trajectories:
                if t.get("cmd_base", "") == cmd_base:
                    intent_id = INTENTS.index(name) if name in INTENTS else -1
                    samples.append({
                        "state_text": t["state_text"],
                        "intent": name,
                        "intent_id": intent_id,
                    })
        return samples

    def _quick_train_new_intents(self, training_data: list):
        """P6.4: 用生成的数据对新意图做小批量训练"""
        if not training_data:
            return
        import torch
        import torch.nn.functional as F

        self.classifier.head.train()
        optimizer = torch.optim.AdamW(
            self.classifier.head.parameters(), lr=1e-3, weight_decay=1e-4)

        embs = []
        labels = []
        for s in training_data:
            emb = self.classifier.encoder.encode(
                s["state_text"], convert_to_tensor=True, device=self.device)
            embs.append(emb)
            labels.append(s["intent_id"])

        if not embs:
            return

        X = torch.stack(embs).to(self.device)
        y = torch.tensor(labels, device=self.device)

        for epoch in range(20):
            optimizer.zero_grad()
            logits = self.classifier.head(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.head.parameters(), 1.0)
            optimizer.step()

        self.classifier.head.eval()
        print(f"  [TRAIN] 新意图增量训练: {len(training_data)} 样本 + 20 epoch")

    def _try_expand_intents(self):
        """P6.4: 检查并扩展新意图 (供 run() 调用)"""
        if not hasattr(self.intent_discoverer, '_known_intents'):
            return
        self.intent_discoverer.filter_known(INTENTS)
        candidates = self.intent_discoverer.discover()
        stable = [c for c in candidates if c['n_samples'] >= 5 and c['name'] not in INTENTS]
        if stable:
            self._expand_intents(stable)

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

        # P1.5: 自适应采样率 (对齐运行时公式)
        n_cond = self.ab_stats["conductor"]
        cond_rate = self.ab_stats["conductor_success"] / max(n_cond, 1)
        clf_rate = self.ab_stats["classifier_success"] / max(self.ab_stats["classifier"], 1)
        if n_cond < 5:
            p_cond = 0.1
        else:
            p_cond = 0.2 + 0.5 * max(0, min(1.0, cond_rate / max(clf_rate, 0.01)))
        print(f"  A/B 自适应: p_conductor={p_cond:.0%}  (Conductor胜率={cond_rate:.0%} vs 分类器胜率={clf_rate:.0%})")
        # P8.0: 失败恢复统计
        try:
            rec = self.error_recovery.get_stats()
            if rec['recovery_attempts'] > 0:
                print(f"  恢复: {rec['recovery_success']}/{rec['recovery_attempts']} ({rec['recovery_rate']:.0%})")
                blk = self.error_recovery.get_blocked()
                print(f"  黑名单: {rec['blocked_cmds']}命令, {rec['blocked_paths']}路径")
        except Exception:
            pass
        print(f"{'=' * 45}")

        # P9: 关闭日志
        try:
            self.logger.log_snapshot(
                step=self.step_count,
                success_rate=result['success_rate'],
                intent_dist=result['intent_distribution'],
                facts=list(self.workbench.facts.keys()),
            )
            self.logger.close()
        except Exception:
            pass

        return result
