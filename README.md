# CaSolis_ — 多脑自主 Linux Agent

一个在多脑架构上自主探索的 Linux agent。核心是一组训练好的神经网络（66K-0.5M 参数），模拟不同脑区的认知功能——前额叶做模式选择、杏仁核判断显著性、基底神经节做习惯、海马体做情景记忆、顶叶做世界模型预测。LLM 只做编码翻译。

## 架构

```
训练模型（大脑）            LLM（手）
─────────────────────     ───────────
MetaCognitiveSelector     qwen3.5:0.8b
IntentClassifier          DeepSeek (可选)
Conductor                 OpenRouter (可选)
WorldModel V4/V5.1
EpisodicMemory
ImaginationEngine
SelfModel
```

训练模型层**做所有决策**：选模式、定意图、预测结果、发现缺口、生成假设。LLM 层只把训练模型的"想法"翻译成可执行 Python 代码，不参与主循环决策。

## 核心模块

|模块|功能|参数|
|---|---|---|
|IntentClassifier|状态→3类元意图 (OBSERVE/CREATE/TRY)|~66K|
|Conductor|16维 thought 向量生成|~67K|
|GrowingWorldModel V4|逐意图预测 + 自动扩展|核+16叶|
|WorldModel V5.1|随机隐状态 + 前向预测 + KL 惊喜|~410K|
|RND|新颖度检测|~0.3M|
|MetaCognitiveSelector|模式选择 (EXPLORE/CREATE/LEARN)|~70K|
|SalienceSignal (杏仁核)|整合新颖度/惊喜/成功信号|0|
|HabitSystem (基底神经节)|高置信度时绕过 deliberation|0|
|ImaginationEngine|内部推演|~410K|
|SelfModel|自我意识+自省|~60K|

## 运行

```bash
cd ~/Projects/CaSolis_

# 单次测试 (120秒)
docker rm -f casolis-sandbox 2>/dev/null
source .venv/bin/activate
timeout 120 python3 -u doc/script.py

# 长程跑
nohup python3 -u scripts/marathon.py > marathon.log 2>&1 &

# 查看状态
source .venv/bin/activate
python3 -c "
from agent.persistent_store import PersistentStore
import sqlite3
db = sqlite3.connect('data/persistent/casolis.db')
c = db.cursor()
c.execute('SELECT run_id, n_steps, success_rate FROM run_stats')
for r in c.fetchall(): print(f'{r[0]}: {r[1]} steps, {r[2]:.0%} success')
"

# 清理沙箱
docker rm -f casolis-sandbox
```

## 环境要求

- Python 3.10+
- Docker（sandbox 用 `--network none` 隔离）
- Ollama（qwen3.5:0.8b，CPU 推理 ~2s/次）
- 无 GPU 要求

## 设计原则

- 训练模型决策，LLM 只翻译
- 3 个元意图（非 17 个具体意图）—— 空间固定，新行为不需重训
- 418 命令池自发现（非手写列表）
- 自适应概率（非 step%N 硬编码）
- CPU 推理，4 线程限制
- 持久化：SQLite + PyTorch checkpoint

## 历史

- **2026-07-02**: 改名 CaSolis_ (原 Folunar_)。打通训练模型→LLM 代码翻译链路
- **2026-06-25**: P15 脑启发动态层完成（去人为化）
- **2026-06-24**: P16 推理层完成（因果挖掘+假设实验+验证）
- **2026-06-23**: P17 自我意识层完成（SelfModel + LLM 自省）
- **2026-06-22**: P18-P19 随机隐变量世界模型 + 想象睡眠巩固
- **2026-06-21**: P20 人脑参考架构（杏仁核+基底神经节+丘脑门控）

详见 `CHANGELOG.md`。
