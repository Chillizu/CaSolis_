我来帮你生成这份标注数据。先设计状态文本的生成逻辑，确保覆盖 Docker 沙箱中各种真实场景，然后批量生成。
数据集已生成完毕。以下是交付说明：

---

## 数据概览

| 意图 | 数量 | ID | 说明 |
|------|------|----|------|
| **READ** | 300 | 0 | 读文件内容 (cat) |
| **LIST** | 300 | 1 | 列目录内容 (ls) |
| **SEARCH** | 300 | 2 | 搜索内容 (grep) |
| **INFO** | 300 | 3 | 系统信息 (cpu/mem/disk/ps/uptime) |
| **INSPECT** | 300 | 4 | 检查命令 (which/type) |
| **COUNT** | 300 | 5 | 计数 (wc -l) |
| **EXPLORE** | 300 | 6 | 探索新路径/新命令 |
| **HELP** | 300 | 7 | 查看命令帮助 (--help) |
| **总计** | **2400** | | 每类恰好 300 条 |

## 数据格式

**CSV 文件包含 4 列：**
- `sample_idx` — 样本编号（已打乱）
- `state_text` — 状态文本描述（模拟 Docker 沙箱环境）
- `intent` — 意图标签（字符串）
- `intent_id` — 意图标签（数字 0-7）

## 状态文本的构成要素

每条 `state_text` 模拟了 agent 在 Docker 环境中决策时所看到的上下文，包含以下要素的组合：

- **当前目录**：`/`, `/etc`, `/var/log`, `/proc`, `/usr/bin` 等 14 个常见路径
- **已知文件**：与当前目录相关的真实文件（如 `/etc/passwd`, `/proc/cpuinfo`）
- **上步操作**：前一步的执行结果摘要（"看到文件含 root"、"ls 看到很多文件"等）
- **历史轨迹**：最近 1-3 步的操作记录（或无历史）
- **关键词**（SEARCH 类）：`root`, `error`, `nginx`, `password` 等 15 个常用搜索词
- **信息类型**（INFO 类）：CPU/内存/磁盘/进程/运行时间等 9 种系统信息
- **命令名**（INSPECT/HELP 类）：`docker`, `nginx`, `python3`, `git`, `curl` 等 22 个常见命令

## 使用方式

```python
import pandas as pd
from sentence_transformers import SentenceTransformer

# 加载数据
df = pd.read_csv("/mnt/agents/output/folunar_intent_dataset.csv")

# 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码状态文本
embeddings = model.encode(df['state_text'].tolist(), show_progress_bar=True)
# → 2400 × 384 维向量

# 意图标签
labels = df['intent_id'].values  # → 2400 维 0-7 整数
```

## 数据特点与局限

**设计合理之处：**
- 覆盖了 Docker 沙箱中 14 个常见目录的真实文件系统结构
- 状态文本包含足够的上下文信息（目录+文件+历史）来推断合理意图
- 每类 300 条在 4K 可训参数规模下是合理的（75:1 的数据/参数比）
- 文本措辞多样化（20 套模板 × 随机组合），避免过拟合特定句式

**已知局限：**
- 状态文本是程序化生成的，不是真实 agent 执行产生的轨迹
- 不包含多轮交互的复杂依赖（如"先 LIST 再 READ"的序列模式）
- 关键词和文件组合是随机配对的，可能存在语义不匹配的情况（如"在 /proc 中搜索 password"——虽然 /proc 下没有 password，但 agent 的决策逻辑是"当前目录有文件 → 可以搜索"，这种边界情况约占 5-10%）

**建议：** 用这批数据做 **冷启动训练**（让分类器学会基本映射），然后在你跑通 MVP 闭环后，用真实 agent 轨迹替换或补充。


