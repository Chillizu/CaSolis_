# 闭环系统架构

```
Online Agent Loop
══════════════════════════════════════════════════

  Environment
    │  (目录, 文件, 历史命令)
    ▼
  StateEncoder ──────────────────────────┐
    │  (构建 state_text)                  │
    ▼                                    │
  Brain (Classifier)                     │
    │  (intent)                          │
    ▼                                    │
  ParameterExtractor                     │
    │  (params)                          │
    ▼                                    │
  TemplateEngine                         │
    │  (command)                         │
    ▼                                    │
  Execution (subprocess.run)             │
    │  (stdout, stderr)                  │
    ▼                                    │
  StateEncoder (next_state) ─────────────┤
    ▼                                    │
  RND ──────────────────────────────     │
    │  (novelty_bonus)                   │
    ▼                                    │
  Experience Buffer ─────────────────────┤
    │  (state, intent, params,           │
    │   output, reward, next_state)       │
    ▼                                    │
  Online Training ───────────────────────┘
    │  (fine-tune classifier head)
    ▼
  Better Intent Selection

模块:
  agent/state_encoder.py  环境 → state_text
  agent/rnd.py            RND 好奇心模块
  agent/experience.py     经验回放缓冲区  
  agent/online_agent.py   主循环
  scripts/train_online.py 在线训练脚本
```
