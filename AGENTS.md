# Folunar_ — 项目规则与架构

> 最后更新: 2026-06-21

---

## 工作规则

| 规则 | 说明 |
|------|------|
| Kimi 咨询决策 | **技术决策优先问 Kimi 而不是用户。用 subagent 调用 kimi-coding/kimi-for-coding，Kimi 的回复可直接执行，除非它要求用户确认。报告时向用户总结做了什么和为什么** |
| 自主研究 | **需要查资料/论文时用 web_search / fetch_content / librarian 工具搜索，不要每步都问用户** |
| 汇报风格 | 做完事直接汇报结果，不需要每一步都征求许可。只有改变用户偏好的决策才需要问
| 分步实施 | 大改动拆成 P0→P1→P1.5→P2→... 的渐进步骤，每步完成后再规划下一步 |
| 先读后写 | 改代码前先读完整文件，理解上下文再动手 |
| 任务跟踪 | 用 todo 工具跟踪每一步进度 |

---

## 闭环系统架构

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
