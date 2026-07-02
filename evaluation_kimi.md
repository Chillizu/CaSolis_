## 评估报告 (2026-06-24)

评估对象: CaSolis_ 自治 Linux Agent (P8 完成版本)
评估根目录: /home/chillizu/Projects/CaSolis_
数据来源: agent/*.py、config/command_registry.json、benchmark/template_engine.py、exec_log.jsonl（最近 100 步）

---

### 1. 自主性: 7/10

**优势**
- 决策链路完整: StateEncoder → 分类器/Conductor/Nanny → TemplateEngine → Docker 沙箱执行 → Workbench 事实提取 → RND 好奇心 → World Model 想象力 → ExperienceBuffer → 在线训练, 形成真正闭环。
- 0 用户输入设计: 目标池、工作栏推荐、探针、想象力评分均内建, 启动后可自行运行 100+ 步。
- 自我维护: 分类头、Conductor class_proj、世界模型输入层均可自动扩展; MetaLearner 追踪 131+ 行为并执行淘汰; ReduceLROnPlateau 自动调节学习率。
- 失败恢复自动化: ErrorRecovery 对命令不存在、文件不存在、权限拒绝、不支持的参数等 4 类错误有回退映射, 并将失败命令/路径加入黑名单避免重复踩坑。

**劣势**
- 40% 的行动仍以失败告终(exec_log 显示 exit_code ≠ 0 约占 4 成), 且存在明显的模式化失败: `INFO` 意图在 registry 中标记为 special=info_cmds 却仍触发「不支持的意图」; `COUNT`/`SEARCH`/`READ` 多次收到 path="CUSTOM" 这类非法参数; `USB_DEVICES` 反复调用不存在的 `lsusb`。
- 目标池与回退映射仍大量依赖人工预设: 13 个预定义探索目标、40+ 命令回退表、黑名单都是先验知识, 系统并未真正从零生成可行动作空间。
- 工作栏推荐的 follow_up 常固定指向 CUSTOM, 导致循环在「事实 → CUSTOM → hostname」这类低信息增益路径上打转。

**评分理由**: 闭环完整且工程化程度高, 但失败率偏高、部分失败属于可避免的低级参数错误, 说明自主性尚未达到「稳健自运行」级别。

---

### 2. 创造性: 5/10

**优势**
- 机制层面具备创造性要素: RND 新颖度奖励、World Model 心理模拟(imagine_top_k)、IntentDiscoverer 从 CUSTOM 轨迹聚类、CommandMiner 挖掘系统命令、Workbench 脚本生成。
- Conductor 输出 16 维 thought vector, Nanny 将其映射为参数微调(如 SEARCH 中 thought dim[0] 高激活时切换 pattern), 理论上支持非离散探索。
- CUSTOM 占用 17%, 说明系统确实在利用自由探索通道; 部分 CUSTOM 命令如 `cat /proc/modules` 产出了长而有信息量的输出。

**劣势**
- CUSTOM 使用质量低: 在最近 100 步中,CUSTOM 多数重复执行 `hostname`、`cat /etc/group`、`cat /etc/hostname` 等已在模板库中的命令, 并未真正生成新的行为模式。
- 想象力机制(World Model 评分)虽有代码, 但日志中未见其显著改变意图分布; A/B p_conductor=70% 更多是指挥家与分类器之间的切换, 而非由世界模型驱动的新颖策略。
- 新颖度均值 0.003 极低, RND 已高度适应当前状态分布, 好奇心奖励几乎失效, 系统倾向于重复已知路径。
- 意图自动发现(min_trajectories=30)门槛过高, 100 步内通常无法触发, 导致「发现新意图」停留在潜力阶段。

**评分理由**: 创造性基础设施齐全, 但实际输出以重复、低熵行为为主, 尚未形成持续的新型行为流。

---

### 3. 创新性: 4/10

**优势**
- 架构有创新: 67K 参数的 Conductor 作为「想法向量生成器」、World Model 同时预测 next_thought / value / agreement / exit / length / error 六个头、A/B 门控在分类器与指挥家之间动态切换, 这些设计在小模型自治 Agent 中具有实验价值。
- 元学习器与淘汰机制、动态探针、工作栏事实链式追踪, 体现了自我改进的尝试。

**劣势**
- 没有发现新的问题解决路径: 100 步日志中的「发现」局限于读取 /etc/passwd、/proc/cpuinfo、/etc/group 等标准系统文件, 所有产物都是常规系统侦察结果。
- 对失败的响应缺乏创新: 面对 `path="CUSTOM"` 或 `INFO` 不支持这类系统性错误, 系统没有自我诊断并修改 TemplateEngine 或参数提取器, 而是持续重复同样错误。
- 意图扩展虽然工程上自动化, 但新意图名称由硬编码 cmd_intent_map 决定(如 cat→READ_ETC), 并未真正涌现出超出人类预设类别的概念。

**评分理由**: 架构创新性明显, 但行为层面尚未突破预设模板和已知系统侦察任务, 缺乏真正的「从零发现」。

---

### 4. 创造力: 4/10

**优势**
- Workbench 能自动提取并持久化事实: hostname、kernel、cpu_cores、node_name、os_version、磁盘使用、目录内容等, 形成可查询的知识库。
- 支持脚本/产物生成: discoveries.md 在 /tmp 中被创建并追加; WRITE/APPEND 意图可输出文件; 元学习器记录行为效用并持久化到 data/persistent/metadata/meta.json。
- 多命令执行能力强: INFO、READ、SEARCH 等意图会组合多条命令并输出结构化结果。

**劣势**
- 产物价值有限: 生成的文件主要是系统信息的罗列, 没有分析、归纳、策略文档或自动化脚本。
- 事实提取规则高度手工化: Workbench 中为 uname、os-release、free、df、ls、cpuinfo、passwd 等写了大量专门正则, 系统并未自行学会新的提取模式。
- 没有证据表明系统创造出了新的工具、新的命令组合策略或新的问题分解方法; 所有「创造」都是对已有模板的实例化。

**评分理由**: 有产物输出能力, 但产物的信息价值与创造性较低, 尚未达到「生成有用新工具/知识」的水平。

---

### 5. 能力: 6/10

**优势**
- 工具使用广度好: command_registry.json 覆盖 READ/LIST/SEARCH/COUNT/INSPECT/EXPLORE/ARCH_INFO/USB_DEVICES/DISK_USAGE 等 14 种意图, custom_commands 含 28 条备用命令, 多命令模板支持管道。
- 执行层安全: Docker 沙箱、subprocess.run(args=...)、SAFE_PATHS、BLOCKED_COMMANDS、危险参数过滤, 具备基础隔离能力。
- 恢复能力: 错误恢复模块对 40+ 缺失命令提供替代方案, 分类器 85% 胜率, 指挥家 100% 胜率, 恢复率 88%。
- 工程化良好: 模块化代码、checkpoint 自动选择、经验缓冲、训练日志、执行日志 JSONL 化, 便于迭代。

**劣势**
- 实际成功率仅 60%, 与「稳健 Agent」差距明显; 多次出现同一错误连续发生(如连续 3 次读取 /proc 目录失败、连续多次执行 hostname)。
- 参数提取器存在系统性 bug: `params["path"] = "CUSTOM"` 被直接传入 wc/grep/cat, 说明参数未经过意图-参数一致性校验。
- 多命令 46% 的占比下, 仍有大量单步命令产出空转或重复结果; 长程目标坚持不足, 目标驱动 82% 但多为短目标轮转。
- 对动态环境的适应能力有限: 沙箱内缺少 python3、lsusb、ip 等命令, 系统反复尝试而非快速学习并跳过, 说明行为级元学习未能有效抑制无效行为。

**评分理由**: 工具广度和恢复机制较强, 但稳健性和执行正确率不足, 参数校验与长程一致性仍需加强。

---

### 整体总结

| 维度 | 评分 | 关键结论 |
|------|------|----------|
| 自主性 | 7/10 | 闭环完整、工程化好, 但失败率与重复错误削弱完全自主可信度。 |
| 创造性 | 5/10 | 机制丰富, 实际行为重复、低新颖度, CUSTOM 未产生高质量新行为。 |
| 创新性 | 4/10 | 架构有实验价值, 行为未突破预设模板, 未真正发现新问题解决路径。 |
| 创造力 | 4/10 | 能产出事实/文件, 但价值有限, 未生成新工具或深度分析。 |
| 能力 | 6/10 | 工具广、恢复快, 但成功率 60%、参数错误频发、长程一致性弱。 |

**总体星级评价: ★★★☆☆ (3/5)**

CaSolis_ 已经搭建了一个具备自感知、自决策、自学习、自恢复雏形的自治 Agent 框架, 在小模型自治 Linux Agent 的方向上完成了大量工程探索。然而,P8 的实际运行数据表明, 系统仍处于「机制齐全但行为未收敛」的阶段: 好奇心奖励耗尽、CUSTOM 探索低质量、参数错误反复出现、产物价值不高。下一步的关键不是继续堆叠模块, 而是提升执行正确率、强化参数校验、让 World Model 真正驱动意图选择, 并让工作栏产生更高阶的链式任务而非简单事实罗列。
