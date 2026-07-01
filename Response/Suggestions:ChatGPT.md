我不建议你直接生成 2000~3000 条。

原因很简单：

**你现在最缺的不是数据量，而是数据分布设计。**

如果我现在帮你生成 3000 条，大概率会变成：

```text
READ:
  当前目录:/etc
  发现 passwd
  -> READ

READ:
  当前目录:/etc
  发现 hostname
  -> READ

READ:
  当前目录:/etc
  发现 shadow
  -> READ
```

实际上只是同一个样本换词。

Sentence Transformer 很快就能记住：

```text
passwd => READ
hostname => READ
```

而不是学会：

```text
什么时候该 READ
什么时候该 SEARCH
```

---

我建议先设计 Dataset Grammar。

这是很多 Agent 项目最容易忽略的部分。

---

# 先定义状态空间

你的状态应该由：

```python
{
    cwd,
    visible_files,
    previous_action,
    previous_result,
    history,
    discovered_commands,
    current_goal
}
```

组成。

而不是：

```python
{
    cwd,
    visible_files
}
```

否则分类器其实在做：

```text
目录名 -> 意图
```

映射。

---

# 8类意图

我建议：

```text
READ
SEARCH
LIST
STATUS
COUNT
INSPECT
FILTER
EXPLORE
```

先不要 PIPE。

PIPE 实际是组合规划。

不是基础意图。

---

# 每类应该覆盖的模式

## READ

触发条件：

```text
发现具体文件
需要看内容
```

例如：

```text
当前目录:/etc
发现:
passwd
hostname
hosts

上一步:
ls

目标:
查看用户配置
```

→ READ

---

## SEARCH

触发条件：

```text
已经有文件
需要找关键词
```

例如：

```text
当前目录:/etc

已知:
passwd

目标:
寻找root账户
```

→ SEARCH

---

## LIST

触发条件：

```text
不知道目录里有什么
```

例如：

```text
当前目录:
/opt

未探索

历史:
无
```

→ LIST

---

## STATUS

触发条件：

```text
需要系统状态
```

例如：

```text
目标:
检查CPU占用

当前信息:
未知
```

→ STATUS

---

## COUNT

触发条件：

```text
统计数量
```

例如：

```text
目标:
统计日志行数

已知:
access.log
```

→ COUNT

---

## INSPECT

触发条件：

```text
不知道命令能力
```

例如：

```text
发现命令:
jq

历史:
从未使用
```

→ INSPECT

---

## FILTER

触发条件：

```text
已有输出
需要处理
```

例如：

```text
上一步:
cat users.txt

结果:
1000行

目标:
去重
```

→ FILTER

---

## EXPLORE

触发条件：

```text
没有明确目标
寻找新能力
```

例如：

```text
当前状态:
已完成所有任务

已探索:
/

目标:
无
```

→ EXPLORE

---

# 真正重要的部分

你应该故意制造混淆样本。

例如：

---

样本1

```text
当前目录:/etc

已知文件:
passwd

目标:
查看文件内容
```

→ READ

---

样本2

```text
当前目录:/etc

已知文件:
passwd

目标:
查找root
```

→ SEARCH

---

样本3

```text
当前目录:/etc

已知文件:
passwd

目标:
统计用户数量
```

→ COUNT

---

样本4

```text
当前目录:/etc

已知文件:
passwd

目标:
提取用户名
```

→ FILTER

---

注意：

环境完全一样。

标签不同。

---

这样模型才会学：

```text
目标决定意图
```

而不是：

```text
passwd => READ
```

---

# 我会怎么做

与其：

```text
8类 × 250条
```

不如：

```text
8类 × 100条基础样本
+
8类 × 150条对抗样本
```

总计：

```text
2000条
```

---

其中：

### 1000条

普通样本

---

### 1000条

刻意混淆样本

例如：

```text
同目录
同文件
不同目标
```

```text
同目标
不同历史
```

```text
同历史
不同环境
```

---

这种数据的价值会远远超过 3000 条模板替换数据。

如果我是这个项目的数据负责人，我会要求：

> 数据集至少有 30% 样本属于“人第一眼都会犹豫”的情况。

因为这些样本才真正定义了分类边界。

否则你训练出来的不是“大脑”，而是一个目录名匹配器。
