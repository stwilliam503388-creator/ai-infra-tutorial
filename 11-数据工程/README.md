---
title: 第11章 数据工程
tags: ['ai-infra', '教程', '数据', 'dvc']
章节: 11
状态: 完成
---

# 第11章：数据工程

> 数据工程 = 食材采购和预处理。没有好食材，再好的厨师也做不出好菜。

## 11.1 问题：大模型训练需要大量数据，怎么准备？

你想训练一个中文客服机器人，但：
- 没有现成的中文客服对话数据
- 网上爬的数据格式乱七八糟
- 不知道数据质量好不好

**核心矛盾**：数据决定模型上限，但数据准备最耗时间。

## 11.2 方案：数据工程五步

```
采集 → 清洗 → 标注 → 增强 → 版本控制
```

| 步骤 | 一句话 | 工具 |
|------|--------|------|
| 采集 | 从各种来源获取原始数据 | 爬虫、API、公开数据集 |
| 清洗 | 去掉脏数据、重复数据 | Pandas、正则表达式 |
| 标注 | 给数据打标签 | Label Studio、人工 |
| 增强 | 扩充数据量 | 同义词替换、回译 |
| 版本控制 | 记录数据版本变化 | DVC、Git-LFS |

## 11.3 项目

| 项目 | ⭐ | 定位 | 一句话 |
|------|-----|------|--------|
| **HuggingFace datasets** | 21.9k | 数据集库 | 现成的 NLP 数据集 |
| **DVC** | 14k | 数据版本控制 | Git for Data |
| **Label Studio** | 20k | 数据标注平台 | 可视化标注界面 |
| **Snorkel** | 6k | 弱监督标注 | 用规则自动标注 |

## 11.4 代码示例

### 数据清洗脚本

```python
import pandas as pd
import re

def clean_text(text):
    """清洗文本数据"""
    # 去掉 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去掉多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 去掉特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    return text

# 加载数据
df = pd.read_csv('raw_data.csv')

# 清洗
df['clean_text'] = df['text'].apply(clean_text)

# 去重
df = df.drop_duplicates(subset=['clean_text'])

# 去掉空值
df = df.dropna(subset=['clean_text'])

# 保存
df.to_csv('clean_data.csv', index=False)
print(f"清洗完成：{len(df)} 条数据")
```

### DVC 数据版本控制

```bash
# 初始化 DVC
dvc init

# 添加数据文件
dvc add data/train.csv

# 提交到 Git
git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1.0"

# 修改数据后
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Update training data v1.1"

# 回溯到旧版本
git checkout HEAD~1 data/train.csv.dvc
dvc checkout
```

## 11.5 排障：数据质量问题

| 问题 | 现象 | 根因 | 解法 |
|------|------|------|------|
| NaN 值 | 训练报 NaN | 数据中有空值 | 清洗时 dropna 或填充 |
| 重复数据 | 模型过拟合 | 同一条数据出现多次 | drop_duplicates |
| 格式错误 | 解析报错 | CSV 引号/换行符问题 | 用 `pandas.read_csv` 的 `error_bad_lines=False` |
| 数据不平衡 | 模型偏向多数类 | 正负样本比例悬殊 | 过采样/欠采样/类别权重 |

## 11.6 动手实验：用 DVC 管理一个数据集

**目标**：体验数据版本控制的完整流程。

**步骤**：

```bash
# 1. 创建项目目录
mkdir my-data-project && cd my-data-project
git init
dvc init

# 2. 创建示例数据
echo "id,text,label" > data.csv
echo "1,你好,正面" >> data.csv
echo "2,再见,负面" >> data.csv

# 3. 用 DVC 追踪
dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "v1.0: 初始数据集"

# 4. 修改数据
echo "3,谢谢,正面" >> data.csv
dvc add data.csv
git add data.csv.dvc
git commit -m "v1.1: 新增一条数据"

# 5. 查看历史
git log --oneline
```

**预期输出**：
- `data.csv.dvc` 文件记录数据哈希
- `git log` 显示两个版本

**常见错误**：
- DVC 未初始化 → 先运行 `dvc init`
- 文件太大 → 用 `dvc add` 而不是 `git add` 管理大数据

## 11.7 自测题

1. DVC 和 Git 的区别是什么？（提示：Git 管代码，DVC 管数据）

<details>
<summary>参考答案</summary>

Git 适合管理代码（文本文件，体积小，变化频繁）。DVC 适合管理数据（大文件，体积大，变化相对少）。DVC 把数据文件存到远程存储（S3/MinIO），Git 只存元数据（.dvc 文件）。
</details>

2. 数据清洗的常用步骤有哪些？（提示：去重、去空值、格式统一）

<details>
<summary>参考答案</summary>

①去重（drop_duplicates）②去空值（dropna）③格式统一（正则表达式）④文本清洗（去 HTML、去特殊字符）⑤标准化（大小写、全半角）
</details>

3. 为什么需要数据版本控制？（提示：实验可复现）

<details>
<summary>参考答案</summary>

模型训练结果依赖于数据版本。如果数据变了但代码没变，实验结果不可复现。DVC 记录数据版本，配合 Git 记录代码版本，实现"代码+数据"的完整可复现。
</details>

---

**上一章**：[第10章：开源框架昇腾适配](../10-开源框架昇腾适配/README.md) | **下一章**：[第12章：模型压缩](../12-模型压缩/README.md)
