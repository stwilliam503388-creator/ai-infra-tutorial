---
title: 第13章 模型评测
tags: ['ai-infra', '教程', '评测', '基准']
章节: 13
状态: 完成
---

# 第13章：模型评测

> 模型评测 = 给学生出考卷。考卷出得好，才能知道学生真水平。

## 13.1 问题：模型好不好，怎么客观评价？

你训练了一个模型，但：
- 自己感觉不错，但不知道和别人比怎么样
- 用户反馈时好时坏，没有统一标准
- 老板说"再优化一下"，但不知道优化什么

**核心矛盾**：没有客观评测，就没有优化方向。

## 13.2 方案：三种评测方式

| 方式 | 一句话 | 优点 | 缺点 |
|------|--------|------|------|
| **基准测试** | 用标准数据集考试 | 客观、可复现 | 可能过拟合 |
| **人类评估** | 真人打分 | 最接近真实体验 | 贵、慢、主观 |
| **自动评估** | 用另一个模型评分 | 快、便宜 | 不够准确 |

## 13.3 项目

| 项目 | ⭐ | 定位 | 昇腾支持 |
|------|-----|------|---------|
| **OpenCompass** | 6k | 中文评测框架 | ✅ 支持昇腾 |
| **lm-evaluation-harness** | 10k | 英文评测框架 | ✅ 可通过 vLLM-Ascend 评测 |
| **HELM** | 3k | 斯坦福评测框架 | ⚠️ 学术为主 |
| **FlagEval** | 13 | 启智评测 | ✅ 昇腾原生 |

## 13.4 代码示例

### 用 lm-evaluation-harness 评测

```bash
# 安装
pip install lm-eval

# 评测 GPT-2 在 WikiText 上的困惑度
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks wikitext \
    --device cuda:0
```

### 用 OpenCompass 评测

```bash
# 安装
pip install opencompass

# 评测 Qwen
opencompass --models qwen --datasets ceval
```

### 自定义评测脚本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def evaluate(model_name, test_cases):
    """简单评测脚本"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    results = []
    for case in test_cases:
        inputs = tokenizer(case['input'], return_tensors='pt')
        outputs = model.generate(**inputs, max_length=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'input': case['input'],
            'expected': case['expected'],
            'actual': answer,
            'correct': case['expected'] in answer
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"准确率：{accuracy:.2%}")
    return results

# 测试用例
test_cases = [
    {'input': '中国的首都是', 'expected': '北京'},
    {'input': '1+1=', 'expected': '2'},
]

evaluate('Qwen/Qwen2-0.5B-Instruct', test_cases)
```

## 13.5 排障：评测常见问题

| 问题 | 现象 | 根因 | 解法 |
|------|------|------|------|
| 评测结果不稳定 | 每次跑分数不一样 | 温度参数太高 | 设 temperature=0 |
| 基准测试过拟合 | 在测试集上表现好，实际差 | 训练数据泄露 | 用未见过的测试集 |
| 评测太慢 | 跑一轮要几小时 | 模型太大/数据太多 | 抽样评测、用更小模型 |

## 13.6 动手实验：用 OpenCompass 评测 Qwen

**目标**：体验完整的模型评测流程。

**步骤**：

```bash
# 1. 安装
pip install opencompass

# 2. 查看支持的模型和数据集
opencompass --list-models
opencompass --list-datasets

# 3. 评测 Qwen 在 C-Eval 上的表现
opencompass --models qwen --datasets ceval --work-dir ./output

# 4. 查看结果
cat output/results/summary.csv
```

**预期输出**：
```
Model          Dataset    Score
Qwen2-0.5B     ceval      0.45
```

**常见错误**：
- 下载数据集失败 → 检查网络，或手动下载放到指定目录
- 显存不够 → 换更小的模型，或加 `--max-num-workers 1`

## 13.7 自测题

1. 基准测试和人类评估的优缺点是什么？（提示：客观可复现 vs 接近真实体验）

<details>
<summary>参考答案</summary>

基准测试：客观、可复现、便宜，但可能过拟合、不接近真实体验。人类评估：最接近真实体验，但贵、慢、主观。实际中通常两者结合：先用基准测试筛选，再用人类评估验证。
</details>

2. 为什么评测时要设 temperature=0？（提示：结果可复现）

<details>
<summary>参考答案</summary>

temperature 控制随机性，temperature>0 时每次输出可能不同，导致评测结果不稳定。设 temperature=0 后模型输出确定，评测结果可复现。
</details>

3. 什么是基准测试过拟合？（提示：训练数据泄露）

<details>
<summary>参考答案</summary>

模型在训练时见过测试数据，导致在测试集上表现异常好，但实际能力并没有那么强。避免方法：用严格未见过的测试集、定期更换测试集。
</details>

---

**上一章**：[第12章：模型压缩](../12-模型压缩/README.md) | **下一章**：[第14章：模型部署进阶](../14-模型部署进阶/README.md)
