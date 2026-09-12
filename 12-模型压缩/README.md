---
title: 第12章 模型压缩
tags: ['ai-infra', '教程', '量化', '压缩']
章节: 12
状态: 完成
---

# 第12章：模型压缩

> 模型太大、部署成本高？量化、剪枝、蒸馏、ONNX 一条龙变小

## 12.1 问题：模型太大，部署成本高，怎么变小？

### 现象

你训练/下载了一个 7B 模型，部署时发现：

```text
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB
```

算一笔账——**7B 模型在不同精度下的显存占用**：

| 精度 | 每参数字节 | 7B 模型权重 | 加 KV Cache/激活 ≈ 实际需求 |
|------|-----------|------------|---------------------------|
| FP32 | 4 B | 28 GB | ~40 GB（2 张 A100） |
| FP16/BF16 | 2 B | 14 GB | ~20 GB（1 张 A100） |
| INT8 | 1 B | 7 GB | ~10 GB（1 张 A10） |
| INT4 | 0.5 B | 3.5 GB | ~6 GB（消费级显卡/手机） |

### 为什么模型会这么大？

1. **参数多**：7B = 70 亿个参数，每个参数都要存一个数
2. **精度冗余**：默认 FP32 存一个数要 4 字节，但推理时权重其实不需要这么高的精度
3. **模型冗余**：神经网络里大量参数是"锦上添花"，删掉一部分对精度影响很小
4. **格式不兼容**：PyTorch 模型是通用格式，部署到特定硬件（手机/嵌入式/特定 GPU）需要转换

### 核心矛盾

**精度 vs 体积/速度**。压缩的本质是：在精度可接受的范围内，尽可能去掉冗余。

---

## 12.2 方案：四大压缩技术

### 方案对比总表

| 技术 | 原理 | 压缩比 | 精度损失 | 难度 | 适用场景 |
|------|------|--------|---------|------|---------|
| **量化** | 降低权重数值精度（FP16→INT8/INT4） | 2-8x | 小（INT8）/中（INT4） | 低 | 最通用，几乎必用 |
| **剪枝** | 删掉不重要的权重/神经元 | 1.5-3x | 中 | 高（需重训练） | 结构化剪枝、瘦身 |
| **蒸馏** | 大模型（教师）教小模型（学生） | 取决于学生大小 | 小（训练得好时） | 高（需训练） | 直接换小模型 |
| **ONNX** | 转换到跨框架中间格式 | 0-2x（本身不压缩） | 极小 | 低 | 部署兼容性、加速器 |

> 注意：**ONNX 本身不压缩**，它是"格式转换"，但常常是部署量化的前提（很多硬件加速器只吃 ONNX）。

### 12.2.1 量化（Quantization）——最常用

**核心思想**：用更少的比特位表示一个数。好比人民币不需要精确到"分"的时候，可以只保留"元"。

**精度层级**：

```
FP32 (4B) → FP16/BF16 (2B) → INT8 (1B) → INT4 (0.5B)
   训练精度      推理标配     部署主流     极致压缩
```

**两种量化时机**：

| 类型 | 做法 | 是否需要数据 | 典型工具 |
|------|------|-------------|---------|
| **PTQ（训练后量化）** | 训练完直接量化，用少量校准数据 | 少量（几百条） | GPTQ、AWQ、torch.ao |
| **QAT（量化感知训练）** | 训练时就模拟量化误差 | 全量训练数据 | PyTorch、TensorRT |

**INT8 量化的数学原理**（对称量化）：

```
原始权重范围:  [-max, max]        (FP32)
量化后范围:    [-127, 127]        (INT8)

scale = max / 127
q = round(x / scale)              # 量化
x ≈ q * scale                     # 反量化
```

### 12.2.2 剪枝（Pruning）

**核心思想**：神经网络里很多权重接近 0，删掉它们对输出影响极小。

```text
剪枝前:  权重矩阵 [4x4] = 16 个参数
剪枝后:  权重矩阵 [4x4] = 9 个参数（删掉 7 个接近 0 的，置 0）
```

- **非结构化剪枝**：单独删零散的权重 → 矩阵变稀疏，但硬件不一定加速（需要稀疏计算库）
- **结构化剪枝**：整行/整列/整个通道删 → 矩阵变小，硬件友好，但精度损失更大

### 12.2.3 蒸馏（Distillation）

**核心思想**：教师模型（大）教学生模型（小）。学生学的不只是正确答案，还有教师输出的"分布"（软标签）。

```text
教师模型 (7B) ----软标签（概率分布）----> 学生模型 (1.5B)
                蒸馏损失 = α * 硬标签损失 + (1-α) * 软标签损失
```

- 小模型学会"模仿"大模型的判断方式，比直接用小模型训练效果更好
- 代表方法：KD（知识蒸馏）、DistilBERT、MiniLM

### 12.2.4 ONNX 转换

**核心思想**：把 PyTorch/TF 模型导出成**跨框架、跨硬件**的中间表示。

```text
PyTorch (.pt) ──┐
TensorFlow (.pb) ──┤→ ONNX (.onnx) ──→ ONNX Runtime / TensorRT / 手机端
PaddlePaddle ──┘
```

- ONNX Runtime 可以在 CPU 上获得 1.5-3x 加速
- 配合 INT8 量化后可在特定硬件上获得 3-5x 加速
- 支持动态/静态形状，导出后不依赖 PyTorch 环境

### 选型建议

| 场景 | 推荐组合 |
|------|---------|
| 本地/小显存跑大模型 | 量化：GGUF（llama.cpp）或 GPTQ |
| 生产服务化部署（GPU） | vLLM + AWQ/GPTQ 量化 |
| 跨平台部署（CPU/手机/嵌入式） | ONNX + ONNX Runtime |
| 极致性能（特定 GPU） | TensorRT（含 INT8/FP16） |

---

## 12.3 项目：主流工具介绍

### 12.3.1 llama.cpp —— 让大模型在笔记本/手机上跑

- **定位**：C++ 实现的 LLM 推理引擎，主打 CPU 推理 + 低内存
- **核心**：GGUF 格式（量化后模型文件格式，支持 2-8 bit 量化）
- **用法**：
```bash
# 1. 下载量化好的 GGUF 模型（HuggingFace 上搜 "GGUF"）
# 2. 直接用 llama.cpp 推理
./llama-cli -m models/qwen2-7b-instruct-q4_k_m.gguf -p "你好"
```
- **量化工具**：`llama-quantize`（把 FP16 GGUF 转成 INT4/INT8 GGUF）

### 12.3.2 GPTQ —— 经典训练后量化方法

- **定位**：基于二阶梯度（Hessian 矩阵）的 PTQ 方法，逐层量化，误差补偿
- **特点**：量化质量高，INT4 下精度损失小，是目前 4-bit 量化的主流之一
- **用法**：`auto-gptq` 库 / HF `GPTQConfig`

### 12.3.3 AWQ —— 激活感知量化

- **定位**：MIT 提出的激活感知权重量化（Activation-aware Weight Quantization）
- **核心思想**：不是所有权重都同等重要——只看 1% 对激活影响大的权重，保护它们不量化
- **特点**：INT4 量化质量好，且**量化后不需要校准集重训**，vLLM 等框架原生支持
- **用法**：`autoawq` 库 / HF `AwqConfig`

### 12.3.4 ONNX Runtime —— 跨平台推理引擎

- **定位**：微软出品的跨平台推理引擎（CPU/GPU/手机/网页都支持）
- **特点**：吃 ONNX 格式，自带图优化 + 算子融合，CPU 上也有明显加速
- **用法**：
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(None, {"input_ids": input_ids})
```

### 12.3.5 TensorRT —— NVIDIA 专属极致优化

- **定位**：NVIDIA 出品的深度学习推理优化器，把模型编译成 GPU 专属的 engine
- **特点**：算子融合 + INT8/FP16 量化 + 内核自动调优，性能天花板最高
- **代价**：只支持 NVIDIA GPU，编译时间长，部署灵活性差

### 12.3.6 msAMCT —— 昇腾专属模型压缩工具

- **定位**：华为昇腾模型压缩工具（Ascend Model Compression Toolkit）
- **特点**：支持 INT8/FP16 量化、通道剪枝，与 CANN 深度集成
- **场景**：昇腾 NPU 上的模型压缩，AI 推理服务器/一体机部署
- **对比**：对标 NVIDIA 的 TensorRT 量化方案

### 工具对比表

| 工具 | 格式 | 压缩方法 | 目标硬件 | 典型加速 | 上手难度 |
|------|------|---------|---------|---------|---------|
| llama.cpp | GGUF | 量化 | CPU/消费级 GPU/手机 | 2-4x | ⭐ |
| GPTQ | safetensors + 量化参数 | 量化（INT4/INT8） | GPU | 2-4x | ⭐⭐ |
| AWQ | safetensors + 量化参数 | 量化（INT4） | GPU | 2-4x | ⭐⭐ |
| ONNX Runtime | ONNX | 图优化 + 可选量化 | CPU/GPU/跨平台 | 1.5-5x | ⭐⭐ |
| TensorRT | engine | 图优化 + 量化 + 调优 | NVIDIA GPU | 3-10x | ⭐⭐⭐⭐ |
| **msAMCT** | 昇腾专用 | 量化 + 剪枝 | **昇腾 NPU** | 2-4x | ⭐⭐⭐ |

---

## 12.4 代码：GPTQ 量化示例 + ONNX 导出示例

### 12.4.1 GPTQ 量化示例（PyTorch + auto-gptq）

```python
# install: pip install auto-gptq transformers datasets
# 注意：量化需要 GPU，且模型较大时耗时较长

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"   # 换成你的模型
QUANT_BITS = 4                           # 4-bit 量化
QUANT_SAVE_DIR = "./qwen2-1.5b-gptq-int4"

# 1. 定义量化配置（指定 bits、数据集）
quant_config = GPTQConfig(
    bits=QUANT_BITS,                     # 量化位数：4 或 8
    dataset="c4",                        # 校准数据集（可选 wikitext2/c4）
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID),
    group_size=128,                      # 权重分组的粒度，越小精度越好但文件越大
    damp_percent=0.01,                   # 数值稳定性参数，一般不用动
    desc_act=True,                       # 按激活大小降序处理列，精度更好
)

# 2. 加载原模型并量化（会自动下载模型 + 校准数据集）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",                   # 自动分配到可用 GPU
    quantization_config=quant_config,
)

# 3. 保存量化后的模型
model.save_pretrained(QUANT_SAVE_DIR)
AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(QUANT_SAVE_DIR)
print(f"✅ 量化完成，已保存到 {QUANT_SAVE_DIR}")

# 4. 对比模型大小
import os
def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1024**3

print(f"量化后模型大小: {dir_size(QUANT_SAVE_DIR):.2f} GB")
# 预期：Qwen2-1.5B 原 FP16 约 3.0 GB → INT4 约 0.9 GB（约 1/3）
```

**加载量化模型推理**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./qwen2-1.5b-gptq-int4",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./qwen2-1.5b-gptq-int4")

inputs = tokenizer("介绍一下量化是什么", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

> 显存不够跑 GPTQ 量化？小模型（1.5B/3B）+ 单卡即可。大模型（7B+）建议直接用 HuggingFace 上别人量化好的权重。

### 12.4.2 ONNX 导出示例（PyTorch → ONNX）

```python
# install: pip install transformers onnx onnxruntime
# 导出 LLM 推荐用 optimum 的 export 工具，这里演示最小原理

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

# 1. 准备示例输入（导出需要给一个具体形状做"模板"）
dummy_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)  # [batch, seq_len]

# 2. 导出
torch.onnx.export(
    model,
    (dummy_input,),                    # 示例输入（位置参数）
    "qwen2-0.5b.onnx",                 # 输出文件
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={                     # 动态维度：batch 和 seq_len 不固定
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=17,                  # 算子集版本，老硬件选低版本更兼容
    do_constant_folding=True,          # 常量折叠优化
)

print("✅ 导出完成: qwen2-0.5b.onnx")

# 3. 用 ONNX Runtime 推理（不再需要 PyTorch）
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "qwen2-0.5b.onnx",
    providers=["CPUExecutionProvider"],   # 无 GPU 也能跑
)
inputs = tokenizer("你好", return_tensors="np")["input_ids"]  # [1, n]
outputs = session.run(None, {"input_ids": inputs.astype(np.int64)})[0]
next_token = outputs[0, -1].argmax()
print("下一个 token:", tokenizer.decode([next_token]))
```

> LLM 完整导出（含 KV Cache、past_key_values）建议直接用 `optimum-cli export onnx`，官方工具已处理这些细节。

---

## 12.5 排障：量化后精度下降 & ONNX 转换失败

### 坑 1：量化后精度明显下降

**典型现象**：
- 回答质量变差、乱码、重复输出
- 数学/代码能力断崖式下降（这类任务对精度最敏感）

**排查顺序**：

| 步骤 | 检查项 | 修复方法 |
|------|--------|---------|
| 1 | 用的是不是 INT4？ | 先试 INT8，精度不够再上 INT4 |
| 2 | 校准数据集合不合适？ | 用**和部署场景同分布**的数据做校准（如代码模型用代码校准） |
| 3 | 有没有用 group_size？ | 从 128 降到 64/32，精度更好（文件稍大） |
| 4 | 有没有异常权重？ | 检查权重是否有极端离群值（scale 会被它们拉大） |
| 5 | 量化的层范围？ | 敏感层（如 attention 的 QKV）可以跳过量化或保持高精度 |

**关键经验**：
- **INT8 基本无损，INT4 有损但可用**。7B 模型 INT4 通常掉 1-3 个点，可接受
- 量化后**必须跑评测**（如 perplexity、任务准确率），不要只看能不能跑
- 数学/推理类任务对量化最敏感 → 这类场景优先 AWQ（比 GPTQ 略好）或保留 FP16
- 蒸馏模型再量化会叠加误差，慎用

### 坑 2：ONNX 转换失败

**典型报错**：

```text
RuntimeError: Unsupported operator: aten::scaled_dot_product_attention
```

或

```text
ValueError: Unsupported opset version: 19
```

**根因与解法**：

| 报错 | 根因 | 解法 |
|------|------|------|
| `Unsupported operator` | PyTorch 算子在 ONNX 里没有对应实现 | 降 opset 版本、换 PyTorch 版本、用 `torch.onnx.export` 的 `custom_ops` 自定义映射 |
| `opset version` 报错 | 算子集版本太新，目标推理器不支持 | 降到 11-14 试试（兼容性最好） |
| `ONNX Runtime 报 shape 错误` | 动态维度没配好 | `dynamic_axes` 里显式声明 batch/seq_len |
| `导出很慢 / 内存爆` | 模型太大，trace 全部算子 | 用小模型测试流程；LLM 用 `optimum-cli export onnx` |
| `推理结果和 PyTorch 不一致` | 算子融合/精度差异 | 比较 logits 差异；关掉 `do_constant_folding` 试试 |

**通用排查步骤**：
1. 先用**最小模型**（0.5B/随机权重小模型）跑通导出流程，确认代码没问题
2. 导出后先 `onnxruntime` 跑一遍，对比 PyTorch 输出是否一致
3. 用 `onnx.checker.check_model(model)` 检查 ONNX 图是否合法
4. LLM 类模型优先用官方工具：`optimum-cli export onnx --model <model_id> <out_dir>`

---

## 12.6 动手实验：用 GPTQ 量化一个模型

**目标**：把 Qwen2-1.5B 从 FP16 量化为 INT4，对比大小和输出效果。

**前置条件**：有 GPU（8GB+ 显存即可）、已装 `pip install auto-gptq transformers`、网络能访问 HuggingFace（否则配 `HF_ENDPOINT=https://hf-mirror.com`）。

### 步骤 1：量化前基线

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

inputs = tokenizer("12 只鸡和 5 只鸭，一共有多少只脚？", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80)
print("基线输出:", tokenizer.decode(out[0], skip_special_tokens=True))

# 记录基线显存
print(f"基线显存占用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

**预期输出**：模型能正确算出 12×2 + 5×2 = 34 只脚，显存约 4-5 GB。

### 步骤 2：运行 12.4.1 的 GPTQ 量化代码

```bash
python quantize_gptq.py    # 把 12.4.1 的代码存成脚本运行
```

**预期输出**：`✅ 量化完成，已保存到 ./qwen2-1.5b-gptq-int4`，目录大小约 0.9 GB（原 3.0 GB）。

### 步骤 3：量化后对比

```python
# 用 12.4.1 的"加载量化模型推理"代码跑同一个问题
# 然后对比显存
print(f"量化后显存占用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

**预期结果**：

| 指标 | FP16 基线 | INT4 量化后 | 变化 |
|------|----------|------------|------|
| 模型文件大小 | ~3.0 GB | ~0.9 GB | -70% |
| 推理显存 | ~4-5 GB | ~2 GB | -50%+ |
| 输出质量 | 正常 | 基本一致（可能个别数字错） | 可接受 |

### 步骤 4：踩坑练习

故意把 `dataset="wikitext2"` 换成一个小众数据集，或把 `group_size=32`，观察：
- 校准数据差异对输出质量的影响
- group_size 变小后文件大小和精度的变化

**常见错误**：
- `auto-gptq` 和 `transformers` 版本不兼容 → 一起升级：`pip install -U auto-gptq transformers`
- 显存不够跑 7B 量化 → 先换 1.5B 模型，或直接下载现成量化权重
- 校准数据集下载失败 → 检查网络，设置 `HF_ENDPOINT=https://hf-mirror.com`

---

## 12.7 自测题

1. 为什么 INT8 量化后模型文件变小了，但模型参数个数没变？（提示：每参数字节数）

<details>
<summary>参考答案</summary>

参数个数没变，但**每个参数的表示精度变了**：FP32 每参数 4 字节 → INT8 每参数 1 字节。7B 模型从 28 GB 降到 7 GB，就是 70 亿个参数 × (4-1) 字节。量化是"降低精度"，不是"删参数"。
</details>

2. GPTQ 和 AWQ 都是 INT4 训练后量化，它们的主要区别是什么？（提示：权重视角 vs 激活视角）

<details>
<summary>参考答案</summary>

GPTQ 从**权重**角度出发，用二阶梯度（Hessian）做误差补偿，逐层量化；AWQ 从**激活**角度出发，找出对激活输出影响最大的 ~1% 权重（salient weights），量化时保护它们保持高精度。两者都能达到 INT4 高质量量化，AWQ 校准更简单（无需反向传播），vLLM 原生支持两者。
</details>

3. ONNX 本身能压缩模型吗？那它的价值是什么？（提示：格式 vs 优化）

<details>
<summary>参考答案</summary>

ONNX 本身**不压缩**（文件大小基本不变），它做的是**格式转换 + 图优化**：①把 PyTorch/TF 模型转成跨框架、跨硬件的中间格式；②ONNX Runtime 会做算子融合、常量折叠等图优化，CPU 上也能提速 1.5-3x；③是通往 ONNX Runtime/TensorRT/手机端等专用推理引擎的桥梁，这些引擎再叠加量化（INT8/FP16）才能获得 3-5x 以上的压缩加速。
</details>

---

## 12.8 类比：模型压缩 = 把大行李箱装进小背包

想象你要去旅行：

| 行李箱概念 | 模型压缩对应物 |
|-----------|--------------|
| 大行李箱（28 寸） | FP32 模型（28 GB） |
| 小背包（10 升） | 显存有限的部署环境（10 GB） |
| **量化**：把衣服真空压缩 | 用 INT8/INT4 存权重（体积小，衣服还是那几件） |
| **剪枝**：扔掉不用的纪念品 | 删掉接近 0 的权重（内容少了，但核心还在） |
| **蒸馏**：不带"旅行攻略书"，只带精华笔记 | 大模型教小模型（学生学老师的精华，体积小能力近） |
| **ONNX**：换个更合理的打包方式（卷着塞） | 格式转换 + 图优化（体积没变，但更紧凑、更易携带） |

**关键认知**：
- 量化像真空压缩——**东西（参数）一件不少，只是占的地方小了**，代价是取用时可能稍有"褶皱"（精度损失）
- 剪枝像断舍离——**真的扔掉了一些东西**，但扔的是不重要的
- 蒸馏像带精华笔记——**换了个更小的载体**，前提是笔记记得好
- 四种方法可以组合：先剪枝/蒸馏换小模型 → 再量化 → 最后转 ONNX/TensorRT 部署

**选型口诀**：本地/手机用 GGUF，GPU 服务用 AWQ/GPTQ，跨平台用 ONNX，NVIDIA 极致性能上 TensorRT。先量化，不够再剪枝/蒸馏，永远先跑评测再上线。

---

**上一章**：[第11章 数据工程](../11-数据工程/README.md) | **返回目录**：[教程首页](../README.md)
