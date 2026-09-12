---
title: 第16章 边缘部署
tags: ['ai-infra', '教程', '边缘', 'llama.cpp']
章节: 16
状态: 完成
---

# 第16章：边缘部署

> 边缘部署 = 把大餐做成便当。大模型是满汉全席，边缘设备只能装下便当。

## 16.1 问题：模型太大，怎么跑在手机/边缘设备上？

你想在手机上跑大模型，但：
- 手机只有 8GB 内存，模型要 16GB
- 手机算力弱，推理要 10 秒
- 不能联网，必须本地运行

**核心矛盾**：大模型能力 vs 边缘设备资源。

## 16.2 方案：四种边缘部署方式

| 方式 | 一句话 | 适合场景 |
|------|--------|---------|
| **llama.cpp** | C++ 推理，极致优化 | 手机、树莓派、Mac |
| **MLC LLM** | 编译到目标平台 | 手机 App 集成 |
| **ONNX Runtime** | 跨平台推理 | 模型格式统一 |
| **TFLite** | Google 移动端 | Android 深度集成 |

## 16.3 项目

| 项目 | ⭐ | 定位 | 昇腾支持 |
|------|-----|------|---------|
| **llama.cpp** | 125k | C++ 端侧推理 | ✅ CANN 后端已合入 |
| **MLC LLM** | 20k | 编译部署 | ⚠️ 探索中 |
| **ONNX Runtime** | 18k | 跨平台推理 | ⚠️ 通过 CANN EP |
| **MediaPipe** | 30k | Google 边缘 AI | ❌ 无昇腾支持 |
| **昇腾 310P** | 华为 | 昇腾边缘推理卡 | ✅ 原生支持 |

## 16.4 代码示例

### llama.cpp 在 Mac 上运行

```bash
# 1. 克隆
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. 编译
cmake -B build
cmake --build build --config Release

# 3. 下载模型（Qwen 0.5B GGUF 格式）
# 从 HuggingFace 下载或转换

# 4. 运行
./build/bin/llama-cli -m models/qwen2-0.5b-instruct-q4_0.gguf -p "你好" -n 50
```

### ONNX 导出

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 导出 ONNX
dummy_input = tokenizer("你好", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input['input_ids'],),
    "qwen2-0.5b.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}}
)
```

## 16.5 排障：边缘部署常见问题

| 问题 | 现象 | 根因 | 解法 |
|------|------|------|------|
| 模型太大 | 加载失败/崩溃 | 内存不足 | 量化（INT4/INT8） |
| 速度太慢 | 推理 10 秒 | CPU 算力弱 | 换更小模型、用 GPU 加速 |
| 精度损失 | 输出质量差 | 量化损失 | 用 QAT（量化感知训练） |
| 格式不兼容 | 加载报错 | 框架不支持 | 转 ONNX/GGUF 格式 |

## 16.6 动手实验：用 llama.cpp 在 Mac 上跑一个模型

**目标**：在 Mac 上本地运行大模型。

**步骤**：

```bash
# 1. 安装依赖
brew install cmake

# 2. 克隆编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release

# 3. 下载模型（示例用 TinyLlama）
curl -L -o models/tinyllama-1.1b-q4_0.gguf \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# 4. 运行
./build/bin/llama-cli -m models/tinyllama-1.1b-q4_0.gguf -p "你好，请介绍一下自己" -n 100
```

**预期输出**：
- 模型加载成功
- 逐字生成回答

**常见错误**：
- 编译失败 → 检查 CMake 版本、Xcode 命令行工具
- 模型下载失败 → 用镜像站或手动下载

## 16.7 自测题

1. llama.cpp 和 PyTorch 的区别是什么？（提示：C++ 推理 vs Python 训练）

<details>
<summary>参考答案</summary>

llama.cpp 是纯 C++ 推理引擎，无 Python 依赖，适合边缘部署；PyTorch 是 Python 框架，适合训练和开发。llama.cpp 用 GGUF 格式，PyTorch 用原生格式或 ONNX。
</details>

2. 为什么边缘部署需要量化？（提示：内存和算力限制）

<details>
<summary>参考答案</summary>

边缘设备内存小（手机 8GB）、算力弱（CPU/GPU 不如服务器）。量化把 FP32 变成 INT8/INT4，模型体积缩小 4-8 倍，推理速度提升 2-4 倍，但会损失一定精度。
</details>

3. MLC LLM 和 llama.cpp 的区别是什么？（提示：编译 vs 解释执行）

<details>
<summary>参考答案</summary>

MLC LLM 把模型编译成目标平台原生代码（如 iOS/Android），性能最优但编译复杂；llama.cpp 是解释执行，跨平台兼容性好但性能略低。MLC 适合深度集成到 App，llama.cpp 适合快速原型。
</details>

---

**上一章**：[第15章：MLOps](../15-MLOps/README.md) | **返回目录**：[教程首页](../README.md)
