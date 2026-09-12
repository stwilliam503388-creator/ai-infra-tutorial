---
title: 第08章 实战路径
tags: ['ai-infra', '教程', '实战', '动手']
章节: 08
状态: 完成
---

# 第8章：实战路径

> 4 周从"能调 API"到"能部署/优化/排障"

## 8.0 总览

| 周 | 主题 | 核心目标 | 参考结构 |
|----|------|---------|---------|
| 1 | 跑通第一个模型 | 本地运行 LLM | openai-cookbook Notebook 风格 |
| 2 | 部署到服务器 | 模型变成 API | LLM-Engineers-Handbook `pipelines/` |
| 3 | 多卡并行 | 分布式训练 | ml-engineering `training/model-parallelism/` |
| 4 | 监控排障 | 生产环境运维 | ml-engineering `debug/` + `orchestration/` |

**学习原则**：
- 每周一个主题，不要跳周
- 每个代码示例必须亲手运行
- 遇到错误先查第 9 章避坑指南

## 8.1 Week 1：跑通第一个模型

### 目标
在本地电脑上跑通一个大模型，输入文字，输出回答。

### 前置检查

```bash
# 检查 Python 版本（需要 3.9+）
python --version

# 检查是否有昇腾 NPU（npu-smi 是昇腾的"仪表盘"，装好 CANN 即自带）
npu-smi info  # 有 NPU 会显示设备信息，没有会报错
```

### 动手任务

**任务 1：环境准备**
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖（torch_npu 是昇腾 NPU 的 PyTorch 适配层，版本须与 torch、CANN 配套）
pip install torch torch_npu transformers accelerate
```

**任务 2：加载模型**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu  # 关键：import 一次即注册 npu 后端

# 检查 NPU 是否可用（没有昇腾卡时退回 CPU）
device = "npu:0" if torch.npu.is_available() else "cpu"
print(f"使用设备：{device}")

# 加载小模型（Qwen 0.5B，约 1GB）
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# 推理
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**任务 3：理解输出**
- 模型回答了什么？
- 回答质量如何？
- 推理花了多长时间？（用 `time` 命令或 `%%time` 计时）

**任务 4：尝试不同参数**
```python
# 温度越高越随机，越低越确定
outputs = model.generate(**inputs, max_length=100, temperature=0.1)  # 保守
outputs = model.generate(**inputs, max_length=100, temperature=1.5)  # 随机

# top_p 控制词汇范围
outputs = model.generate(**inputs, max_length=100, top_p=0.9)  # 只考虑概率前 90% 的词
```

### 验证标准
- [ ] 能成功加载模型（不报错）
- [ ] 能输入文字并获取回答
- [ ] 理解 `tokenizer` 和 `model` 的分工
- [ ] 能调整 `temperature` 和 `top_p` 并观察输出变化

### 常见问题

| 问题 | 原因 | 解法 |
|------|------|------|
| 下载模型慢 | HuggingFace 在国外 | 用镜像 `export HF_ENDPOINT=https://hf-mirror.com` |
| NPU 显存不够 | 模型太大 | 换更小的模型，如 `Qwen/Qwen2-0.5B` |
| `import torch_npu` 报错 | torch / CANN / torch_npu 版本不配套 | 按 CANN 官方版本配套表安装（如 torch 2.1.0 + torch_npu 2.1.0.post6），`pip install torch_npu` 后查 `npu-smi info` 确认 CANN 正常 |
| 没有昇腾 NPU | 电脑没有昇腾卡 | 用 CPU 模式，速度会慢 10 倍；或租云上昇腾实例（ModelArts/华为云） |

### 参考
- openai-cookbook `examples/` 结构
- HuggingFace transformers 文档

---

## 8.2 Week 2：部署到服务器

### 目标
把模型部署成 HTTP API，让其他人也能调用。

### 动手任务

**任务 1：安装 vLLM-Ascend**
```bash
# 昇腾 NPU 上部署推理服务用 vLLM-Ascend（vLLM 的昇腾适配版）
pip install vllm-ascend

# 声明目标设备为昇腾（每次启动前都要 export）
export VLLM_TARGET_DEVICE=ascend
```

**任务 2：启动 API 服务**
```python
from vllm import LLM, SamplingParams

# 加载模型（vllm-ascend 会自动调度到 NPU）
llm = LLM(model="Qwen/Qwen2-0.5B-Instruct")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# 推理
outputs = llm.generate(["你好，请介绍一下自己"], sampling_params)
print(outputs[0].outputs[0].text)
```

**任务 3：启动 OpenAI 兼容 API**
```bash
export VLLM_TARGET_DEVICE=ascend
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-0.5B-Instruct \
    --port 8000 \
    --host 0.0.0.0
```

**任务 4：用 curl 测试**
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "prompt": "你好",
        "max_tokens": 50
    }'
```

**任务 5：用 Python 调用**
```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"  # vLLM-Ascend 不需要真实 key

response = openai.Completion.create(
    model="Qwen/Qwen2-0.5B-Instruct",
    prompt="你好",
    max_tokens=50
)
print(response.choices[0].text)
```

### 验证标准
- [ ] 能启动 vLLM-Ascend API 服务
- [ ] 能用 curl 或 Python 调用 API
- [ ] 理解 `temperature`/`top_p`/`max_tokens` 参数
- [ ] 能看懂 API 返回的 JSON 结构

### 参考
- LLM-Engineers-Handbook `pipelines/` 结构
- [vLLM-Ascend 官方文档](https://github.com/vllm-project/vllm-ascend)

---

## 8.3 Week 3：多卡并行

### 目标
用 2 张昇腾 NPU 训练一个模型，理解数据并行原理。

### 动手任务

**任务 1：理解并行策略**
- **数据并行**：每张卡一份完整模型，数据分批
- **张量并行**：模型太大，拆开放到多张卡
- **流水线并行**：模型分层，每层一张卡

**任务 2：DeepSpeed 配置**
```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.0001
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

**任务 3：启动训练**
```bash
# 昇腾环境：用 ASCEND_RT_VISIBLE_DEVICES 指定可见卡（等价于 CUDA_VISIBLE_DEVICES）
export ASCEND_RT_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 train.py --deepspeed_config ds_config.json
# 多卡通信走 HCCL（昇腾的集合通信库，等价于 NCCL），由 torch_npu 自动接管
```

**任务 4：观察多卡利用率**
```bash
# 另一个终端运行
npu-smi info -t usages   # 每秒刷新：watch -n 1 npu-smi info -t usages
```

### 验证标准
- [ ] 理解三种并行策略的区别
- [ ] 能写 DeepSpeed 配置文件
- [ ] 能启动多卡训练（通信走 HCCL）
- [ ] 能用 npu-smi 确认多卡都在工作

### 参考
- ml-engineering `training/model-parallelism/`
- DeepSpeed 官方文档

---

## 8.4 Week 4：监控排障

### 目标
监控 NPU 利用率，能看懂监控图表，会排查常见问题。

### 动手任务

**任务 1：NPU 监控**
```bash
# 实时查看 NPU 状态（每秒刷新）
watch -n 1 npu-smi info -t usages

# 单次查看完整信息（型号/温度/功耗/显存）
npu-smi info

# 深度性能分析用 msprof（CANN 自带，替代 nvidia-smi 只能看占用、不能看算子的短板）
msprof --application="python train.py" --output=./prof_result
# 结果会生成 AI Core 利用率、算子耗时排名等 JSON/CSV 报告
```

**任务 2：Prometheus + Grafana**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'npu'
    static_configs:
      - targets: ['localhost:9100']  # ascend-exporter（昇腾官方指标导出器）
```

**任务 3：常见排障**
- OOM → 减小 batch size / 换小模型 / 量化
- 速度慢 → 用 msprof 看哪个算子耗时最长；换 vLLM-Ascend / 开 Continuous Batching
- 多卡不工作 → 检查 HCCL（`HCCL_CONNECT_TIMEOUT`）/ 网络
- 单卡利用率低 → 数据加载瓶颈，用 `msprof` 分析 Host/Device 时间分布

**任务 4：写排障日记**
记录你遇到的错误、排查过程、最终解法。这是你最宝贵的学习资料。

### 验证标准
- [ ] 能看懂 npu-smi 输出
- [ ] 能配置 Prometheus 监控（ascend-exporter）
- [ ] 能排查常见错误
- [ ] 有至少一篇排障日记

### 参考
- ml-engineering `debug/`
- ml-engineering `orchestration/`

---

## 8.5 端到端项目结构（参考 LLM-Engineers-Handbook）

```
my-llm-project/
├── steps/
│   ├── etl/              # Week 1: 数据准备
│   │   ├── download.py   # 下载数据集
│   │   ├── clean.py      # 数据清洗
│   │   └── split.py      # 训练/验证集划分
│   ├── training/         # Week 2: 模型训练
│   │   ├── config.py     # 训练配置
│   │   ├── model.py      # 模型定义
│   │   └── train.py      # 训练脚本
│   ├── evaluating/       # Week 3: 模型评估
│   │   ├── metrics.py    # 评估指标
│   │   └── evaluate.py   # 评估脚本
│   └── export/           # Week 4: 导出部署
│       ├── quantize.py   # 模型量化
│       └── deploy.py     # 部署脚本
├── pipelines/
│   └── end_to_end.py     # 端到端流水线
├── configs/              # 配置文件
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                 # 数据目录
├── checkpoints/          # 模型保存
├── logs/                 # 日志
└── README.md
```

## 8.6 动手实验

> 4 周 4 个实验，每个实验都有明确产出。先看第 9 章避坑指南再动手。

### 实验 1（Week 1）：跑通 Qwen 0.5B 昇腾 NPU 本地推理 ⭐

**目标**：在昇腾 NPU 上加载 `Qwen2-0.5B-Instruct` 并完成一次对话推理，理解 tokenizer + model 的分工 + torch_npu 适配层。

**步骤**

```bash
# 1. 准备环境（已有 .venv 则跳过）
python -m venv .venv && source .venv/bin/activate
pip install torch torch_npu transformers accelerate

# 2. 国内网络先挂镜像，否则下载很慢
export HF_ENDPOINT=https://hf-mirror.com
```

```python
# 3. 保存为 qwen_npu.py 并运行
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time
import torch_npu  # 注册昇腾 NPU 后端

device = "npu:0" if torch.npu.is_available() else "cpu"
print(f"使用设备：{device}")

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt").to(device)
t0 = time.time()
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
print(f"推理耗时：{time.time()-t0:.1f}s")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```bash
python qwen_npu.py
```

**预期输出**

```
使用设备：npu:0
推理耗时：3.2s
你好！我是 Qwen，一个由阿里云开发的开源人工智能助手……
```

**常见错误**

| 错误 | 原因 | 解法 |
|------|------|------|
| `Connection error` / 下载卡在 0% | HuggingFace 无法访问 | `export HF_ENDPOINT=https://hf-mirror.com` 后重跑 |
| `NPU out of memory` | 显存不足（0.5B 也需约 2GB） | 加 `model.half()`，或换 CPU 模式 |
| `import torch_npu` 失败 / 找不到 CANN | torch、CANN、torch_npu 版本不配套 | 按 CANN 官方版本配套表重装，确认 `npu-smi info` 正常 |
| `FutureWarning` 刷屏 | 版本差异 | 忽略，不影响结果；或 `pip install -U transformers` |

**产出**：一份可复现的昇腾 NPU 本地推理脚本 + 一次真实对话输出。

### 实验 2（Week 2）：vLLM-Ascend 部署 API 并用 curl 调用 ⭐

**目标**：用 vLLM-Ascend 在昇腾 NPU 上启动 OpenAI 兼容 API，用 curl 和 Python 各调一次。

**步骤**

```bash
# 1. 安装并启动服务（首次会下载模型，耐心等）
pip install vllm-ascend
export VLLM_TARGET_DEVICE=ascend
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-0.5B-Instruct \
    --host 0.0.0.0 --port 8000
```

```bash
# 2. 新开终端，等服务就绪后调用（模型加载完会打印 "Starting vLLM API server"）
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 50
    }'
```

**预期输出**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
}
```

**常见错误**

| 错误 | 原因 | 解法 |
|------|------|------|
| `Connection refused` | 服务还没起好或端口占用 | 等服务打印 `Application startup complete`；`lsof -i :8000` 查占用 |
| `model not found` | 模型名与启动时不一致 | curl 的 `"model"` 字段必须与 `--model` 完全一致 |
| 启动即报 `NPU error / RuntimeError` | vLLM-Ascend 与 CANN/torch 版本不配套，或忘设设备 | `export VLLM_TARGET_DEVICE=ascend` 后重启；`pip install -U vllm-ascend`；无 NPU 机器加 `--device cpu`（慢） |

**产出**：一个跑在 8000 端口的 LLM API，任何语言都能用 HTTP 调它。

### 实验 3（Week 3）：DeepSpeed 昇腾多卡训练 ⭐⭐

**目标**：用 DeepSpeed 在 2 张昇腾 NPU 上跑通一次数据并行训练，用 npu-smi 确认两张卡都在工作（通信走 HCCL）。

**步骤**

```bash
# 1. 准备最小训练脚本 train.py
pip install deepspeed
```

```python
# train.py —— 2 卡数据并行最小示例
import torch, deepspeed, torch.nn as nn
import torch_npu  # 注册 npu 后端，HCCL 通信随之接管

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.ln = nn.Linear(64, 1000)

    def forward(self, x):
        return self.ln(self.emb(x))

model = TinyLM()
loss_fn = nn.CrossEntropyLoss()
data = torch.randint(0, 1000, (64, 16))  # 64 样本会被平分到 2 卡

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)
for step in range(10):
    logits = model_engine(data)
    loss = loss_fn(logits.view(-1, 1000), data.view(-1))
    model_engine.backward(loss)
    model_engine.step()
    if step % 5 == 0:
        print(f"step {step} | loss {loss.item():.4f}")
```

```json
// ds_config.json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {"type": "Adam", "params": {"lr": 0.0001}},
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 2}
}
```

```bash
# 2. 启动 2 卡训练（指定可见的 2 张 NPU）
export ASCEND_RT_VISIBLE_DEVICES=0,1
deepspeed --num_gpus=2 train.py

# 3. 新开终端，确认两张卡都被占用（每张卡出现一个 python 进程）
npu-smi info
```

**预期输出**

```
step 0 | loss 6.9087
step 5 | loss 6.9021
# npu-smi 中：
# 2 个 python 进程各占 1 张 NPU，util 交替跳动
```

**常见错误**

| 错误 | 原因 | 解法 |
|------|------|------|
| `AssertionError: Zero-2 is not supported`（仅一台机器时） | ZeRO-2 默认要求多节点 | 单机多卡加 `"zero_optimization": {"stage": 2, "reduce_bucket_size": "auto"}`，或换 stage 1 |
| 启动报 `HCCL error / init failed` | 卡间通信初始化失败 | 先单卡 `deepspeed --num_gpus=1 train.py` 验证脚本本身没问题；检查 `ASCEND_RT_VISIBLE_DEVICES` 是否合法 |
| 显存爆掉 | 模型 × 卡数 × 优化器状态 | 调小 `train_batch_size` 或关掉 fp16 |
| npu-smi 只有一张卡有进程 | `--num_gpus=2` 没生效或第二张卡被占 | `npu-smi info` 确认卡空闲，删掉 `--num_gpus` 让 DeepSpeed 自动探测 |

**产出**：多卡训练日志 + npu-smi 中"每卡一个进程"的直接证据。

### 实验 4（Week 4）：Prometheus 监控昇腾 NPU ⭐⭐

**目标**：用 ascend-exporter + Prometheus 采集 NPU 指标，用 HTTP 查询验证数据链路。

**步骤**

```bash
# 1. 安装并启动 ascend-exporter（昇腾官方指标导出器，默认 9100 端口）
# 参见 https://gitee.com/ascend/ascend-exporter 按对应 CANN 版本部署
# 或用容器：docker run -d --privileged -p 9100:9100 ascendexporter/ascend-exporter

# 2. 写 Prometheus 配置
```

```yaml
# prometheus.yml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'npu'
    static_configs:
      - targets: ['localhost:9100']
```

```bash
# 3. 启动 Prometheus（macOS 用 brew install prometheus，Linux 直接下二进制）
prometheus --config.file=prometheus.yml

# 4. 验证采集：Prometheus 自身指标
curl 'http://localhost:9090/api/v1/targets' | head -20
```

**预期输出**

```json
{"status":"success","data":{"activeTargets":[{"labels":{"job":"npu"},"health":"up","scrapeUrl":"http://localhost:9100/metrics"}]}}
```

```bash
# 5. 查询 NPU 利用率指标（指标名见 ascend-exporter 文档，这里查算力利用率）
curl 'http://localhost:9090/api/v1/query?query=ascend_npu_ai_core_utilization'
```

**预期输出**

```json
{"status":"success","data":{"resultType":"vector","result":[{"metric":{"npu":"0"},"value":[1720000000,"35"]}]}}
```

> 结果里的 `"35"` 表示 NPU 0 当前 AI Core 利用率 35%。浏览器打开 `http://localhost:9090` 还能用 Graph 页画曲线。

**常见错误**

| 错误 | 原因 | 解法 |
|------|------|------|
| 查询返回空数组 | job 名或端口不对 | 先 `curl localhost:9100/metrics` 确认 exporter 有数据 |
| `health: "down"` | Prometheus 抓不到 exporter | 确认 exporter 进程在跑、防火墙没挡 9100 |
| 指标名找不到 | exporter 版本差异 | `curl localhost:9100/metrics` 看实际指标名，把查询里的名字换成现成的 |
| 本地没有昇腾 NPU | 无卡可采 | 用远程昇腾机器的 9100 端口，targets 改成 `'<机器IP>:9100'` |

**产出**：一条"NPU → exporter → Prometheus → 查询"全链路，浏览器里能看到实时利用率曲线。

---

## 8.7 自测题

1. Week 1 的核心目标是什么？（提示：本地跑通模型）

<details>
<summary>参考答案</summary>

在本地电脑上加载一个大模型，输入文字，输出回答。核心是让模型"跑起来"，验证环境配置正确。
</details>

2. vLLM-Ascend 的 OpenAI 兼容 API 怎么启动？（提示：`VLLM_TARGET_DEVICE=ascend` + `python -m vllm.entrypoints.openai.api_server`）

<details>
<summary>参考答案</summary>

```bash
export VLLM_TARGET_DEVICE=ascend
python -m vllm.entrypoints.openai.api_server \
    --model <模型名> \
    --port 8000
```
然后可以用标准 OpenAI SDK 或 curl 调用。
</details>

3. 数据并行和张量并行的区别？（提示：数据分批 vs 模型拆分）

<details>
<summary>参考答案</summary>

数据并行：每张卡一份完整模型，数据分成多份，各算各的，最后合并梯度。张量并行：模型太大放不下，把模型的某一层拆开到多张卡上，需要卡间通信。
</details>

---

**上一章**：[第7章：应用层](../07-应用层/README.md) | **下一章**：[第9章：避坑指南](../09-避坑指南/README.md)
