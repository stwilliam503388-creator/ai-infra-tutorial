---
tags:
  - AI-Infra
  - 推理层
  - 教程
  - vLLM
---

# 05 推理层：1000 个用户同时问，怎么不排队？

> **本章问题**：1000 个用户同时发消息，为什么有的服务卡成幻灯片，有的服务却流畅如丝？
> **本章答案**：靠「厨房出菜系统」一样的推理引擎——把 NPU/GPU 当成一口锅，把请求当成一道道菜，高手不炒完一桌再开下一桌，而是**同时炒好几个人的菜**。
>
> 前置：第 04 章（框架层）、第 03 章（算子层）。读完本章你将掌握：推理引擎的三大核心技术（Continuous Batching / PagedAttention / RadixAttention）、四个主流开源项目怎么选、用 vLLM-Ascend 十分钟起一个 OpenAI 兼容的 API 服务。

---

## 0. 先记住这个类比：推理引擎 = 厨房出菜系统

把大模型推理服务想象成一家餐厅的后厨：

| 餐厅后厨 | 推理引擎 | 说明 |
|---------|---------|------|
| 厨师（锅） | NPU/GPU | 最贵、最稀缺的资源，一刻都不能闲 |
| 一张桌子 | 一个请求（Prompt） | 占着资源直到上齐菜 |
| 前菜 → 主菜 | Prefill（预填充） | 一次性把整桌客人的点单读进脑子，算得快 |
| 一道道上菜 | Decode（逐字生成） | 一次只能吐一个 token，慢，但必须串行 |
| 后厨的灶台 | KV Cache（键值缓存） | 每桌客人的"半成品"都要放灶台上，出锅前不能扔 |
| 翻台率 | 吞吐量（Throughput） | 单位时间能服务多少桌客人 |
| 出第一道菜的速度 | 首 token 延迟（TTFT） | 客人等第一口菜的时间 |
| 灶台空间不够 | OOM（显存溢出） | 半成品放不下了，直接砸锅 |

**推理引擎要解决的核心矛盾**：NPU/GPU 那么贵，不能让它在等某一道菜的间隙闲着；但每一桌客人的"半成品"（KV Cache）又都占着灶台，放得越多，能同时服务的客人越少。

接下来的三个方案，全是围绕"**如何让一桌菜还没上完就开下一桌**"做文章。

---

## 1. 问题：1000 个用户同时问，怎么不排队？

### 1.1 朴素做法：一个一个来（串行）

最朴素的推理服务：来一个请求，GPU 全算完，吐完结果，再服务下一个。

```
用户1 ████████████████████░░░░░░░░░░░░░░░░
用户2                                   ████████████████████░░░░░░░░
用户3                                                        ██████████
        GPU 利用率 20%…… 排队 999 人……
```

- **TTFT（首 token 延迟）**：第 1000 个用户要等前面 999 个全部跑完 → 分钟级起步。
- **GPU 利用率**：Decode 阶段（逐字吐）只用到 GPU 计算能力的一小部分，大部分算力在空转。

> 就像餐厅只有一个灶台，一次只接待一桌，上完一桌菜才接下一桌——客人全堵在门口。

### 1.2 为什么"并排炒"很难？

直觉方案是"多开几个模型副本"（多副本 + 负载均衡）。但 GPU 显存是硬约束：

- 一个 7B 模型的权重就要 ~14GB（FP16），13B 要 ~26GB。
- 更大的瓶颈是 **KV Cache**：每个并发请求都要在显存里保存自己的"半成品"，一个长对话可能吃掉几 GB。
- 显存里能塞下 10 个副本，就最多服务 10 个并发——1000 个用户照样排队。

所以真正的问题是：**如何在一个模型副本里，同时处理成百上千个请求，并且不浪费任何显存和算力。**

### 1.3 三个关键技术，逐个击破

| 问题 | 技术 | 解决什么 |
|------|------|---------|
| GPU 算力空转 | **Continuous Batching**（连续批处理） | 让"菜没上完的桌"和"刚点单的桌"共用一个灶台 |
| 显存碎片化、装不下更多人 | **PagedAttention**（分页注意力） | 让"半成品"像虚拟内存一样按页管理，几乎零浪费 |
| 相同前缀重复计算 | **RadixAttention**（前缀树缓存） | 多轮对话/多用户问同一段上下文时，直接复用"半成品" |

---

## 2. 方案一：Continuous Batching（连续批处理）

### 2.1 传统 Batching 的问题

传统做法是**静态批处理**：攒满 N 个请求，一起算，等这批全部算完，再开下一批。

```
批1: 用户A ██████████░░░░░░░░   ← A 的菜上完了，但整批还在等 B、C
批1: 用户B ██████████████████   ← 最慢的决定了整批的速度
批1: 用户C ██████████░░░░░░░░░░
批2: 用户D ██████████░░░░░░░░
```

- 慢的用户拖慢所有人（队头阻塞）。
- A 的菜上完了，算力空转等 B、C。
- 新到的请求（用户 D）必须干等下一批。

### 2.2 连续批处理：随时来，随时上桌

**Continuous Batching**（[论文：Orca, 2022](https://www.usenix.org/conference/osdi22/presentation/yu)）的做法：**每一轮迭代（一个 token 的生成）结束时，都重新组批**。

```
t=0  批: [A prefill, B prefill]          ← 新请求立即进批
t=1  批: [A decode, B decode, C prefill] ← C 来了，这轮就带上
t=2  批: [A decode, C decode, D prefill] ← A 快好了，B 已走（完成）
t=3  批: [C decode, D decode, E prefill] ← 完成一个，补一个，锅永远不空
```

效果：
- **请求完成即退出批次**，新请求**即刻加入**，GPU 永不空转。
- 吞吐量提升 **10~20 倍**（Orca 论文实测，对比静态 batching）。
- 实现代价：动态拼接不同长度的序列，需要专门的管理器（Scheduler）——这就是 vLLM 等引擎的"调度器"组件。

> **类比**：传统 batching 是一桌菜上齐才翻台；Continuous Batching 是"谁吃完谁走，空位立刻补新人"。灶台利用率拉满。

---

## 3. 方案二：PagedAttention（分页注意力）

### 3.1 问题：KV Cache 的浪费

Decode 阶段每生成一个 token，都要把该 token 的 K/V 向量存进显存（KV Cache），供后续 token 做注意力计算。

朴素实现按**最大长度**给每个请求**预分配一整块连续显存**：

```
┌────────────────────────── 一整块连续显存 ──────────────────────────┐
│ 请求1: [已用 30%] [预留 70% 的空闲]                                  │
│ 请求2: [已用 80%] [预留 20% 的空闲]                                  │
│ 请求3: [已用 10%] [预留 90% 的空闲]                                  │
└────────────────────────────────────────────────────────────────────┘
```

- **内部碎片**：预留空间没用上（请求实际只有 100 token，却预留了 2048）。
- **外部碎片**：块与块之间缝隙凑不成一个整块，新请求放不进去。
- 实测中 KV Cache 的浪费率高达 **60%~80%**——显存被浪费，能并发的请求数就少。

### 3.2 分页：像操作系统管理内存一样管理 KV Cache

**PagedAttention**（[论文：vLLM, 2022](https://arxiv.org/abs/2309.06180)）借鉴操作系统的**虚拟内存分页**思想：

- 把 KV Cache 切成固定大小的**块（Block，如 16 个 token 一块）**。
- 每个请求只分配**用得到的块**，用完了再按需申请新块。
- 物理上不连续的块，逻辑上通过**块表（Block Table）**串成完整序列。

```
请求1: 块A → 块B → 块E        ← 逻辑连续，物理上散布
请求2: 块C → 块D → 块F
        ┌─────┬─────┬─────┬─────┐
物理内存 │ 块A │ 块B │ 块C │ 块D │
        ├─────┼─────┼─────┼─────┤
        │ 块E │ 块F │ 空  │ 空  │   ← 空闲块可以随时给任何请求
        └─────┴─────┴─────┴─────┘
```

效果：
- KV Cache 内存浪费从 **60%~80% 降到 4% 以下**。
- 相同显存能容纳 **2~4 倍**的并发请求。
- 额外红利：**块级共享**——多个请求可以共享同一个物理块（见下一个方案）。

> **类比**：以前每个客人独占一整层楼（哪怕只住一间房）；现在变成快捷酒店按房间分配，谁退房房间立刻给新人。

---

## 4. 方案三：RadixAttention（前缀树缓存）

### 4.1 问题：重复计算

真实流量里大量请求共享相同的前缀：

- **多轮对话**：第 3 轮请求 = 第 1、2 轮全文 + 新问题。前面的内容每次都重新算一遍 Prefill。
- **Agent / RAG 场景**：系统提示词（几万 token）每次都重新处理。
- **评测 / 批量任务**：1000 个请求共用同一个 prompt 模板。

按传统做法，每轮对话都要把整段历史重新 Prefill 一次——**极其浪费**。

### 4.2 前缀树：复用一切可复用的计算

**RadixAttention**（[SGLang 论文, 2024](https://arxiv.org/abs/2312.07104)）把所有请求的 KV Cache 组织成一棵**基数树（Radix Tree）**：

- 相同前缀的请求**共享同一段 KV Cache 物理块**（PagedAttention 的块共享机制支持）。
- 新请求先做**最长前缀匹配**，命中的部分直接复用，只计算增量部分。
- 缓存空间不足时，按 **LRU** 策略淘汰最久没用的分支。

```
                    ┌─ "你是 AI 助手"（系统提示词，全部请求共享）
                    │
 请求1: "…帮我写周报" ├─ 共享节点 ─ "帮我写周报" ─ "下周的计划…"
                    │
 请求2: "…帮我写周报" ├─ 共享节点 ─ "帮我写周报" ─ "并翻译成英文"
                    │
 请求3: "…帮我写周报" └─ 共享节点 ─ "帮我写总结"
                    ↑
          请求 4 来了："…帮我写周报 3 点前给我"
          命中共享节点，只算 "3 点前给我" 这一段
```

效果（SGLang 论文实测，对话场景）：
- 对话式应用的吞吐提升 **3.6 倍**、TTFT 降低 **6.4 倍**（相比 vLLM 无前缀缓存）。
- 长系统提示词的 RAG / Agent 场景收益更大。

> **类比**：后厨把"高汤、底料"提前熬好放冷柜，任何菜都能直接舀一勺，不用每次重新熬。

### 4.3 小结：三件套的分工

| 技术 | 管什么 | 收益 |
|------|--------|------|
| Continuous Batching | 时间维：什么时候算 | 吞吐 ↑10~20x |
| PagedAttention | 空间维：显存怎么放 | 显存利用率 ↑，并发 ↑2~4x |
| RadixAttention | 复用维：哪些不用重算 | 多轮/RAG 场景吞吐再 ↑3~4x |

---

## 5. 项目：四个主流推理引擎怎么选

| 项目 | ⭐ Stars | 定位 | 适合谁 | 后端/生态 |
|------|---------|------|--------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | ⭐90k | 生产级 LLM 推理服务，事实标准 | 大多数团队的首选；OpenAI 兼容 API 开箱即用 | CUDA / ROCm / **Ascend NPU**（vllm-ascend） |
| [**vLLM-Ascend**](https://github.com/vllm-project/vllm-ascend) | ⭐2.7k | vLLM 昇腾硬件插件 | **昇腾用户首选**；与 vLLM 主仓库同步更新 | CANN / torch_npu / AscendC |
| [SGLang](https://github.com/sgl-project/sglang) | ⭐32k | 追求极致性能 + 结构化生成（Agent/多模态） | 高并发对话、Agent 工作负载；想用 RadixAttention 全量收益 | CUDA / Ascend NPU（sglang 原生支持） |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | ⭐125k | 本地/边缘推理，CPU 也能跑，GGUF 格式 | 个人电脑、Mac、树莓派；量化部署 | 纯 C/C++，无 CUDA 依赖，Apple Silicon 优化极好 |
| [lmdeploy](https://github.com/InternLM/lmdeploy) | ⭐8k | 国产全流程工具箱（推理 + 量化 + 服务），TurboMind 内核 | 想用中文文档/国产生态；InternLM 系模型 | CUDA / Ascend NPU，PPL 量化工具链完善 |
| [MindIE](https://www.hiascend.com/software/mindie) | 闭源 | 华为官方推理引擎 | 企业级生产部署；华为云/昇腾一体机 | 昇腾原生，性能最优 |

### 5.1 选型建议

- **线上生产 API**（要吞吐、要稳定、要 OpenAI 兼容）：**vLLM** —— 生态最全、踩坑文档最多、国产卡（昇腾）支持最成熟。
- **Agent / 长上下文 / 结构化输出**：**SGLang** —— RadixAttention + 约束解码在这类负载上优势明显。
- **本地跑 / 没 GPU / 极致省显存**：**llama.cpp** —— GGUF 量化后 7B 模型 4GB 内存可跑。
- **要量化 + 部署一条龙、偏好中文社区**：**lmdeploy**。

> ⭐ 数据为 2026-08 参考值，GitHub 上持续增长，看实时数字请点链接。

### 5.2 一眼看懂：同一个模型，四种打开方式

```bash
# vLLM：一条命令起 OpenAI 兼容服务
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# SGLang：同样一条命令
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 8000

# llama.cpp：本地 CPU/GPU 混合，GGUF 格式
llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8000

# lmdeploy：TurboMind 引擎 + 自带量化
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct --server-port 8000
```

### 5.3 SGLang 深潜：架构、昇腾落地与版本配套 ⭐⭐

§4 讲了 RadixAttention 的**原理**，这一节讲 SGLang 作为引擎的**实现层**。

**5.3.1 架构两分法**

SGLang = **前端 DSL + 高性能运行时**：

- **前端**：radix 感知的生成程序（`sglang.function`），可以把多轮对话、工具调用、约束解码写成一个程序，运行时自动做前缀复用
- **运行时**：调度器 + KV Cache 管理 + 约束解码引擎，负责把 RadixAttention 的收益真正兑现

对比 vLLM 的"纯推理服务"定位：vLLM 给你 OpenAI 兼容 API；SGLang 在此之上多了一层**编程接口**，Agent / 结构化生成场景的收益主要来自这层。

**5.3.2 与 vLLM 对比**

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV 复用 | PagedAttention（块管理） | RadixAttention（前缀树复用） |
| PD 分离 | 支持（v1 架构） | 支持，与 Mooncake/MemFabric 配合成熟 |
| 结构化输出 | 支持（guided decoding） | 原生强项（regex / JSON schema / EBNF） |
| 投机解码 | EAGLE 等 | EAGLE + tree 验证，迭代更快 |
| 最适合的负载 | 通用 API 服务 | 多轮对话 / RAG / Agent 工具链（前缀共享率高） |

**5.3.3 昇腾落地：sgl-kernel-npu ⭐⭐**

SGLang 是除 vLLM 外昇腾支持最完整的社区推理引擎，官方维护 [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) 算子库，包含两块：

- **DeepEP-Ascend**：DeepSeek DeepEP 的昇腾实现，MoE 专家并行通信（dispatch/combine），支持 A3 全互联 HCCS 和 A2 节点内 HCCS + 跨节点 RDMA
- **推理 kernel 全集**：MLA（Paged KV Cache）、GQA、FLA（Gated Delta Rule）、融合 RMSNorm+RoPE、投机解码 tree kernel、Mamba causal conv1d 等

**算子开发双路径**（与第 2 章呼应）：

1. **Ascend C**：编译进 `libsgl_kernel_npu.so`，经 `TORCH_LIBRARY_FRAGMENT` 注册，SGLang 里用 `torch.ops.npu.<op_name>()` 调用
2. **Triton-Ascend**：Python kernel 直接 `from sgl_kernel_npu.xxx import ...` 调用，适合快速迭代

**版本配套（严格，不能混搭）**：

| 组件 | 版本 |
|------|------|
| CANN | 9.1.0（Toolkit + Kernels + NNAL） |
| torch_npu | 2.10.0.post6 |
| triton-ascend | 3.2.2 |
| Python | 3.12 |
| 硬件 | Atlas A2 / A3 / 950PR 系列 |

```bash
# 启动（昇腾）
python -m sglang.launch_server --model-path <模型> \
    --attention-backend ascend --host 0.0.0.0 --port 30000
```

PD 分离模式在昇腾上走 **MemFabric**（替代 Mooncake 传输引擎做 KV Cache 传输），需单独安装 `memfabric_hybrid` 和 `mf-adapter`。

**5.3.4 真实坑（来自官方文档）**

Gemma 系模型（含 Qwen3.5）在昇腾 950 上必须安装 **950 专用** sgl-kernel-npu 包（asset 名带 `950`）：950 走 ACLNN RMSNorm，910B/910C 走 torch_npu 原生算子，装错包会在第一个 Gemma forward 直接挂掉。验证方法：

```bash
python -c "import sgl_kernel_npu.norm.gemma_rmsnorm"
```

---

## 6. 动手：用 vLLM 起一个 API 服务

> 本节代码结构参考 [openai-cookbook](https://github.com/openai/openai-cookbook) 的 Notebook 惯例：**先装依赖 → 启动服务 → 用 OpenAI SDK 调用 → 流式输出 → 批量测试**。
> 硬件要求：一张 ≥16GB 显存的 GPU（7B 模型 FP16），或者改用量化模型/小模型（如 `Qwen/Qwen2.5-3B-Instruct`）。

### 6.1 安装

```bash
# 建议用 Python 3.10+ 独立虚拟环境
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install vllm openai   # vllm 会连带装好 torch（CUDA 版）
```

> ⚠️ macOS / 无 NVIDIA GPU：`pip install vllm` 装的是 CPU 版，性能仅够体验 API 形态。
> 昇腾 NPU 用户请改用：`pip install vllm-ascend`（参考 04-昇腾技术 章节）。

### 6.2 启动服务（终端 1）

```bash
# 首次运行会自动从 HuggingFace 下载模型权重（约 15GB，7B FP16）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9          # 90% 显存给 KV Cache，剩余留给权重和激活
```

看到 `INFO: Application startup complete.` 即就绪。验证：

```bash
curl http://localhost:8000/v1/models   # 应返回模型列表（OpenAI 兼容）
```

### 6.3 客户端调用（终端 2 / Notebook）

```python
# client.py —— 结构参考 openai-cookbook 的 chat 示例
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # 指向 vLLM，而非 OpenAI 官方
    api_key="EMPTY",                       # vLLM 默认不校验 key，占位即可
)

# 1. 基础对话
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache"}],
)
print(resp.choices[0].message.content)

# 2. 多轮对话（验证 RadixAttention 的前缀复用）
messages = [{"role": "user", "content": "我要学 AI Infra，请给我列出 5 个主题"}]
resp = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", messages=messages)
messages.append({"role": "assistant", "content": resp.choices[0].message.content})
messages.append({"role": "user", "content": "详细展开第 3 个"})
resp = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", messages=messages)
print(resp.choices[0].message.content)
```

### 6.4 流式输出（Streaming，聊天体验必备）

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一首关于 GPU 的四行诗"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # 逐 token 打印，体验 TTFT
```

> 注意流式响应的首包时间 = **TTFT**。它比完整响应快很多——这就是聊天产品"边想边出字"的原理。

### 6.5 模拟 1000 并发（验证本章主题）

```python
# bench.py —— 用并发请求观察：为什么不用排队
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def one(i: int):
    resp = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": f"1+1 等于几？回答第 {i} 号" }],
        max_tokens=32,
    )
    return i

async def main():
    results = await asyncio.gather(*[one(i) for i in range(100)])
    print(f"完成 {len(results)} 个请求，全部成功，无排队等待")

asyncio.run(main())
```

跑完你会看到：**100 个请求几乎同时开始、同时结束**——这就是 Continuous Batching + PagedAttention 的效果。GPU 日志里可以看到 batch size 在动态变化（`Running batch size: N`）。

> 想看真实的吞吐/延迟指标？vLLM 自带指标：
> ```bash
> curl http://localhost:8000/metrics | grep -E "llm_requests|llm_generation_tokens"
> ```
> 关键指标：`vllm:num_requests_running`（在跑的批大小）、`vllm:time_to_first_token_seconds`（TTFT）、`vllm:e2e_request_latency_seconds`（端到端延迟）。

---

## 7. 排障参考：ml-engineering

[awesome-ml-engineering](https://github.com/huggingface/awesome-ml-engineering)（作者 stas00，HuggingFace 研究员，机器学习系统调优圣经级仓库）是推理层排障的第一参考，重点看两个目录：

### 7.1 `inference/` 目录

- **`inference/`**（[链接](https://github.com/huggingface/awesome-ml-engineering/tree/main/inference)）——推理优化实战：
  - `README.md` — 推理概览：批处理、量化（GPTQ/AWQ/GGUF）、投机解码（Speculative Decoding）、Tensor Parallelism。
  - `inference-techniques/` — 逐项技术拆解。
  - 适合场景：**服务慢 / 吞吐低 / 想压榨性能**时按图索骥。

### 7.2 `debug/` 目录

- **`debug/`**（[链接](https://github.com/huggingface/awesome-ml-engineering/tree/main/debug)）——排障手册：
  - `debug-cuda-memory.md` — 显存不足（CUDA OOM）逐案分析。
  - `debug-gpu.md` — GPU 行为异常（功耗、温度、利用率）。
  - `debug-pytorch.md` — PyTorch 常见崩溃。
  - 适合场景：**OOM / 卡死 / 输出 NaN** 时先查这里。

### 7.3 推理层高频故障速查表

| 症状 | 可能原因 | 先查哪 |
|------|---------|--------|
| `CUDA out of memory` | `--gpu-memory-utilization` 太高 / `max-model-len` 太大 / 并发太多 | debug-cuda-memory.md；调低利用率或换小模型 |
| 首 token 极慢（TTFT 高） | 长 Prompt 每次全量 Prefill，没吃到前缀缓存 | 检查是否用了 SGLang；或开 `--enable-prefix-caching`（vLLM 1.x 默认开） |
| 吞吐上不去 | batch 太小 / 单请求 max_tokens 过大 | 看 `/metrics` 的 `num_requests_running`；调大并发 |
| 输出乱码/中断 | 模型与分词器不匹配 / `max_tokens` 设太小 | 换官方模型名；调大 max_tokens |
| 多 GPU 但只用了 1 张 | 没开张量并行 | vLLM 加 `--tensor-parallel-size 2`（2 卡） |

---

## 8. 动手实验 ⭐

> 三个实验递进：**起服务 → 比性能 → 纯命令行验证**。环境沿用 6.1 的虚拟环境（`pip install vllm openai transformers`），服务用 6.2 的方式启动。没 GPU 就把模型换成 `Qwen/Qwen2.5-3B-Instruct`（或 CPU 版 vLLM，性能只够体验形态，不影响看流程）。

### 实验 1：用 vLLM 启动本地 API 服务并调用

**目标**：亲手把第 5 章讲的 Continuous Batching / PagedAttention 变成一个跑在 8000 端口的 OpenAI 兼容服务，并用 Python 调通它。

**步骤**：

```bash
# 终端 1：启动服务（首次运行会自动下载模型权重）
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.9
```

```python
# 终端 2：client.py —— 用 OpenAI SDK 调用本地服务
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "用一句话解释什么是 KV Cache"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

```bash
python client.py
```

**预期输出**：

```text
KV Cache 是 Transformer 推理时缓存的历史 token 的键值向量，
用来避免每生成一个 token 都重新计算之前的注意力。
```

同时终端 1 的日志里能看到请求被记录（`INFO` 级别，含请求 id、prompt tokens、生成 tokens）。如果并发发多个请求，日志里会出现 `Running batch size: N`——这就是 Continuous Batching 在动态组批。

**常见错误**：

| 症状 | 原因 | 解法 |
|------|------|------|
| `Connection refused` / `Failed to connect` | 服务没起来或端口不对 | 确认终端 1 出现 `Application startup complete.`；`curl localhost:8000/v1/models` 验证 |
| `404 model not found` | `model=` 参数与启动时不一致 | 必须和 `vllm serve` 后面的模型名完全一致（含组织名） |
| `CUDA out of memory` | 7B FP16 权重 + KV Cache 超显存 | 换 `Qwen/Qwen2.5-3B-Instruct`，或把 `--gpu-memory-utilization` 降到 0.7 |
| 出字很慢 | macOS / 无 NVIDIA GPU 装的是 CPU 版 | 属正常，仅体验 API 形态；要性能需 GPU 或昇腾（`vllm-ascend`） |

---

### 实验 2：对比 vLLM 和原生 transformers 的推理速度

**目标**：量化「引擎」的价值——同一个模型、同一批请求，vLLM（连续批处理）比 transformers 朴素 `generate` 快多少。这也是选型时最常被问到的数据。

**步骤**：

```python
# baseline.py —— 原生 transformers：逐条串行生成
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
prompts = [f"用三句话介绍 {c}" for c in ["GPU", "CPU", "NPU", "TPU", "ASIC", "FPGA", "显存", "内存"]] * 3  # 24 条

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()

t0 = time.perf_counter()          # 计时从 generate 开始，排除权重加载
for p in prompts:
    inputs = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
t1 = time.perf_counter()

total_tokens = len(prompts) * 64
print(f"transformers: {t1-t0:.1f}s, 吞吐 {total_tokens/(t1-t0):.1f} tokens/s")
```

```python
# engine.py —— vLLM 离线接口：整批并发生成
import time
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
prompts = [f"用三句话介绍 {c}" for c in ["GPU", "CPU", "NPU", "TPU", "ASIC", "FPGA", "显存", "内存"]] * 3

llm = LLM(model=MODEL, gpu_memory_utilization=0.9)   # 引擎初始化（加载权重），不计时

t0 = time.perf_counter()
out = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0))
t1 = time.perf_counter()

total_tokens = sum(len(o.outputs[0].token_ids) for o in out)
print(f"vLLM: {t1-t0:.1f}s, 吞吐 {total_tokens/(t1-t0):.1f} tokens/s")
```

```bash
python baseline.py && python engine.py
```

**预期输出**（7B FP16、单卡 A100/H100 量级；小模型/小并发倍数会缩水，但方向一致）：

```text
transformers: 128.5s, 吞吐 12.0 tokens/s
vLLM:         8.2s,   吞吐 187.3 tokens/s
```

即 **vLLM 快 10~20 倍**——这就是 Orca 论文里 Continuous Batching 的收益：transformers 每条请求串行、GPU 在 decode 间隙空转；vLLM 把 24 条请求揉进一个动态批次，decode 的碎片算力被其他请求的 prefill 填满。请求越多、max_tokens 越长，差距越大。

**常见错误**：

| 症状 | 原因 | 解法 |
|------|------|------|
| `transformers` 报 `CUDA out of memory` | 7B 权重 + 推理缓存占满显存 | `torch_dtype=torch.float16` 已开仍不够就换 3B 模型 |
| 计时把权重加载算进去了 | `from_pretrained` 在 `t0` 之前没跑完 | `perf_counter()` 必须包住 `generate`/`generate` 调用，加载放外面 |
| vLLM 吞吐反而更慢 | 无 GPU（CPU 版）/ 并发太小 | CPU 版对比无意义；把 prompts 加到 50+ 条再看差距 |
| 两个脚本 tokens 数对不上 | 模型提前停了（EOS） | 用 `len(o.outputs[0].token_ids)` 按实际生成数计，别硬乘 64 |

---

### 实验 3：用 curl 测试 OpenAI 兼容 API

**目标**：不写一行 Python，用 `curl` 验证服务的三个事实：**活着**（/v1/models）、**能对话**（/v1/chat/completions）、**能流式**（SSE）。这是日常排查服务问题时最快的手段。

**步骤**：

```bash
# 1. 健康检查：服务在跑吗？
curl -s http://localhost:8000/v1/models

# 2. 普通对话：POST JSON 请求
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "说一句你好"}],
    "max_tokens": 64
  }'

# 3. 流式对话：观察 SSE 逐 token 到达（体验 TTFT）
curl -N -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "数到 5"}],
    "stream": true,
    "max_tokens": 32
  }'
```

**预期输出**：

```text
# 1. /v1/models —— JSON 数组，id 与启动的模型名一致
{"object":"list","data":[{"id":"Qwen/Qwen2.5-7B-Instruct","object":"model",...}]}

# 2. chat/completions —— 完整响应，回答在 choices[0].message.content
{"id":"chatcmpl-...","choices":[{"index":0,"message":{"role":"assistant",
"content":"你好！很高兴见到你，有什么可以帮你的吗？"},...}],...}

# 3. stream:true —— 一长串 data: 行，每行是一个 token 的增量
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"你"},...}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"好"},...}]}
data: [DONE]
```

第 3 步的 `data: [DONE]` 是流结束标记。观察第一个 `data:` 包到达前的时间——那就是 **TTFT**。

**常见错误**：

| 症状 | 原因 | 解法 |
|------|------|------|
| `400 Bad Request` | 忘了 `-H "Content-Type: application/json"` | curl 默认不认 JSON body，必须带这个头 |
| `404` | base URL 写错 | vLLM 的 OpenAI 兼容路由在 `/v1/...` 下，别漏 `/v1` |
| 流式输出全挤在一行 | 没加 `-N` | `-N`（`--no-buffer`）让 curl 边收边打印，不加会等全部收完 |
| Windows PowerShell 下 JSON 报错 | 双引号被吞 | 用 `cmd` 跑，或把 JSON 写成文件：`curl -d @body.json` |

---

## 9. 自测题

<details>
<summary><b>题 1</b>：餐厅后厨用了「Continuous Batching」后，一桌客人吃完前菜还在等主菜，此时门口来了新客人。后厨应该怎么做？为什么？（简答）</summary>

**答**：应该让新客人立刻上桌开炒（新请求立即加入当前批次做 Prefill），而不是等当前批全部上完菜。因为 Decode 阶段 GPU 利用率低，插入新请求的 Prefill 正好填补空闲算力；同时老客人吃完就走，空出的灶位立即补新人，锅（GPU）永不空转。这正是 Continuous Batching 相比静态批处理吞吐提升 10~20 倍的原因。
</details>

<details>
<summary><b>题 2</b>：一个 7B 模型 FP16 权重占 14GB 显存，80GB 的 GPU 上跑 vLLM。为什么 `--gpu-memory-utilization 0.9` 时，能同时服务的请求数不是（80×0.9−14）÷ 每请求 KV 大小，而是取决于 Block 分配？PagedAttention 在这里起了什么作用？（简答）</summary>

**答**：因为 KV Cache 不是按"请求"预分配整块显存，而是按 **Block（块）** 按需分配的。PagedAttention 把剩余显存（约 58GB）切成固定大小的块，每个请求按生成进度动态申请；请求提前结束时，未用的块立即释放给其他请求。所以并发上限由"总块数 ÷ 平均每请求实际用块数"决定，而不是"预留给单个请求的最大长度"。这消除了 60%~80% 的预留浪费，同显存并发数提升 2~4 倍。
</details>

<details>
<summary><b>题 3</b>：你的 RAG 应用里，1000 个用户提问前都要先发一段 20k token 的相同系统提示词。用 vLLM 和 SGLang 各会怎么处理？哪个更快，为什么？（简答）</summary>

**答**：vLLM 依赖 PagedAttention 的**块级共享**，相同前缀的 KV Cache 物理块会被多个请求共享，避免重复计算（vLLM 1.x 的 prefix caching 功能）；SGLang 用 **RadixAttention**，把所有请求的前缀组织成基数树，最长前缀命中后直接复用缓存、只算增量，缓存淘汰也更精细（LRU）。两者都能复用，但 SGLang 的树形结构对"长公共前缀 + 大量请求"的命中率更高、管理开销更低——这也是 SGLang 论文在对话/RAG 场景吞吐比 vLLM 高 3.6 倍的来源。结论：同前缀重度场景优先 SGLang。
</details>

---

## 10. 本章回顾

- **问题**：1000 并发不排队 = 让 GPU 永不空转 + 显存零浪费 + 计算不重复。
- **三件套**：Continuous Batching（时间）、PagedAttention（空间）、RadixAttention（复用）。
- **选型**：生产首选 vLLM；Agent/长上下文用 SGLang；本地无卡用 llama.cpp；国产一条龙用 lmdeploy。
- **动手**：`vllm serve` 一条命令起服务，OpenAI SDK 直接调用，100 并发轻松吃下。
- **排障**：slow / OOM / 吞吐低 → 查 ml-engineering 的 `inference/` 和 `debug/`。

> 下一章（06 推理层进阶 or 07 编排层）：当服务不止一个模型、要动态扩容、要按用户限流时，推理层之上还需要什么？——进入 **07 编排层**。

---

## 参考链接

- [vLLM 官方文档](https://docs.vllm.ai) / [GitHub ⭐90k](https://github.com/vllm-project/vllm)
- [SGLang 官方文档](https://docs.sglang.ai) / [GitHub ⭐32k](https://github.com/sgl-project/sglang)
- [llama.cpp GitHub ⭐125k](https://github.com/ggerganov/llama.cpp)
- [lmdeploy GitHub ⭐8k](https://github.com/InternLM/lmdeploy)
- [Orca 论文：Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM 论文：PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang 论文：RadixAttention](https://arxiv.org/abs/2312.07104)
- [awesome-ml-engineering（inference/ 与 debug/）](https://github.com/huggingface/awesome-ml-engineering)
- [openai-cookbook（客户端代码结构参考）](https://github.com/openai/openai-cookbook)

---

**上一章**：[第4章 训练层](../04-训练层/README.md) | **下一章**：[第6章 编排层](../06-编排层/README.md)
