# 第18章 DeepSeek 开源体系：从模型权重到推理框架

> 昇腾视角 ⭐⭐⭐ | 本章关键词占比目标：昇腾相关 ≥55%

## 🥢 类比开场：DeepSeek 开源 = 餐厅后厨一块块往外搬

一家餐厅火了之后，大家最想要的是它的"秘方"。DeepSeek 的开源策略不是一次给一整本菜谱，而是分批搬：

1. **先给菜**（模型权重）：V3、R1、V3.2、V4 —— 你可以直接吃，但不知道怎么做的
2. **再给刀具**（算子库）：FlashMLA、DeepGEMM、DeepEP、TileKernels —— 关键工序的专用工具
3. **然后给刀法教材**（TileLang）：教你并让你能自己造刀具
4. **最后给整个后厨**（DeepInfer，2026-10-20）：生产环境实际在跑的推理框架

本章把这四批东西拼成一张完整地图，重点回答昇腾用户最关心的问题：**哪些能在昇腾上跑？怎么选？**

---

## 1. 开源地图：时间线总览

| 时间 | 开源内容 | 类别 |
|------|---------|------|
| 2025 开源周 | FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS | 算子/通信/存储 |
| 2026-04-24 | DeepSeek-V4 预览版（权重 + 技术报告） | 模型 |
| 2026-09-10 | V4.1-Flash（新架构系列最小尺寸） | 模型 |
| 2026-09-29 | V3.2-Exp（DSA 稀疏注意力，TileLang 实现） | 模型 + 算子 |
| 2026-09-30 | **昇腾基础组件全家桶**：TileLang 昇腾版、DeepGEMM、DeepEP、TileKernels、FlashMLA、DeepSelect | 昇腾基础设施 |
| 2026-10-17 | V4.1-Flash-1017 | 模型 |
| 2026-10-20 | **DeepInfer 推理框架** + deepseek-recipe 编解码组件 | 推理框架 |

**关键认知**：DeepSeek 的昇腾组件与 NVIDIA 组件**一一对应**——这不是"施舍一个阉割版"，而是把昇腾当作一等公民平台来建设。官方原话：与华为团队共同推进基于昇腾 950 的 128 卡超节点方案。

### 1.1 为什么昇腾 950 是转折点

在这批开源之前，昇腾用户跑 DeepSeek 新模型的常态是"等"：等华为或社区把 CUDA 算子移植成 Ascend C，等 vLLM-Ascend 跟进模型适配，周期以月计。2026-09-30 之后逻辑变了：

- **TileLang 昇腾版**封装了 Ascend C 底层指令，DeepSeek 内部写算子的同一套工具链直接可用，"等移植"变成"同一源码 `target=ascend` 重编译"
- 算子库（FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect）昇腾版与 NVIDIA 版**同步发布**，官方称多项关键用例性能接近硬件上限
- 目标平台明确是昇腾 950（PR/DT 系列，架构代号 `dav-3510`），与华为共建 128 卡超节点——这是训练级投入，不是推理玩具

注意代际边界：**950 之前的 A2/A3（910B/910C 等）不在这批原生支持的范围内**，这些平台仍走社区适配路径（Triton-Ascend、vLLM-Ascend、sgl-kernel-npu）。

---

## 2. V4 模型架构：推理框架要优化的对象 ⭐⭐

要理解 DeepInfer 的每个优化点，先看它服务的模型长什么样。

| 规格 | V4-Pro | V4.1-Flash |
|------|--------|-----------|
| 总参数 | 1.6T | 284B |
| 激活参数 | 49B | 13B |
| 上下文 | 1M | 1M |
| 架构 | MoE + MLA | MoE + MLA |

两个架构特征决定了推理框架的设计：

- **MoE（混合专家）**：每次前向只激活部分专家 → 需要**专家并行（EP）+ 负载均衡（EPLB）**，否则热点专家排长队、冷门专家闲置
- **MLA（多头潜在注意力）**：KV 压缩成低秩潜在向量 → 需要**专用注意力内核**（FlashMLA），标准 FlashAttention 不适用

> ⚠️ 置信度说明：V4 是否沿用 V3.2 的 DSA（DeepSeek Sparse Attention）稀疏注意力，官方资料未明说，本章不展开。V3.2 的 DSA kernel 是 TileLang 写的，这一点已确认。

---

## 3. DeepInfer：生产级推理框架（本章核心）⭐⭐⭐

### 3.1 它是什么

[DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache 2.0）是 DeepSeek **线上生产服务实际在跑**的推理框架，2026-10-20 开源。README 自述血统：**"最初主要基于 vLLM 构建，同时借鉴了 SGLang 和 TensorRT-LLM"**。

官方生产数据（注意：官方口径，尚无第三方复测）：单集群日吞吐 **155 万亿 tokens**。

### 3.2 核心优化清单

| 优化 | 解决什么 | 对应第几章的知识 |
|------|---------|----------------|
| FlashMLA | MLA 专用注意力内核 | 第 2 章算子层 |
| DeepGEMM | FP8 等低精度 GEMM | 第 2、12 章 |
| DeepEP + EPLB | MoE 专家并行通信 + 负载均衡 | 第 4 章训练层（并行策略同样适用推理） |
| DSpark + 置信度调度验证 | 投机解码：动态置信度决定验证时机，而非固定窗口 | 第 5 章推理层 |
| SWA Bounded Replay | Decode 阶段只重放有界后缀，减少重复 Prefill 计算 | 第 5 章 |
| Attention DP + balanced CP | 注意力数据并行 + 均衡上下文并行（1M 长上下文必需） | 第 4、5 章 |
| 异步零开销批调度器 | 调度不占 GPU 时间 | 第 5 章 |
| Batch-invariant 推理 | 相同输入+采样参数 → 相同输出（可复现性；当前仅 NVIDIA） | 第 13 章评测 |
| 结构化输出 | regex / JSON schema / EBNF | 第 5 章 |

自定义 kernel 大量用 **TileLang** 编写（见第 2 章 §3.6）——这就是为什么 TileLang 是读懂这套代码的钥匙。

### 3.3 开源边界：拿到了什么，没拿到什么 ⚠️

当前开源版本是**离线推理**，一个完整在线服务还缺：

- KV Cache 存储系统
- Prefill/Decode 分离部署（PD 分离）
- 数据并行副本间的负载均衡

官方解释：这些组件与内部基础设施耦合太深，暂未开源。**所以 DeepInfer 现阶段不是 vLLM/SGLang 的直接替代品**——它更适合"有 Blackwell 或昇腾集群、要大规模部署 V4.1-Flash"的用户，个人开发者跑单模型仍然用 vLLM/SGLang 更实际。

### 3.4 硬件与环境

| 项目 | 要求 |
|------|------|
| GPU（参考） | NVIDIA SM100（Blackwell：B200 等），CUDA 13 |
| **NPU** | **昇腾 950，CANN 9.2** |
| 编译器 | GCC 13+ |
| Python | 3.12 |
| 支持模型 | 目前仅 DeepSeek-V4.1-Flash（默认 checkpoint：V4.1-Flash-1017） |

**昇腾是一等公民**：与 NVIDIA 走两条硬件抽象路径，原生支持而非社区适配。官方参照部署模板：Prefill 单机 8 卡，Decode 64~128 卡超节点。

### 3.5 deepseek-recipe

配套开源的 Token 编解码组件，作用是让第三方部署时获得**与 DeepSeek 官方服务一致的 API 处理体验**（chat template、special token 处理等）。小组件，但解决了"自部署输出和官方 API 对不上"的经典坑。

---

## 4. 算子通信三件套 + TileKernels ⭐⭐

| 组件 | 作用 | 昇腾支持 |
|------|------|---------|
| FlashMLA | MLA 稀疏注意力内核 | ✅ 昇腾版已开源（2026-09-30） |
| DeepGEMM | FP8/低精度矩阵乘 | ✅ 昇腾版已开源 |
| DeepEP | MoE 专家并行通信 | ✅ 昇腾版已开源；SGLang 生态另有 DeepEP-Ascend（见第 5 章 §5.3） |
| TileKernels | 常规向量计算和访存算子 | ✅ 昇腾版已开源 |
| DeepSelect | 高效数据筛选 | ✅ 昇腾版已开源 |

官方口径：多项关键测试用例中，上述组件的计算与通信性能**已接近硬件上限**。

**昇腾实现的两点细节**：

1. **DeepEP 在昇腾上有两条线**：DeepSeek 官方的昇腾版（本表）和 SGLang 生态的 [DeepEP-Ascend](https://github.com/sgl-project/sgl-kernel-npu)（sgl-kernel-npu 的子组件，面向 A3 全互联 HCCS / A2 HCCS+RDMA 拓扑，支持 INT8/FP8/BF16 量化压缩通信）。前者配套 DeepInfer（950），后者配套 SGLang-NPU（A2/A3），选型时别搞混。
2. **TileKernels 与 TileLang 的关系**：TileKernels 里的算子大量用 TileLang 编写（NVIDIA 和昇腾同源）。这意味着昇腾 950 用户可以直接用 TileLang 改写/扩展这些算子，经毕昇编译器生成设备码——不需要回到 Ascend C 手写。

---

## 5. 昇腾选型对比：四条路径怎么选 ⭐⭐⭐

在昇腾上部署 DeepSeek 系模型，现在有四个选项：

| 框架 | 昇腾路径 | 成熟度 | 适合谁 |
|------|---------|--------|--------|
| **DeepInfer** | 原生支持 950（CANN 9.2） | 新（2026-10 首发），仅离线推理 | 有 950 集群、部署 V4.1-Flash 的厂商 |
| **vLLM-Ascend** | 插件（第 10 章路径 2） | 最成熟，社区最大 | 大多数团队的首选 |
| **SGLang-NPU** | 原生支持（sgl-kernel-npu） | 成熟，A2/A3/950PR 全覆盖 | 多轮对话/RAG/Agent 负载 |
| **MindIE** | 华为官方闭源 | 企业级 | 华为云/一体机用户 |

**一句话决策**：

- 手上的卡是 **950 且只跑 V4.1-Flash** → DeepInfer（官方生产同款调优）
- **910B/910C/A3** → vLLM-Ascend 或 SGLang-NPU（DeepInfer 只支持 950）
- 要**在线服务**（PD 分离、负载均衡）→ vLLM-Ascend / SGLang-NPU（DeepInfer 当前版本没有）

---

## 6. 动手：昇腾 950 上跑 V4.1-Flash ⭐⭐

```bash
# 环境：昇腾 950 + CANN 9.2 + GCC 13+ + Python 3.12
# 1. 确认硬件与驱动（应看到 950 系列设备）
npu-smi info

# 2. 确认 CANN 版本（DeepInfer 要求 9.2，低于此版本直接放弃）
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 拉代码安装
git clone https://github.com/deepseek-ai/DeepInfer.git
cd DeepInfer
# 按 README 安装；首次运行自动从 HuggingFace 下载
# deepseek-ai/DeepSeek-V4.1-Flash-1017 权重（284B 总参，激活 13B，需足量存储与显存）
```

**存储估算**：284B 参数即使 FP8 也要 ~284GB 权重，单机 8 卡 950 放不下时需要多机/超节点——这正是官方 Prefill 8 卡、Decode 64~128 卡模板的由来。个人/小团队没有 950 集群的，先用 vLLM-Ascend 在 A2/A3 上跑 Qwen 等模型练手（第 5 章 §6），DeepInfer 等社区跟进更多硬件或你拿到 950 资源再上。

> ⚠️ 本章代码为安装路径示意。DeepInfer 刚开源 5 天（2026-10-25 计），快速迭代中，以仓库 README 为准。

---

## 7. 避坑：三个容易误读的点

1. **"DeepInfer 支持昇腾" ≠ "支持所有昇腾卡"**：仅 950（dav-3510），910B/910C/A3 不在支持列表。910 系用户部署 DeepSeek 模型的现实路径仍是 vLLM-Ascend 或 SGLang-NPU
2. **"开源了推理框架" ≠ "开源了整套服务"**：KV 缓存系统、PD 分离、负载均衡未开源（§3.3）。昇腾上的 PD 分离目前要借助 SGLang + MemFabric 这类社区方案
3. **性能数字是官方口径**：155 万亿 tokens/天来自官方生产环境，无第三方复测；昇腾版算子的"接近硬件上限"同样是官方口径，引用时注明来源
4. **CANN 版本是硬门槛**：DeepInfer 要求 CANN 9.2，而 SGLang-NPU 当前配套 CANN 9.1.0——同一台 950 机器上两个框架的 CANN 版本可能冲突，部署前确认基线

---

## 8. 自测题

<details>
<summary>Q1：DeepInfer 与 vLLM-Ascend 在昇腾支持上的本质区别是什么？</summary>

DeepInfer 把昇腾 950 作为**原生一等公民**支持（与 NVIDIA 走两条硬件抽象路径，DeepSeek 官方维护）；vLLM-Ascend 是 vLLM 的**硬件插件**（社区/华为维护，跟随 vLLM 主仓库）。前者只支持 950 + V4.1-Flash，后者覆盖 A2/A3 全系 + 任意模型。

</details>

<details>
<summary>Q2：为什么 MoE + MLA 架构需要 FlashMLA 和 DeepEP+EPLB 这两类专用组件？</summary>

MLA 把 KV 压缩成低秩潜在向量，标准 FlashAttention 内核无法直接处理这种结构，需要专用内核 FlashMLA。MoE 每次前向只激活部分专家，专家分布在不同时会出现热点专家排队、冷门专家闲置，需要 DeepEP 做专家并行通信、EPLB 做负载均衡。

</details>

<details>
<summary>Q3：当前开源版 DeepInfer 不能直接用于在线服务，缺哪三个组件？</summary>

KV Cache 存储系统、Prefill/Decode 分离部署、数据并行副本间的负载均衡。

</details>

---

## 参考资料

- [DeepInfer GitHub](https://github.com/deepseek-ai/DeepInfer)
- [DeepSeek-V4 发布页](https://api-docs.deepseek.com/zh-cn/news/news260424)
- [DeepSeek 开源昇腾基础组件公告（2026-09-30）](https://news.qq.com/rain/a/20260930A03X0Y00)
- [tilelang PR #3308：Ascend 950 后端](https://github.com/tile-ai/tilelang/pull/3308)
- [SGLang 昇腾文档](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation)
