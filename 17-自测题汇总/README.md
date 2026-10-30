---
title: 第17章 自测题汇总
tags: ['ai-infra', '教程', '自测', '测验', '昇腾']
章节: 17
状态: 完成
---

# 第17章：自测题汇总

> 所有章节的自测题汇总，用于复习和检验学习效果。含**昇腾专项自测**（昇腾用户重点看）。

## 第1章：全景图 + 硬件层

1. 用一句话向同事解释"NPU 和 CPU 的区别"，以及"HBM 和内存的区别"。
2. `npu-smi info` 显示 NPU 利用率 0% 但 HBM 占用 70%，说明什么？可能的原因有哪些？
3. `torch.npu.is_available()` 返回 `False`，你的排查步骤是什么？（写出至少 3 步）

## 第2章：算子层

1. Triton 和 Ascend C 的主要区别是什么？（提示：Python DSL vs C++ 原生）
2. FlashAttention 优化了什么？（提示：减少 HBM 访问次数）
3. 昇腾的 UB（Unified Buffer）对应 GPU 的什么？（提示：Shared Memory）
4. TileLang 和 Triton 的抽象层次差异是什么？（提示：手写调度 vs 编译器自动调度）
5. TileLang 的昇腾后端怎么启用？目标芯片是什么？（提示：`target="ascend"`，950/dav-3510）

## 第3章：框架层

1. PyTorch 的动态图和 TensorFlow 的静态图有什么区别？（提示：调试方便 vs 性能优化）
2. torch_npu 解决了什么问题？（提示：PyTorch 跑在昇腾 NPU 上）
3. 为什么需要 `loss.backward()`？（提示：触发梯度计算）

## 第4章：训练层

1. ZeRO Stage 1/2/3 分别优化了什么？（提示：优化器状态/梯度/参数）
2. 昇腾训练栈中，ModelLink / MindSpeed / AscendSpeed 分别对应什么？（提示：核心框架/加速库/社区版）
3. 昇腾多卡训练的通信库是什么？（提示：HCCL，对标 NCCL）

## 第5章：推理层

1. PagedAttention 解决了什么问题？（提示：KV Cache 显存管理）
2. vLLM-Ascend 和 vLLM 什么关系？（提示：硬件插件）
3. MindIE 和 vLLM-Ascend 的区别是什么？（提示：华为官方 vs 社区插件）
4. SGLang 在昇腾上的算子有哪两条开发路径？（提示：Ascend C 经 torch.ops.npu / Triton-Ascend 直接 import）
5. SGLang 昇腾版的版本配套为什么不能混搭？（提示：CANN/torch_npu/triton-ascend 互相绑定）

## 第6章：编排层

1. Volcano 和默认 K8s 调度器有什么区别？（提示：批处理/Gang Scheduling vs 通用）
2. 昇腾 NPU 在 K8s 中的资源名是什么？（提示：`huawei.com/Ascend910`）
3. ascend-device-plugin 的作用是什么？（提示：让 K8s 识别 NPU）

## 第7章：应用层

1. Dify 和 LangChain 的区别是什么？（提示：低代码 vs 代码）
2. FastGPT 的核心功能是什么？（提示：知识库+问答）
3. LangChain 的 Chain 和 Agent 有什么区别？（提示：固定流程 vs 自主决策）

## 第8章：实战路径

1. Week 1 的核心目标是什么？（提示：本地跑通模型）
2. vLLM-Ascend 怎么启动 API 服务？（提示：`vllm serve` + `VLLM_TARGET_DEVICE=ascend`）
3. 昇腾多卡训练用什么环境变量控制可见设备？（提示：`ASCEND_RT_VISIBLE_DEVICES`）

## 第9章：避坑指南

1. 遇到 NPU OOM，第一步应该做什么？（提示：减小 batch size / `npu-smi` 清场）
2. 多卡训练只有 1 卡在跑，可能是什么原因？（提示：HCCL 配置 / `ASCEND_RT_VISIBLE_DEVICES`）
3. `HCCL error` 报错怎么排查？（提示：单卡→单机多卡→多机，三步法）

## 第10章：开源框架昇腾适配

1. 昇腾适配的三条路径分别是什么？（提示：原生支持 / 插件适配 / Triton 间接）
2. `triton-ascend` 为什么装不上？怎么解决？（提示：不在默认 PyPI）
3. FlashInfer、FBGEMM、TensorRT 在昇腾上不可用，分别用什么替代？（提示：FlagAttention / msAMCT / MindIE）

## 第11章：数据工程

1. DVC 和 Git 的区别是什么？（提示：Git 管代码，DVC 管数据）
2. 数据清洗的常用步骤有哪些？（提示：去重、去空值、格式统一）
3. 为什么需要数据版本控制？（提示：实验可复现）

## 第12章：模型压缩

1. 量化的主要类型有哪些？（提示：PTQ vs QAT）
2. 昇腾的模型压缩工具有哪些？（提示：msAMCT、MindIE 量化）
3. msAMCT 和 TensorRT 的关系是什么？（提示：昇腾对标 GPU 的方案）

## 第13章：模型评测

1. 基准测试和人类评估的优缺点是什么？（提示：客观可复现 vs 接近真实体验）
2. 昇腾生态有哪些评测框架？（提示：FlagEval 启智评测）
3. 什么是基准测试过拟合？（提示：训练数据泄露）

## 第14章：模型部署进阶

1. Kserve 和 MindIE 的区别是什么？（提示：K8s 原生 vs 华为官方）
2. 什么是冷启动问题？（提示：模型加载需要时间）
3. 昇腾生产部署的首选方案是什么？（提示：MindIE）

## 第15章：MLOps

1. MLflow 的四个核心功能是什么？（提示：跟踪、项目、模型、注册）
2. 为什么需要 DVC？（提示：Git 不适合管理大数据）
3. 实验不可复现的常见原因有哪些？（提示：随机种子、环境、数据版本）

## 第16章：边缘部署

1. llama.cpp 和 PyTorch 的区别是什么？（提示：C++ 推理 vs Python 训练）
2. 昇腾的边缘推理硬件是什么？（提示：310P）
3. 为什么边缘部署需要量化？（提示：内存和算力限制）

## 第18章：DeepSeek 开源体系

1. DeepInfer 与 vLLM-Ascend 在昇腾支持上的本质区别是什么？（提示：原生一等公民 vs 硬件插件）
2. 为什么 MoE + MLA 架构需要 FlashMLA 和 DeepEP+EPLB 这两类专用组件？（提示：低秩 KV 结构 / 热点专家）
3. 当前开源版 DeepInfer 缺哪三个在线服务组件？（提示：KV 缓存系统 / PD 分离 / 负载均衡）

---

## 昇腾专项自测 ⭐⭐⭐

> 如果你只用昇腾，做这 10 道题就够了。

1. **全景认知**：昇腾软件栈从底到顶依次是什么？（CANN → AscendCL → torch_npu → PyTorch）
2. **硬件选型**：910B 和 310P 分别适合什么场景？（训练/推理 vs 边缘推理）
3. **算子开发**：Triton-Ascend 和 Ascend C 怎么选？（快速原型 vs 极致性能）
4. **框架适配**：torch_npu 和 torchair 的区别？（运行时 vs 编译时）
5. **训练并行**：昇腾训练栈三件套是什么？（ModelLink/MindSpeed/AscendSpeed）
6. **通信库**：HCCL 对标什么？怎么排查通信问题？（NCCL；单卡→单机多卡→多机）
7. **推理引擎**：vLLM-Ascend 和 MindIE 怎么选？（社区生态 vs 华为官方）
8. **集群编排**：昇腾在 K8s 里的资源名和 device plugin 是什么？（`huawei.com/Ascend910` / ascend-device-plugin）
9. **性能分析**：昇腾的性能分析工具是什么？（msprof）
10. **模型压缩**：昇腾的量化工具是什么？（msAMCT）
11. **框架适配**：昇腾适配的三条路径怎么选？（原生 > 插件 > Triton 间接）
12. **版本配套**：CANN / torch_npu / triton-ascend 版本必须一致吗？（是，错一个就报错）

---

## 答案速查

| 章节 | 题1 | 题2 | 题3 |
|------|-----|-----|-----|
| 第1章 | NPU=多而糙并行计算，CPU=少而精复杂逻辑；HBM=NPU 工作台，内存=CPU 工作台 | HBM 被占但计算没跑，可能数据加载瓶颈或残留进程 | ①npu-smi 确认 NPU ②确认 torch_npu 安装 ③确认 CANN 版本兼容 |
| 第2章 | Triton=Python DSL 跨硬件，Ascend C=C++ 昇腾专用 | 减少 HBM 访问，O(n²)→O(n) | Shared Memory（共享内存） |
| 第3章 | 动态图=边运行边建图，静态图=先建图再运行 | 让 PyTorch 模型跑在昇腾 NPU 上 | 触发反向传播，计算所有参数梯度 |
| 第4章 | Stage1=优化器状态，Stage2=梯度，Stage3=参数 | ModelLink=核心框架，MindSpeed=加速库，AscendSpeed=社区版 | HCCL |
| 第5章 | KV Cache 显存分页管理，减少碎片 | vLLM-Ascend 是 vLLM 的昇腾硬件插件 | MindIE=华为官方闭源，vLLM-Ascend=社区开源插件 |
| 第6章 | Volcano=批处理/Gang Scheduling，默认=通用 | `huawei.com/Ascend910` | 让 K8s 识别 NPU 设备并分配 |
| 第7章 | Dify=拖拽式低代码，LangChain=代码编排 | 知识库问答，文档→向量→检索 | Chain=固定流程，Agent=自主决策 |
| 第8章 | 本地运行 LLM，输入文字输出回答 | `VLLM_TARGET_DEVICE=ascend vllm serve <模型>` | `ASCEND_RT_VISIBLE_DEVICES=0,1` |
| 第9章 | 减小 batch size，用 npu-smi 查残留进程 | HCCL 没配好 / ASCEND_RT_VISIBLE_DEVICES 只设了 1 卡 | 三步法：单卡→单机多卡→多机，逐层定位 |
| 第10章 | ①原生支持(vLLM-Ascend/SGLang/FLA) ②插件适配(torch_npu) ③Triton间接(FlagGems) | 加 `--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple` | FlashInfer→FlagAttention、FBGEMM→msAMCT、TensorRT→MindIE |
| 第11章 | Git 管代码，DVC 管数据 | 去重、去空值、格式统一、文本清洗、标准化 | 代码+数据版本对应，实验可复现 |
| 第12章 | PTQ=训练后量化，QAT=量化感知训练 | msAMCT、MindIE 量化工具 | 都是"硬件专属极致优化"，msAMCT=昇腾，TensorRT=GPU |
| 第13章 | 基准测试=客观可复现，人类评估=接近真实体验 | FlagEval（启智评测，昇腾原生）、OpenCompass | 训练数据泄露，测试集表现异常好 |
| 第14章 | Kserve=K8s 声明式，MindIE=华为官方推理引擎 | 模型加载需要时间，第一个请求超时 | MindIE |
| 第15章 | 跟踪、项目、模型、注册 | Git 不适合管理大数据 | 随机种子、环境、数据版本不一致 |
| 第16章 | llama.cpp=C++ 推理，PyTorch=Python 训练 | 昇腾 310P（边缘推理卡） | 内存和算力限制，量化缩小体积 |
| 第18章 | DeepInfer=昇腾950 原生一等公民，vLLM-Ascend=vLLM 硬件插件 | MLA=低秩 KV 需专用内核；MoE 热点专家需 EP+EPLB | KV 缓存系统、PD 分离、负载均衡 |

注：第2章新增题——TileLang=编译器自动调度（Triton 手写调度）、`target="ascend"` 面向昇腾950（dav-3510）；第5章新增题——Ascend C（torch.ops.npu 调用）+ Triton-Ascend（直接 import）双路径、CANN/torch_npu/triton-ascend 版本互相绑定。

---

**上一章**：[第16章 边缘部署](../16-边缘部署/README.md) | **返回目录**：[教程首页](../README.md)
