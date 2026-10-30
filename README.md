# AI Infra 全景学习教程

> 从"能调 API"到"能部署/优化/排障"的完整学习路径 —— **以昇腾 NPU 为主线，GPU 作为参考对照**

## 这是什么

面向有 Python 基础、但无 AI Infra 经验的工程师，系统讲解大模型基础设施全栈知识。

特点：

- **昇腾中心**：每个技术点以昇腾方案为主（CANN / torch_npu / HCCL / MindIE / vLLM-Ascend），GPU 方案作对照参考
- **问题驱动**：每章按「问题 → 方案 → 项目 → 代码 → 排障」五段式组织
- **可运行**：每章配动手实验，代码可直接跑
- **可验证**：每章末尾 3 道折叠答案自测题，第17章汇总全部题库

## 目录导航

> 主线：第1-9章（核心知识）→ 第10章（开源框架适配）→ 第11-16章（扩展领域）→ 第17章（自测题汇总）

| 章节 | 主题 | 核心内容 | 难度 | 行数 |
|------|------|---------|------|------|
| [第1章](./01-全景图/README.md) | 全景图 + 硬件层 | 7 层架构、NPU/GPU/TPU、npu-smi、昇腾 OOM 排障 | ⭐ | 434 |
| [第2章](./02-算子层/README.md) | 算子层 | Triton-Ascend、Ascend C、FlashAttention、FlagGems | ⭐⭐ | 679 |
| [第3章](./03-框架层/README.md) | 框架层 | 自动微分、PyTorch、torch_npu、动态图 vs 静态图 | ⭐ | 563 |
| [第4章](./04-训练层/README.md) | 训练层 | 显存计算、ZeRO、并行策略、DeepSpeed/HCCL | ⭐⭐⭐ | 727 |
| [第5章](./05-推理层/README.md) | 推理层 | Continuous Batching、PagedAttention、vLLM-Ascend | ⭐⭐ | 620 |
| [第6章](./06-编排层/README.md) | 编排层 | K8s、Volcano、ascend-device-plugin、NPU 调度 | ⭐⭐ | 996 |
| [第7章](./07-应用层/README.md) | 应用层 | Dify、FastGPT、LangChain、选型指南 | ⭐ | 342 |
| [第8章](./08-实战路径/README.md) | 实战路径 | 4 周昇腾实战计划、端到端项目、动手任务 | ⭐⭐ | 626 |
| [第9章](./09-避坑指南/README.md) | 避坑指南 | NPU OOM、HCCL 排障、诊断脚本、排障流程 | ⭐⭐⭐ | 475 |
| [第10章](./10-开源框架昇腾适配/README.md) | **开源框架昇腾适配** | **vLLM/SGLang/FLA/Liger/verl 适配矩阵 + 三条路径 + 版本配套** | ⭐⭐⭐ | 532 |
| [第11章](./11-数据工程/README.md) | 数据工程 | 数据采集→清洗→标注→增强→版本控制 | ⭐⭐ | 179 |
| [第12章](./12-模型压缩/README.md) | 模型压缩 | 量化、剪枝、蒸馏、msAMCT、GPTQ | ⭐⭐ | 490 |
| [第13章](./13-模型评测/README.md) | 模型评测 | 基准测试、OpenCompass、FlagEval | ⭐⭐ | 166 |
| [第14章](./14-模型部署进阶/README.md) | 模型部署进阶 | Kserve、MindIE、Seldon Core、Istio | ⭐⭐⭐ | 165 |
| [第15章](./15-MLOps/README.md) | MLOps | MLflow、W&B、DVC、CI/CD | ⭐⭐ | 181 |
| [第16章](./16-边缘部署/README.md) | 边缘部署 | llama.cpp CANN、310P、ONNX Runtime | ⭐⭐ | 151 |
| [第17章](./17-自测题汇总/README.md) | 自测题汇总 | 16 章题目 + 昇腾专项 12 题 + 答案速查 | - | 153 |
| [第18章](./18-DeepSeek开源体系/README.md) | DeepSeek 开源体系 | V4 架构、DeepInfer 推理框架（昇腾950 原生）、FlashMLA/DeepGEMM/DeepEP、TileLang | ⭐⭐⭐ | ~170 |

## 学习路径建议

### 快速路径（1 周）
- 第1章（全景图）→ 第5章（推理层）→ 第8章（实战路径 Week 1-2）

### 标准路径（4 周）
- Week 1：第1-3章（硬件+算子+框架）
- Week 2：第4-5章（训练+推理）
- Week 3：第6-7章（编排+应用）
- Week 4：第8-9章（实战+排障）

### 深度路径（8 周）
- Week 1-4：标准路径
- Week 5：第10章（开源框架昇腾适配）+ 第11章（数据工程）
- Week 6：第12章（模型压缩）+ 第13章（模型评测）
- Week 7：第14章（部署进阶）+ 第15章（MLOps）+ 第16章（边缘部署）
- Week 8：综合实战项目

### 昇腾专项路径（2 周，只学昇腾）
- Week 1：第1章（硬件/npu-smi）→ 第2章（Triton-Ascend/Ascend C）→ 第5章（vLLM-Ascend）
- Week 2：第10章（**框架适配矩阵**）→ 第17章（昇腾专项 12 题自测）

## 昇腾 vs GPU 速查

| 能力 | 昇腾（主线） | GPU（参考） |
|------|------------|-----------|
| 计算架构 | CANN | CUDA |
| 设备监控 | `npu-smi info` | `nvidia-smi` |
| PyTorch 适配 | `torch_npu` + `torch.npu` | `torch.cuda` |
| 集合通信 | HCCL | NCCL |
| 算子开发 | Ascend C / Triton-Ascend | CUDA C++ / Triton |
| 推理引擎 | MindIE / vLLM-Ascend | TensorRT / vLLM |
| 注意力内核 | FlagAttention | FlashInfer |
| 模型压缩 | msAMCT | TensorRT / FBGEMM |
| 性能分析 | msprof | ncu / nsys |

## 参考资源

本教程的编写参考了以下开源项目（结构、代码风格、排障方法论）：

| 项目 | 用途 |
|------|------|
| [openai/openai-cookbook](https://github.com/openai/openai-cookbook) | "问题→代码→解释"三段式结构、示例组织方式 |
| [stas00/ml-engineering](https://github.com/stas00/ml-engineering) | "症状→根因→解法"排障方法论 |
| [PacktPublishing/LLM-Engineers-Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook) | 分层架构、端到端流水线设计 |

## 项目结构

```
ai-infra-tutorial/
├── README.md                    # 本文件（总览 + 导航）
├── 01-全景图/README.md          # 7 层架构、昇腾硬件、npu-smi
├── 02-算子层/README.md          # Ascend C、Triton-Ascend、FlashAttention
│   └── code/vector-add/         # 可运行的 Triton 示例
├── 03-框架层/README.md          # PyTorch、torch_npu、自动微分
├── 04-训练层/README.md          # ZeRO、并行策略、MindSpeed、HCCL
├── 05-推理层/README.md          # PagedAttention、vLLM-Ascend、MindIE
├── 06-编排层/README.md          # K8s、Volcano、NPU device plugin
├── 07-应用层/README.md          # Dify、FastGPT、LangChain
├── 08-实战路径/README.md        # 4 周实战计划 + 动手实验
├── 09-避坑指南/README.md        # NPU OOM、HCCL 排障、诊断脚本
├── 10-开源框架昇腾适配/README.md # 适配矩阵、三条路径、版本配套
├── 11-数据工程/README.md        # 数据管线、DVC
├── 12-模型压缩/README.md        # 量化、剪枝、蒸馏、msAMCT
├── 13-模型评测/README.md        # OpenCompass、FlagEval
├── 14-模型部署进阶/README.md    # Kserve、MindIE
├── 15-MLOps/README.md           # MLflow、DVC、CI/CD
├── 16-边缘部署/README.md        # llama.cpp CANN、310P
└── 17-自测题汇总/README.md      # 全部自测题 + 答案速查
```

## 更新记录

- 2026-08-25：初始版本，10 章 + 自测题汇总
- 2026-08-27：补充动手实验到 01/05/07/08/09 章
- 2026-08-27：新增 11-16 章（数据工程/模型压缩/模型评测/部署进阶/MLOps/边缘部署）
- 2026-09-12：昇腾中心化改造（01/03/04/06/08/09 章昇腾占比提升至 55-98%）
- 2026-09-12：新增第10章「开源框架昇腾适配」，教程编号补全为 01-17
- 2026-10-25：新增第18章「DeepSeek 开源体系」（V4 架构 + DeepInfer 推理框架 + 算子三件套 + TileLang）
- 2026-10-25：02 章新增 §3.6 TileLang 小节；05 章新增 §5.3 SGLang 深潜；10 章矩阵补 DeepInfer/TileLang；17 章自测题同步

## License

MIT
