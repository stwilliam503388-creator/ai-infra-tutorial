---
title: 第18章 开源社区框架的昇腾适配
tags: ['ai-infra', '教程', '昇腾', 'vllm', 'sglang', 'fla', 'liger', 'verl', '适配']
章节: 10
状态: 完成
---

# 第10章：开源社区框架的昇腾适配

> **本章问题**：你想用 SGLang / FLA / Liger-Kernel 这些热门开源框架，但它们是为 GPU 写的——昇腾 NPU 上能跑吗？要改多少？
>
> **本章答案**：**大部分已经能跑了，而且不用改代码**。关键在于"三条适配路径"和"版本配套矩阵"。
>
> 前置：第2章（算子层）、第5章（推理层）。读完本章你将掌握：昇腾适配成熟度矩阵、三种适配路径的识别方法、5 个主流框架的迁移步骤、版本配套速查表。

---

## 🥢 类比开场：适配 = 换灶台但不换菜谱

延续餐厅类比：

| 场景 | 对应 |
|------|------|
| 换灶台（GPU→NPU） | 硬件从 NVIDIA 换成昇腾 |
| 菜谱不变（代码不改） | 上层框架代码无需修改 |
| 但锅铲要换（算子重编） | 底层 kernel 需要 triton-ascend 重编译 |
| 有的菜谱本来就能用电磁炉 | Triton 写的算子天然跨硬件 |

**核心认知**：昇腾适配的本质，是把 CUDA 生态的"锅铲"（算子）换成昇腾版本，而"菜谱"（模型代码/框架 API）保持不变。

---

## 18.1 昇腾适配成熟度矩阵 ⭐⭐

这是本章最重要的一张表。**✅ 官方支持 = 开箱即用，⚠️ = 需配置，❌ = 不可用**。

| 框架 | 昇腾支持 | 适配方式 | 说明 |
|------|---------|---------|------|
| **vLLM** | ✅ 官方 | 硬件插件 | `vllm-ascend` 独立仓库，每周例会 |
| **SGLang** | ✅ 官方 | 原生支持 | README 硬件列表明确含 Ascend NPUs |
| **mindie** | ✅ 官方 | 华为自研 | 昇腾原生推理引擎（闭源） |
| **lmdeploy** | ✅ 官方 | 原生支持 | 上海 AI Lab，中文生态 |
| **llama.cpp** | ✅ 官方 | CANN 后端 | 端侧推理，已合入上游 |
| **FLA** | ✅ 官方 | backend extra | `pip install flash-linear-attention[npu]`（v0.5+）|
| **Liger-Kernel** | ✅ 官方 | 源码安装 | README 有 Ascend NPU 安装章节 |
| **verl** | ✅ 官方 | 文档+recipe | `docs/ascend_tutorial/zh/` + verl-ascend-recipe 仓库 |
| **FlagGems** | ✅ 间接 | Triton 算子库 | 用 Triton 写，经 triton-ascend 编译 |
| **transformers** | ✅ 间接 | 经 torch_npu | 通用模型库，tokenizer 无关硬件 |
| **peft / trl** | ✅ 间接 | 经 torch_npu | 微调库，纯 PyTorch 逻辑 |
| **DeepSpeed** | ✅ 间接 | 经 torch_npu | 需配 HCCL 替代 NCCL |
| **Megatron-LM** | ⚠️ 分叉 | AscendSpeed | 华为维护的昇腾分叉版 |
| **FlashInfer** | ❌ 无 | — | CUDA 专用，昇腾用 FlagAttention 替代 |
| **FBGEMM** | ❌ 无 | — | Meta CUDA 专用，昇腾用 msAMCT 替代 |
| **TensorRT** | ❌ 无 | — | NVIDIA 专用，昇腾用 MindIE 替代 |
| **Mojo** | ❌ 无 | — | 昇腾集成探索中 |

### 关键结论

1. **推理引擎全绿**：vLLM / SGLang / lmdeploy / llama.cpp / MindIE 都能跑
2. **训练框架全绿**：verl / Liger-Kernel / FLA 都官方支持昇腾
3. **红色区域是 NVIDIA 专属优化库**：FlashInfer / FBGEMM / TensorRT —— 这些是"硬件极致优化"产品，天然绑定 CUDA
4. **Triton 是最大公约数**：用 Triton 写的算子（FlagGems、FLA 部分 kernel）天然跨硬件

---

## 18.2 三条适配路径 ⭐⭐

判断一个框架能不能在昇腾上跑，看它属于哪条路径：

### 路径 1：原生支持（最省事）

**特征**：框架作者自己维护昇腾后端

```
框架代码 → 检测到 NPU → 调用昇腾算子 → CANN
```

**代表**：vLLM-Ascend、SGLang、lmdeploy、llama.cpp、FLA、Liger-Kernel、verl

**用法**：按官方文档装对应版本，代码零改动

```python
# 代码完全不用改
from vllm import LLM
llm = LLM(model="Qwen/Qwen2-0.5B-Instruct")  # 自动检测 NPU
```

### 路径 2：插件适配（需要装插件）

**特征**：框架本身不感知昇腾，靠外部插件桥接

```
框架代码 → PyTorch API → torch_npu 插件 → CANN
```

**代表**：transformers、peft、trl、DeepSpeed（经 torch_npu）

**用法**：装 `torch_npu`，然后 `import torch_npu` 激活

```python
import torch
import torch_npu  # 关键：导入即激活

device = "npu"  # 之后 torch API 自动路由到 NPU
model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B").to(device)
```

### 路径 3：Triton 间接适配（自动跨硬件）

**特征**：算子用 Triton 写，Triton 自动编译到目标硬件

```
Triton kernel → triton-ascend 编译 → AscendNPU-IR → CANN
```

**代表**：FlagGems、Liger-Kernel（部分）、FLA（部分）、vLLM（部分算子）

**原理**：Triton DSL 是硬件无关的，后端负责生成目标代码

```python
# 同一份 Triton kernel，在 GPU 和 NPU 上都能跑
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, ...):
    ...
# 编译时由 triton-ascend 生成昇腾代码
```

### 路径速查

| 你想用的框架 | 走哪条路径 | 要装什么 |
|------------|-----------|---------|
| vLLM | 1 原生 | `vllm-ascend` |
| SGLang | 1 原生 | 按官方文档 |
| FLA | 1 原生 | `[npu]` extra |
| Liger-Kernel | 1 原生 | 源码 + Ascend PyPI 镜像 |
| verl | 1 原生 | 官方昇腾镜像 |
| transformers | 2 插件 | `torch_npu` |
| peft/trl | 2 插件 | `torch_npu` |
| DeepSpeed | 2 插件 | `torch_npu` + HCCL |
| FlagGems | 3 Triton | `triton-ascend` |
| FlashInfer | ❌ 不行 | 换 FlagAttention |

---

## 18.3 五个主流框架的迁移指南 ⭐⭐⭐

### 18.3.1 vLLM → vLLM-Ascend

**适配方式**：官方硬件插件（独立仓库）

```bash
# 1. 环境要求（必须版本配套）
# CANN 9.1.0 + PyTorch 2.10.0 + torch_npu 2.10.0.post4

# 2. 安装
pip install vllm-ascend

# 3. 启动服务（代码与 vLLM 完全一致）
export VLLM_TARGET_DEVICE=ascend
vllm serve Qwen/Qwen2-0.5B-Instruct --port 8000
```

**关键差异**：

| 项 | vLLM (GPU) | vLLM-Ascend |
|----|-----------|-------------|
| 设备环境变量 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| 后端标记 | 默认 CUDA | `VLLM_TARGET_DEVICE=ascend` |
| 通信库 | NCCL | HCCL |
| 支持矩阵 | 全模型 | 见官方 support_matrix（部分模型待支持）|

**注意**：vLLM-Ascend 有**支持矩阵**（Support Matrix），不是所有模型都支持。上手前先查 `docs.vllm.ai/projects/ascend/`。

### 18.3.2 SGLang

**适配方式**：官方原生支持（无需插件）

```bash
# 安装（参考官方文档的昇腾章节）
pip install "sglang[all]"

# 启动（RadixAttention 在 NPU 上同样生效）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-0.5B-Instruct \
    --port 30000
```

**SGLang 的昇腾优势**：
- RadixAttention（前缀共享 KV Cache）是纯软件优化，与硬件无关
- Constrained Decoding（结构化输出）同样无硬件依赖
- 适合高并发 Agent 场景

### 18.3.3 FLA（flash-linear-attention）

**适配方式**：官方 backend extra（v0.5+ 新增）

```bash
# NPU 后端安装（不是默认 PyPI 的 triton）
pip install flash-linear-attention[npu] \
    --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

**关键变化**（v0.5+）：
- `torch` / `triton` 不再在 base deps，必须选 backend extra
- backend 选项：`[cuda]` `[rocm]` `[xpu]` `[npu]` `[cpu]`
- NPU 走 `triton-ascend` 而非上游 `triton`

**为什么 FLA 重要**：线性注意力把 O(n²) 降到 O(n)，长序列场景的关键优化。昇腾官方支持意味着你可以直接用。

### 18.3.4 Liger-Kernel

**适配方式**：官方支持（源码安装 + 昇腾 PyPI 镜像）

```bash
# 版本要求（严格配套）
# torch == 2.9.0
# torch_npu == 2.9.0
# triton-ascend == 3.2.2（不在默认 PyPI）
# CANN == 9.1.0

pip install -e ".[dev]" --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

**Liger-Kernel 做什么**：把训练中的多个小算子融合成一个大算子，省显存。例如 RMSNorm + Linear + CrossEntropy 融合成一个 kernel。

**官方说明**："Our kernels inherit the full spectrum of hardware compatibility offered by Triton" —— 即 Triton 算子天然跨硬件。

### 18.3.5 verl（RL 训练）

**适配方式**：官方完整支持（文档 + 镜像 + recipe 仓库）

```bash
# 官方昇腾文档
# github.com/volcengine/verl/blob/main/docs/ascend_tutorial/zh/README.md

# 独立 recipe 仓库
# github.com/verl-project/verl-ascend-recipe
```

**verl 昇腾文档结构**：

| 目录 | 内容 |
|------|------|
| `get_start/` | Docker 镜像构建、安装指南、快速上手 |
| `feature_support/` | NPU 高级特性、环境变量说明 |
| `model_support/` | 支持的模型/算法列表、最佳实践 |
| `dev_guide/` | 模型迁移指南、精度调试、性能调优 |
| `faq/` | 常见问题 |
| `contribution_guide/` | 社区贡献 |

**为什么 verl 值得关注**：RLHF/RLVR 是当前大模型训练热点（o1/R1 类模型都靠这个），verl 是事实标准，且昇腾支持完整。

---

## 18.4 版本配套矩阵 ⭐⭐⭐

**这是昇腾生态最容易踩的坑**：CANN / torch_npu / triton-ascend / 框架版本必须配套，错一个就报错。

### 核心版本链

```
CANN（底层驱动+算子库）
  ↓ 决定
torch_npu（PyTorch 适配层）
  ↓ 决定
triton-ascend（Triton 后端）
  ↓ 决定
上层框架（vLLM-Ascend / Liger-Kernel / FLA）
```

### 版本配套速查（实测自官方 README）

| 框架 | CANN | torch | torch_npu | triton-ascend |
|------|------|-------|-----------|---------------|
| **vLLM-Ascend v0.23.0** | 9.1.0 | 2.10.0 | 2.10.0.post4 | 由 CANN 提供 |
| **Liger-Kernel** | 9.1.0 | 2.9.0 | 2.9.0 | 3.2.2 |
| **FLA** | 按 CANN | 按 [npu] extra | 按 [npu] extra | triton-ascend |
| **verl** | 见官方镜像 | 见官方镜像 | 见官方镜像 | 见官方镜像 |

### 昇腾包安装源

**关键**：`triton-ascend` **不在默认 PyPI**，必须从昇腾镜像装：

```bash
--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

### 版本检查脚本

```python
import torch
import torch_npu
import subprocess

print("PyTorch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU 可用:", torch.npu.is_available())

# CANN 版本（命令行）
# npu-smi info  → 顶部显示 CANN 版本
```

```bash
# 命令行检查
npu-smi info | head -5        # CANN 版本 + NPU 型号
pip show torch torch_npu | grep -E "Name|Version"
python -c "import triton; print('triton-ascend:', triton.__version__)"
```

---

## 18.5 未适配框架的替代方案 ⭐⭐

红色区域的框架不是"不能用"，而是"需要用昇腾等价物替代"。

| 不可用框架 | 为什么 | 昇腾替代 |
|-----------|--------|---------|
| **FlashInfer** | CUDA C++ 专用注意力内核 | **FlagAttention**（昇腾原生 FlashAttention）|
| **FBGEMM** | Meta CUDA 专用低精度算子 | **msAMCT**（昇腾量化工具）|
| **TensorRT** | NVIDIA 推理优化器 | **MindIE**（华为官方推理引擎）|
| **cuDNN** | NVIDIA 深度学习库 | **CANN**（昇腾算子库）|
| **NCCL** | NVIDIA 集合通信 | **HCCL**（华为集合通信）|
| **CUDA** | GPU 编程模型 | **AscendCL / Ascend C** |
| **Triton (upstream)** | CUDA 后端为主 | **triton-ascend**（昇腾后端）|
| **Mojo** | 无昇腾集成 | 暂无（用 Triton-Ascend 替代）|

### 替代映射表（记忆口诀）

```
CUDA      → CANN        （编程模型 → 计算架构）
cuDNN     → CANN ops    （深度学习库 → 算子库）
NCCL      → HCCL        （集合通信）
TensorRT  → MindIE      （推理优化器）
FlashInfer→ FlagAttention（注意力内核）
FBGEMM    → msAMCT      （低精度算子）
Triton    → triton-ascend（编译器后端）
```

---

## 18.6 动手实验：检查你的环境能跑哪些框架 ⭐⭐

**目标**：写一个脚本，检测当前昇腾环境支持哪些框架。

**步骤 1：环境检测脚本**

```python
#!/usr/bin/env python
"""check_ascend_env.py — 检测昇腾环境支持的框架"""

def check_env():
    results = {}

    # 1. 基础环境
    try:
        import torch
        results['torch'] = torch.__version__
    except ImportError:
        results['torch'] = '❌ 未安装'

    try:
        import torch_npu
        results['torch_npu'] = torch_npu.__version__
        results['npu_available'] = torch.npu.is_available()
    except ImportError:
        results['torch_npu'] = '❌ 未安装'
        results['npu_available'] = False

    # 2. Triton 后端
    try:
        import triton
        results['triton'] = triton.__version__
        # 判断是不是 triton-ascend
        results['triton_backend'] = 'ascend' if 'ascend' in str(triton.__file__).lower() else 'unknown'
    except ImportError:
        results['triton'] = '❌ 未安装'

    # 3. 可选框架
    for pkg in ['vllm', 'sglang', 'flash_linear_attention', 'liger_kernel', 'verl']:
        try:
            mod = __import__(pkg)
            results[pkg] = getattr(mod, '__version__', '✅ 已安装')
        except ImportError:
            results[pkg] = '— 未安装'

    return results


if __name__ == "__main__":
    print("=== 昇腾环境检测 ===\n")
    for k, v in check_env().items():
        print(f"  {k:22s}: {v}")
```

**步骤 2：运行并对照**

```bash
python check_ascend_env.py
```

**预期输出**（已配好环境的样子）：

```
=== 昇腾环境检测 ===

  torch                 : 2.10.0
  torch_npu             : 2.10.0.post4
  npu_available         : True
  triton                : 3.2.2
  triton_backend        : ascend
  vllm                  : 0.23.0
  sglang                : — 未安装
  flash_linear_attention: — 未安装
  liger_kernel          : — 未安装
  verl                  : — 未安装
```

**步骤 3：按矩阵决定下一步**

| 你的目标 | 装什么 |
|---------|--------|
| 部署推理服务 | `pip install vllm-ascend` |
| 高并发 Agent | `pip install "sglang[all]"` |
| 长序列优化 | `pip install flash-linear-attention[npu]` |
| 训练提速 | Liger-Kernel（源码装）|
| RL 训练 | verl（官方镜像）|

**常见错误**：

| 症状 | 原因 | 解法 |
|------|------|------|
| `triton-ascend` 装不上 | 用了默认 PyPI | 加 `--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple` |
| `torch_npu` import 报错 | 版本与 torch 不匹配 | 严格按配套矩阵（torch 2.10 ↔ torch_npu 2.10）|
| 装了 vllm 但检测不到 NPU | 装的是 CUDA 版 vllm | 装 `vllm-ascend`，并设 `VLLM_TARGET_DEVICE=ascend` |
| FLA 报 triton 找不到 | v0.5+ 后 torch/triton 不在 base deps | 必须选 `[npu]` extra |

---

## 18.7 配套生态全景图 ⭐

```
┌─────────────────────────────────────────────────────┐
│  推理框架：vLLM-Ascend · SGLang · lmdeploy ·         │
│           llama.cpp · MindIE                         │
├─────────────────────────────────────────────────────┤
│  训练框架：verl · Liger-Kernel · MindSpeed ·         │
│           ModelLink · AscendSpeed · DeepSpeed(torch_npu)│
├─────────────────────────────────────────────────────┤
│  算子库：FlagGems(Triton) · FlagAttention(Ascend C) ·│
│         FLA(Triton) · Liger(Triton) · msAMCT         │
├─────────────────────────────────────────────────────┤
│  编译/适配：triton-ascend · torch_npu · torchair     │
├─────────────────────────────────────────────────────┤
│  底座：CANN · HCCL · AscendCL                        │
└─────────────────────────────────────────────────────┘
```

**三条路线总结**：

1. **用现成的**：vLLM-Ascend / SGLang / MindIE（推理）、verl / Liger-Kernel（训练）
2. **自己写算子**：Triton-Ascend（快速）或 Ascend C（极致）
3. **移植开源框架**：走"路径 2 插件适配"（torch_npu 桥接）

---

## 18.8 自测题

1. 昇腾适配的三条路径分别是什么？各举一个代表性的框架。

<details>
<summary>参考答案</summary>

①**原生支持**：框架自己维护昇腾后端，如 vLLM-Ascend、SGLang、FLA、Liger-Kernel、verl；②**插件适配**：靠 torch_npu 桥接，如 transformers、peft、trl、DeepSpeed；③**Triton 间接适配**：算子用 Triton 写，自动跨硬件，如 FlagGems。
</details>

2. `triton-ascend` 为什么装不上？怎么解决？

<details>
<summary>参考答案</summary>

因为 `triton-ascend` 不在默认 PyPI 上。解决：安装时加昇腾镜像源 `--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple`。Liger-Kernel 和 FLA 的昇腾安装都需要这个源。
</details>

3. FLA 从 v0.5 起有什么重要变化？为什么？

<details>
<summary>参考答案</summary>

`torch` 和 `triton` 不再包含在基础依赖中，改为 backend extra（`[cuda]`/`[rocm]`/`[xpu]`/`[npu]`/`[cpu]`）。原因：之前 ROCm/XPU/NPU 用户会静默装上 CUDA 版的 wheel，导致环境错误。现在必须显式选后端，如 `pip install flash-linear-attention[npu]`。
</details>

4. FlashInfer、FBGEMM、TensorRT 在昇腾上不可用，分别用什么替代？

<details>
<summary>参考答案</summary>

FlashInfer → **FlagAttention**（昇腾原生 FlashAttention）；FBGEMM → **msAMCT**（昇腾量化工具）；TensorRT → **MindIE**（华为官方推理引擎）。这三者都是"NVIDIA 硬件极致优化"产品，天然绑定 CUDA，无法移植。
</details>

5. 为什么说"Triton 是最大公约数"？

<details>
<summary>参考答案</summary>

Triton DSL 是硬件无关的，同一份 kernel 代码通过不同后端（CUDA / ROCm / triton-ascend）编译到不同硬件。所以用 Triton 写的算子库（FlagGems、Liger-Kernel、FLA 部分 kernel）天然具备跨硬件能力，只需换后端。Liger-Kernel 官方也说明"kernels inherit the full spectrum of hardware compatibility offered by Triton"。
</details>

---

## 小结

- **适配成熟度**：推理引擎和训练框架已全绿，红色区只有 NVIDIA 专属优化库
- **三条路径**：原生支持（最省事）> 插件适配（torch_npu）> Triton 间接（自动跨硬件）
- **最大坑**：版本配套（CANN / torch_npu / triton-ascend / 框架版本必须一致）
- **关键源**：`triton-ascend.osinfra.cn/pypi/simple`（不在默认 PyPI）
- **替代映射**：CUDA→CANN、NCCL→HCCL、TensorRT→MindIE、FlashInfer→FlagAttention、FBGEMM→msAMCT

## 参考链接

- [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/ascend/)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [FLA INSTALL.md（含 NPU 后端说明）](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
- [Liger-Kernel README（含 Ascend NPU 章节）](https://github.com/linkedin/Liger-Kernel)
- [verl Ascend 使用教程](https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/zh/README.md)
- [verl-ascend-recipe 仓库](https://github.com/verl-project/verl-ascend-recipe)
- [triton-ascend 仓库](https://github.com/triton-lang/triton-ascend)
- [triton-ascend PyPI 镜像](https://triton-ascend.osinfra.cn/pypi/simple)
- [torch_npu（Ascend PyTorch Adapter）](https://gitee.com/ascend/pytorch)

---

**上一章**：[第9章 避坑指南](../09-避坑指南/README.md) | **下一章**：[第11章 数据工程](../11-数据工程/README.md)
