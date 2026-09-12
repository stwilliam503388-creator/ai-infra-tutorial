---
title: 第1章 AI Infra 全景图 + 硬件层
tags: [AI-Infra, 教程]
difficulty: ⭐~⭐⭐
---

# 第1章：AI Infra 全景图 + 硬件层

> 面向有 Python 基础、无 AI Infra 经验的工程师。读完本章，你将能在脑内画出 AI 系统的 7 层架构，并看懂 `npu-smi` 的输出（GPU 上对应 `nvidia-smi`）。本章以**昇腾 NPU 为主角**，GPU 作为参考对照。

## 1.1 全景图：AI 系统的 7 层架构 ⭐

先看一张总图，之后每一章都会回到这张图（昇腾为默认主角，括号内标注 GPU 对应项）：

```
┌──────────────────────────────────────────────────┐
│ ① 应用层 Application    ChatGPT、推荐、文生图        │
├──────────────────────────────────────────────────┤
│ ② 服务层 Service        vLLM-Ascend、Triton 推理    │
├──────────────────────────────────────────────────┤
│ ③ 编排层 Orchestration  K8s、Ray、Slurm            │
├──────────────────────────────────────────────────┤
│ ④ 框架层 Framework      PyTorch+torch_npu、MindSpore│
├──────────────────────────────────────────────────┤
│ ⑤ 算子层 Operator       FlashAttention、CANN 算子库  │
│                        （GPU 对应：cuDNN）          │
├──────────────────────────────────────────────────┤
│ ⑥ 底座层 Base          CANN / HCCL / AscendCL      │
│                        （GPU 对应：CUDA/NCCL）      │
├──────────────────────────────────────────────────┤
│ ⑦ 硬件层 Hardware       NPU（昇腾）、GPU、TPU        │
└──────────────────────────────────────────────────┘
```

新增的 ⑥ 底座层是昇腾生态的灵魂：PyTorch 代码要先经过 torch_npu 适配层，落到 CANN（昇腾计算架构），由 AscendCL 调用算子、HCCL 做多卡通信，最终驱动 NPU 芯片。

### 开餐厅类比

把 AI 系统想象成一家餐厅：

| 层 | 餐厅角色 | 例子 |
|---|---|---|
| 应用层 | 顾客点的菜 | ChatGPT 对话、图像生成 |
| 服务层 | 服务员上菜 | vLLM-Ascend 推理服务、API 接口 |
| 编排层 | 厨师长排单 | K8s 调度、任务编排 |
| 框架层 | 菜谱和厨具 | PyTorch 定义模型和训练 |
| 算子层 | 刀工火候 | FlashAttention 等高效实现 |
| 底座层 | 灶台的供气管道 | CANN 把"火"送到每一口锅（NPU） |
| 硬件层 | 灶台和锅 | 昇腾 NPU/GPU/TPU 算力 |

用户只关心菜好不好吃（应用层），但一道菜好不好吃，取决于从灶台到摆盘的每一层。排障时也是逐层排查：先看菜谱（框架），再看刀工（算子），接着查供气（底座），最后看火候（硬件）。

### 各层速览

| 层 | 核心问题 | 代表技术 | 典型角色 |
|---|---|---|---|
| 应用层 | 用户看到什么 | ChatGPT、RAG、Agent | 产品/算法工程师 |
| 服务层 | 模型怎么对外提供 | vLLM-Ascend、Triton Inference Server | 推理工程师 |
| 编排层 | 任务怎么调度 | Kubernetes、Ray、Slurm | 平台工程师 |
| 框架层 | 模型怎么定义和训练 | PyTorch + torch_npu、MindSpore | 算法工程师 |
| 算子层 | 计算怎么最快 | FlashAttention、CANN 算子库（GPU: cuDNN） | 算子/CANN 工程师 |
| 底座层 | 软件栈怎么衔接硬件 | CANN、HCCL、AscendCL（GPU: CUDA、NCCL） | 平台工程师 |
| 硬件层 | 计算在哪发生 | 昇腾 910B、H100、TPU v5 | 基础设施工程师 |

## 1.2 硬件层：AI 计算的"灶台" ⭐⭐

### NPU / GPU / TPU：主角与参考

| 维度 | 昇腾 NPU (Ascend) | NVIDIA GPU（参考） | Google TPU |
|---|---|---|---|
| 定位 | 国产算力主力（本章主角） | 事实标准 (de facto standard) | Google 自研 |
| 代表型号 | 昇腾 910B / 310P / 910C | A100 / H100 / H200 | TPU v5e / v5p |
| 软件栈 | CANN + torch_npu / MindSpore | CUDA + cuDNN | XLA + JAX |
| 生态成熟度 | ★★★ 快速追赶，中文文档友好 | ★★★★★ 最丰富 | ★★★ 仅限 Google Cloud |
| 适用场景 | 训练 + 推理全覆盖（国产化首选） | 训练 + 推理全覆盖 | 超大规模训练（内部） |
| 入手难度 | 中，文档中文友好 | 低，教程最多 | 高，基本接触不到 |

一句话总结：

- **NPU（主角）**：国产替代主力，华为昇腾生态。训练用 910B，推理/边缘用 310P，推理场景性价比高，生态正在快速补全。
- **GPU（参考）**：AI 时代的"x86"，几乎所有框架默认支持，遇到问题答案最多——本教程处处用它做对照。
- **TPU**：Google 内部大模型训练的"御用灶台"，外部开发者很少直接接触。

关键点：**软件栈决定一切**。同样的模型，换个硬件就要换软件栈——昇腾用 CANN，GPU 用 CUDA。这也是为什么生态成熟度比纸面算力更重要。

### 昇腾主力型号：910B 与 310P ⭐

| 型号 | 定位 | 关键规格 | 典型场景 |
|---|---|---|---|
| 昇腾 910B | 训练主力 | FP16 约 376 TFLOPS、64GB HBM | 大模型预训练/微调，如 Atlas 800T A2（8 卡） |
| 昇腾 310P | 推理主力 | INT8 约 140 TOPS | 视频分析、OCR、边缘推理，如 Atlas 300I 系列 |

选型口诀：**训练看 910B，推理看 310P**。两张卡软件栈一致（都是 CANN），模型代码可以平移，只是算力规格和功耗不同。

### 设备内存（HBM）是什么 ⭐

设备内存 = 加速卡自己的内存，是计算时的"工作台面"（昇腾叫 **HBM**，GPU 叫显存 **VRAM**）。

- 模型参数、中间激活值 (activation)、梯度、优化器状态都要放设备内存
- 内存不足 = 工作台面太小，东西摆不下 → 昇腾报 `HBM alloc failed`（GPU 报 `CUDA out of memory`）
- 对比：CPU 用内存 (RAM)，加速卡用 HBM/VRAM，两者不能直接互换

训练时设备内存占用的大致构成：

| 占内存的东西 | 说明 | 占比感受 |
|---|---|---|
| 模型参数 | 权重本身 | 1 份 |
| 梯度 (gradient) | 反向传播用 | ~1 份 |
| 优化器状态 | Adam 等额外状态 | 2~3 份 |
| 激活值 | 前向传播的中间结果 | 可大可小 |

这也是为什么训练 7B 模型通常需要几十 GB 设备内存，而推理只需要参数 + KV cache。

### npu-smi：NPU 的"仪表盘" ⭐⭐

**问题**：我怎么知道机器上有几张 NPU、每张还剩多少设备内存？

**代码**：在终端运行：

```bash
npu-smi info
```

**解释**：输出长这样（简化版，昇腾 910B）：

```
+-----------------------------------------------------------------------------+
| npu-smi 23.0.rc3    Driver Version: 23.0.rc3     CANN Version: 7.0.0         |
|-------------------------------+----------------------+----------------------+
| NPU  Name        Health       | Temp  Power         | HBM Memory-Usage      |
| 0    昇腾 910B    OK           | 46C   83W          | 62.3GB / 63.8GB       |
|-------------------------------+----------------------+----------------------+
| NPU-Util        AI Core 使用率 | 0%                 |                      |
+-------------------------------+----------------------+----------------------+
```

重点看 4 个字段：

| 字段 | 含义 | 常见坑 |
|---|---|---|
| `HBM Memory-Usage` | 设备内存使用量 / 总量 | 训练前先看还有没有别的进程占着 |
| `NPU-Util` | AI Core 计算单元利用率 (%) | 不是内存占用率，别混淆 |
| `Temp` | 温度 | 持续 90°C+ 注意散热 |
| `Power` | 当前功耗 | 利用率低但功耗高 → 有异常 |

**实用命令**（GPU 参考对照见右列）：

| 功能 | npu-smi（昇腾） | nvidia-smi（GPU 参考） |
|---|---|---|
| 基本信息 | `npu-smi info` | `nvidia-smi` |
| 动态刷新 | `watch -n 1 npu-smi info` | `nvidia-smi -l 1` |
| 板级信息（温度/功耗/HBM） | `npu-smi info -t board` | `nvidia-smi --query-gpu=...` |
| AI Core 使用率 | `npu-smi info -t usages` | `nvidia-smi --query-gpu=utilization.gpu` |
| 谁占用了 NPU | `npu-smi info -t process` | `nvidia-smi --query-compute-apps=...` |

> 新版 CANN 8.0+ 提供 `ascend-smi`，命令行风格与 nvidia-smi 几乎一致，也支持 `--query-*` 参数。

### PyTorch 检查 NPU：torch.npu.is_available() ⭐

**问题**：代码里怎么确认 PyTorch 能用上昇腾 NPU？

**代码**：

```python
import torch
import torch_npu  # 关键：昇腾 PyTorch 适配层，import 后才能用 torch.npu

print("PyTorch 版本:", torch.__version__)
print("NPU 可用:", torch.npu.is_available())
print("NPU 数量:", torch.npu.device_count())

if torch.npu.is_available():
    print("NPU 名称:", torch.npu.get_device_name(0))
    device = torch.device("npu:0")
else:
    device = torch.device("cpu")

# 把模型和张量放到设备上
model = model.to(device)
x = torch.randn(16, 768).to(device)
```

**解释**：

- GPU 上的 `torch.cuda.*` 在昇腾上对应 `torch.npu.*`，device 写法 `npu:0`（GPU 为 `cuda:0`）
- `torch.npu.is_available()` 返回 `False` 的三大原因：机器没昇腾设备、CANN 驱动/固件没装、torch_npu 没装或版本不匹配（最常见！）
- 检查方法：`pip list | grep torch`，看是否安装了 `torch_npu`
- 永远用 `device` 变量 + `.to(device)`，不要硬编码 `"npu:0"`——这样代码在无 NPU 的机器上也能跑

### 排障实战：昇腾 OOM（HBM alloc failed）⭐⭐

**症状**：训练一开始就报 `RuntimeError: HBM alloc failed` 或 `out of memory`（GPU 侧对应 `CUDA out of memory`）。

**根因**（按概率排序）：

1. batch size 太大，激活值撑爆设备内存
2. 别的进程占着设备内存（最常见：上一个训练任务没 kill 干净）
3. 模型本身太大，单卡放不下

**解法**：

1. 先 `npu-smi info -t process` 看内存被谁占了 → `kill -9 <pid>` 清场
2. 减小 `batch_size`（立竿见影）
3. 开启梯度累积 (gradient accumulation)，用时间换内存
4. 开启梯度检查点 (gradient checkpointing)，`model.gradient_checkpointing_enable()`
5. 换更大的卡或多卡并行（后续章节详解）

## 1.3 排障参考：ml-engineering ⭐⭐

遇到硬件/加速器问题，先查 stas00/ml-engineering 的排障手册：

> 📖 [ml-engineering — NVIDIA GPU debug 手册](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)

这份文档按"症状 → 根因 → 解法"组织，是社区公认的 GPU 排障圣经（昇腾侧可对照思路，命令换成 npu-smi）。核心涵盖：

| 主题 | 解决什么问题 |
|---|---|
| GPU 相关环境变量 | 如 `CUDA_VISIBLE_DEVICES` 控制可见 GPU（昇腾：`ASCEND_RT_VISIBLE_DEVICES`） |
| 性能诊断 | 加速卡利用率低、CPU 瓶颈 (bottleneck) |
| 内存问题 | OOM、内存碎片化 (fragmentation) |
| 集合通信 | 多卡训练时卡住、报错（昇腾用 HCCL） |

昇腾官方排障资源：

> 📖 [昇腾 CANN 文档中心](https://www.hiascend.com/document) — 官方手册，含安装、调优、故障排查章节

推荐工作流：遇到问题 → 先 `npu-smi info` 看硬件状态 → 查 CANN 文档 / ml-engineering 对应章节 → 最后才搜报错文本。

## 1.4 动手实验 ⭐~⭐⭐

三个实验带你亲手摸一遍本章内容，共约 20 分钟。类比：实验1 看灶台（硬件），实验2 点火（软件栈），实验3 故意把台面堆爆再收拾（排障）。

> 前置条件：一台带昇腾 NPU 的机器（如 Atlas 800 训练服务器、云上昇腾实例），已装好 CANN 工具包和 torch_npu。没有昇腾设备时，可用 NVIDIA GPU 机器对照学习（把 npu-smi → nvidia-smi、torch.npu → torch.cuda 即可），实验3 需真机复现。

### 实验1：用 npu-smi 探索你的 NPU ⭐

**目标**：像看仪表盘一样，读出一张 NPU 的型号、HBM、温度和功耗。

**步骤**：

```bash
# ① 基础命令：一张表看清所有 NPU
npu-smi info

# ② 动态刷新：每秒更新一次，观察负载变化（Ctrl+C 退出）
watch -n 1 npu-smi info

# ③ 板级信息：型号 / HBM / 温度 / 功耗
npu-smi info -t board
```

**预期输出**（③ 的简化版，数值因机器而异）：

```
NPU  Name        Health   Temp   Power      HBM Memory-Usage
0    昇腾 910B    OK       42C    68W/400W   2.1GB / 63.8GB
```

对照检查：HBM 总量是灶台大小；`Temp` 持续 90°C+ 要警惕；`Power` 远低于上限说明没在全力烧火。

**常见错误**：

| 现象 | 原因 | 解决 |
|---|---|---|
| `npu-smi: command not found` | CANN 工具链没装或 PATH 不对 | 装 CANN 工具包；或 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 只显示 CPU，没有 NPU 行 | 机器无昇腾设备，或虚拟机没直通 (passthrough) | 云上换昇腾实例，或检查虚拟化设置 |
| `open /dev/davinci0 failed` | 驱动/固件未装，或用户权限不足 | 检查驱动；把用户加入 `HwHiAiUser` 组后重登 |

### 实验2：用 torch.npu.is_available() 检查 PyTorch NPU 支持 ⭐

**目标**：在代码里确认 PyTorch 真的能用上 NPU——"驱动有"和"PyTorch 能调"是两回事。

**步骤**：新建 `check_npu.py`：

```python
import torch
import torch_npu  # 关键：必须 import，否则 torch.npu 不存在

print("PyTorch 版本:", torch.__version__)
print("NPU 可用:", torch.npu.is_available())
print("NPU 数量:", torch.npu.device_count())

if torch.npu.is_available():
    print("NPU 名称:", torch.npu.get_device_name(0))
    print("HBM 总量:", torch.npu.get_device_properties(0).total_memory / 1e9, "GB")
else:
    print("没有可用 NPU，将使用 CPU")
```

运行：`python check_npu.py`

**预期输出**（有 NPU 时）：

```
PyTorch 版本: 2.1.0
NPU 可用: True
NPU 数量: 1
NPU 名称: 910B
HBM 总量: 63.8 GB
```

**常见错误**：

| 现象 | 原因 | 解决 |
|---|---|---|
| `is_available()` 返回 `False`，但 npu-smi 正常 | torch_npu 没装或版本与 CANN/PyTorch 不匹配（最常见） | `pip list \| grep torch` 看有无 torch_npu；按昇腾版本对照表装匹配组合 |
| `ModuleNotFoundError: No module named 'torch_npu'` | 没装 torch_npu | 安装与 CANN、PyTorch 匹配的 torch-npu 版本 |
| `RuntimeError: npu device error` | 驱动/CANN 与固件版本不匹配 | 按 CANN 文档重装匹配的驱动和工具包 |

### 实验3：故意制造昇腾 OOM 并修复 ⭐⭐

**目标**：亲手复现本章 1.2 讲的 `HBM alloc failed`，再用"减小 batch size"修好。类比：往灶台台面（设备内存）上不停堆菜（张量），堆不下就报错；少上几道菜（小 batch）就能继续。

**步骤**：新建 `oom_demo.py`，先跑"闯祸版"：

```python
import torch
import torch_npu

device = torch.device("npu:0")
total = torch.npu.get_device_properties(0).total_memory / 1e9
print(f"HBM 总量: {total:.0f} GB")

tensors = []
for i in range(100):
    x = torch.randn(256, 1024, 1024, device=device)  # 每块约 1GB
    tensors.append(x)  # 不释放，越堆越多
    print(f"已分配 {i+1} 块，约 {(i+1)} GB")
```

**预期输出**（后半段）：

```
HBM 总量: 64 GB
已分配 1 块，约 1 GB
已分配 2 块，约 2 GB
...
RuntimeError: HBM alloc failed ... (昇腾版 out of memory)
```

看到报错就对了——这正是 1.2 说的"台面摆不下"。GPU 机器上对应报 `CUDA out of memory`。

**修复**：把"一口气堆 100 块"改成"一次一小批，用完即扔"：

```python
import torch
import torch_npu

device = torch.device("npu:0")
batch_size = 8  # 关键修复：一次只处理一小批

for step in range(100):
    x = torch.randn(batch_size, 256, 1024, 1024, device=device)  # 每次只占 ~8GB
    y = x @ torch.randn(256, 1024, device=device)
    del x, y                      # 用完即扔，腾出台面
    torch.npu.empty_cache()       # 主动回收碎片（GPU 对应 torch.cuda.empty_cache()）
    if step % 10 == 0:
        print(f"step {step}: 成功，batch_size={batch_size}")
```

**预期输出**：

```
step 0: 成功，batch_size=8
step 10: 成功，batch_size=8
step 20: 成功，batch_size=8
...
```

**常见错误**：

| 现象 | 原因 | 解决 |
|---|---|---|
| 还没跑到 OOM，机器先卡死/无响应 | 一次性分配过大，系统内存 (RAM) 先被吃光 | 别在生产机器上跑"闯祸版"；或用更小的张量 |
| 修复后仍 OOM | 设备内存被别的进程占着 | 先 `npu-smi info -t process` 看占用 → `kill -9 <pid>` 清场（见 1.2） |
| 不同机器 OOM 点不同 | 每张卡内存大小不同，100 只是安全的大数 | 小卡把 batch_size 调到 2；大卡想更快触发 OOM 就把 100 调大 |

**小结**：三个实验对应三种角色——实验1 是巡检（看硬件），实验2 是验收（看软件栈），实验3 是排障（问题定位）。这也是后面每一章都会用到的三板斧。

## 1.5 自测题

**第 1 题** ⭐：用一句话向同事解释"GPU 和 CPU 的区别"，以及"设备内存（HBM/显存）和内存的区别"。

<details>
<summary>参考答案</summary>

CPU 是"少而精"：核少但单核能力强，擅长复杂逻辑；GPU/NPU 是"多而糙"：几千个核做简单并行计算，正好匹配矩阵乘法。设备内存是加速卡的工作台面，内存是 CPU 的工作台面，AI 训练数据主要摆在设备内存上。
</details>

**第 2 题** ⭐⭐：`npu-smi info` 显示 `NPU-Util 0%` 但 `HBM Memory-Usage 70%`，说明什么？可能的原因有哪些？

<details>
<summary>参考答案</summary>

说明设备内存被占了但计算没在跑。可能原因：有进程占着内存没释放（训练中断后残留）、模型已加载但没在推理/训练、或者程序在做 CPU 侧的数据加载 (data loading)，NPU 在空等。GPU 机器上对应 `GPU-Util 0%` + `Memory-Usage 70%` 的同款现象。
</details>

**第 3 题** ⭐⭐：`torch.npu.is_available()` 返回 `False`，你的排查步骤是什么？（写出至少 3 步）

<details>
<summary>参考答案</summary>

1. `npu-smi info` 确认机器有 NPU 且驱动正常；
2. 确认装了 torch_npu：`pip list | grep torch`，没有则按 CANN/PyTorch 版本对照表安装匹配版本；
3. 确认 CANN 工具包、驱动、固件三者版本匹配（版本不匹配是最常见的坑）。
</details>

**第 4 题** ⭐⭐：说出 CANN、HCCL、AscendCL 分别是什么？GPU 生态里各对应什么？

<details>
<summary>参考答案</summary>

CANN（华为昇腾计算架构）是昇腾的完整软件栈，相当于 CUDA；HCCL 是昇腾的集合通信库，用于多卡/多机通信，相当于 NCCL；AscendCL 是昇腾的底层编程接口（API），用于调用算子和管理设备，相当于 CUDA Runtime/Driver API。简单记：CANN 是总称，AscendCL 是编程入口，HCCL 管通信。
</details>

**第 5 题** ⭐：昇腾 910B 和 310P 的定位分别是什么？选型口诀是什么？

<details>
<summary>参考答案</summary>

910B 是训练主力（大模型预训练/微调，64GB HBM），310P 是推理主力（视频分析、OCR、边缘推理）。选型口诀：训练看 910B，推理看 310P；两者软件栈一致（CANN），代码可平移。
</details>

## 参考链接

- [昇腾 CANN 文档中心](https://www.hiascend.com/document) — 安装、调优、故障排查
- [Ascend PyTorch Adapter (torch_npu)](https://gitee.com/ascend/pytorch) — 昇腾 PyTorch 适配层
- [ml-engineering — compute/accelerator/nvidia/debug.md](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)（GPU 参考）
- [PyTorch CUDA 官方文档](https://pytorch.org/docs/stable/notes/cuda.html)（GPU 参考）
- [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface)（GPU 参考）

---

**下一章**：[第2章 算子层](../02-算子层/README.md) | **返回目录**：[教程首页](../README.md)
