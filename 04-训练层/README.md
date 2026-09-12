---
title: 第4章 训练层——单卡放不下的模型，怎么多卡训练
tags: [ai-infra, 教程, 训练, deepspeed, megatron, zeRO, 昇腾, torch_npu, hccl]
章节: 04
状态: 完成
---

# 第4章 训练层：单卡放不下的模型，怎么多卡训练

> 难度：⭐⭐ 进阶
> 一句话版：100B 模型单卡放不下，把"数据"或"模型"切开分到多张卡上并行训练，这就是训练层的核心问题。

> 导航：[← 第3章 框架层](../03-框架层/README.md) | **第4章 训练层** | [第5章 推理层 →](../05-推理层/README.md)

---

## 🥢 类比开场：多个厨师同时做菜

想象一个餐厅同时接待 100 桌客人：

- **数据并行** = 菜谱复印 100 份，100 个厨师各做各的桌，出锅前互相核对口味（同步梯度）。菜谱都一样，但菜量 ×100
- **张量并行** = 一道复杂的菜，A 切葱、B 切肉、C 备酱，几个人同时备一道菜。一张卡干不动的活，几张卡一起干
- **流水线并行** = 切菜 → 炒菜 → 装盘的流水线，每个师傅守一段，前一段干完传给下一段。模型按层分工，逐层传递

本章要解决的核心问题：**模型太大、显存放不下，怎么把训练拆到多张卡上，并且拆完之后不亏算力、不爆显存。**

---

## 1. 问题：单张卡放不下 100B 模型

先算一笔账（以 bf16 精度为例，每个参数占 2 字节）：

- 100B 参数 × 2 字节 = **200GB**，光权重就超过一张 H100（80GB）的显存
- 训练还要额外存：梯度（+200GB）、优化器状态（Adam 的两个动量，+400GB）、激活值（+数十 GB）
- 结论：100B 模型单卡训练总共需要约 **1TB 显存**，而单卡上限 80GB，差了一个数量级

所以必须**并行**：把训练工作拆到多张卡（甚至多台机器）上，让每张卡只承担一部分。

---

## 2. 显存计算详解：训练时显存都花在哪了 ⭐⭐

排障时经常要回答一个问题："这个模型到底能不能塞进这张卡？"答案来自下面这个公式。

### 2.1 四项显存构成（公式版）

设模型参数量为 **Ψ**（如 7B），精度为 bf16（2 字节/参数），优化器为 Adam：

| 占显存的东西 | 每参数字节数 | 公式 | 说明 |
|---|---|---|---|
| 模型参数（bf16） | 2B | **2Ψ** | 权重本身 |
| 梯度（bf16） | 2B | **2Ψ** | 反向传播产出，形状和参数一一对应 |
| 优化器状态（Adam） | 12B | **12Ψ** | 见下方拆解 |
| 激活值（activations） | 动态 | **≈ 2·b·s·h·L**（字节） | 前向传播的中间结果，随 batch/序列长变化 |

**Adam 的 12 字节拆解**（这也是显存大头）：

- fp32 主权重副本（混合精度训练必须）：4Ψ
- 一阶动量 m（fp32）：4Ψ
- 二阶动量 v（fp32）：4Ψ

所以训练一个模型**不含激活值**的总显存：

> **总显存 ≈ 2Ψ + 2Ψ + 12Ψ = 16Ψ 字节**

### 2.2 代入实例：7B 模型

| 项目 | 计算 | 显存 |
|---|---|---|
| 参数 + 梯度 | 2×7B×2B + 2×7B×2B | 28 GB |
| 优化器状态 | 12 × 7B 字节 | 84 GB |
| 激活值（b=1, s=2048, h=4096, L=32） | ≈ 2×1×2048×4096×32 B | ≈ 0.5 GB（开激活检查点后） |
| **合计** | | **≈ 113 GB** |

> 所以 7B 模型 bf16 全参训练要 4×80GB（或 2×A100 80G 勉强）——这也解释了为什么大家都在用 LoRA：**LoRA 只训练新增的少量低秩参数，彻底绕开 12Ψ 的优化器状态**（见第 8 节）。

### 2.3 激活值为什么"可大可小"

激活值是前向传播每层的中间结果，训练时**必须全部保留到反向传播**。它和三个量成正比：

- 微批大小 b：**线性**增长
- 序列长度 s：注意力分数矩阵是 **s² 增长**（这就是长上下文训练显存爆炸的原因）
- 层数 L × 隐藏维度 h：线性增长

**对抗手段**（按性价比排序）：
1. **梯度检查点（activation checkpointing）**：不存中间结果，反向时重算一遍，显存降 ~70%，代价是 ~30% 算力
2. 减小 micro-batch / 序列长度
3. 梯度累积（时间换显存，见第 6 节）

---

## 3. 方案：三大并行 ⭐⭐

### 3.1 数据并行（Data Parallelism, DP）

- 每张卡复制一份**完整模型**，各吃一批不同的数据（数据并行，data parallelism）
- 每步训练后同步梯度（AllReduce 通信），保证所有副本参数保持一致
- 适合：模型单卡放得下、数据量大的场景
- 代价：每步要同步全部梯度，通信量随模型变大线性增长

### 3.2 张量并行（Tensor Parallelism, TP）

- 把单个权重矩阵切成 N 块分到 N 张卡，一张卡只算一块（张量并行，tensor parallelism）
- 每层计算都要卡间通信（AllReduce），通信频繁、延迟敏感
- 适合：单层都放不下的超大模型；通常只在单机内用（NVLink 高速互联；昇腾 NPU 上对应 HCCS 卡间高速互联）

### 3.3 流水线并行（Pipeline Parallelism, PP）

- 按层把模型切成 N 段：卡 1 算前几层、卡 2 算中间、卡 3 算最后（流水线并行，pipeline parallelism）
- 卡间串行传递中间激活值，像工厂流水线一样分工
- 缺点：流水线有空转的"气泡"（bubble），利用率打折，要用 micro-batch（微批次）填充

### 3.4 三种并行详细对比

| 维度 | 数据并行 DP | 张量并行 TP | 流水线并行 PP |
|---|---|---|---|
| 拆什么 | 数据（每卡一份完整模型） | 单个权重矩阵切 N 块 | 模型按层切 N 段 |
| 每卡存 | 完整模型（不省显存） | 1/N 权重 | 1/N 层 |
| **通信量** | 每步全量梯度（2Ψ/卡，随模型线性增长） | 每层 2 次 AllReduce，**通信量最大、最频繁** | 每段只传激活值，总量小 |
| 通信频率 | 每步 1 次（可延迟） | 每层 2 次（无法延迟） | 每 micro-batch 段边界 1 次 |
| 通信带宽要求 | 中（千兆以太网可跑） | **极高，必须 NVLink/IB（昇腾 HCCS/RoCE）** | 低（以太网可跑） |
| 延迟敏感度 | 低 | **极高**（延迟直接卡计算） | 中 |
| 适用规模 | 千卡级横向扩展 | 单机 8 卡（跨机带宽不够） | 数十段 |
| 硬件要求 | 普通服务器 + 高速网卡 | 单机多卡 + NVLink（昇腾 HCCS） | 多机多卡 + 以太网即可 |
| 主要代价 | 显存不省、通信随模型涨 | 通信瓶颈、实现复杂 | 流水线气泡（bubble） |

**选型口诀**：
- 卡少模型大 → TP（先切权重）
- 卡多模型适中 → DP（横向堆卡）
- 模型超深 → PP（按层切）
- 实际几乎都用**混合并行**：TP × PP × DP 三维组合（也叫 3D 并行），再配 ZeRO 压显存（下一节）

---

## 4. ZeRO：不增加卡，也能把显存压下来 ⭐⭐

### 4.1 思路：DP 的显存浪费

数据并行每张卡都存**完整的** 16Ψ 字节（参数+梯度+优化器状态），N 张卡就是 N 份冗余。ZeRO（Zero Redundancy Optimizer）的想法：**这些冗余没必要，切开分着存，用时再聚合**。

以 DP 度 P=8、7B 模型为例（不含激活值，每卡 112GB → 目标每卡 14GB）：

### 4.2 Stage 1：切优化器状态

```
Stage 1 前（每卡存全部）              Stage 1 后（每卡只存 1/8）
┌─────────────────────┐              ┌─────────────────────┐
│ 参数 2Ψ  梯度 2Ψ      │              │ 参数 2Ψ  梯度 2Ψ      │
│ 优化器 12Ψ（m,v,fp32）│              │ 优化器 12Ψ/P=1.5Ψ    │← 只留自己的 1/8
└─────────────────────┘              └─────────────────────┘
每卡：16Ψ = 112 GB                    每卡：4Ψ + 12Ψ/P = 38.5 GB（省 ~66%）
```

**原理**：优化器状态（m、v、fp32 主权重）按 DP rank 切分，每卡只更新自己那份；更新前用 AllGather 把完整参数广播给所有卡做前向/反向，反向完用 ReduceScatter 把梯度归约回对应卡。**通信量不变**（还是每步全量梯度），只是省显存。

### 4.3 Stage 2：再切梯度

```
Stage 2：梯度也不全存了，反向传播时边算边 ReduceScatter，
          每卡只保留自己负责的那 1/8 梯度
每卡：2Ψ + 2Ψ/P + 12Ψ/P = 14 + 1.75 + 10.5 = 26.25 GB
```

**原理**：反向传播的梯度是逐层产生的，算出后立刻按分片归约，只保留自己需要的那份，用不上的直接丢弃。显存从 16Ψ 降到 **2Ψ + 14Ψ/P**。

### 4.4 Stage 3：连参数也切

```
Stage 3：参数也不再每卡放完整副本，权重本身按层切开，
          前向/反向用到哪层，就 AllGather 哪层的权重
每卡：16Ψ/P = 14 GB（P=8 时，理论极限）
```

**原理**：这是 ZeRO 与张量并行的本质区别——**ZeRO 切了但不并行算**（每层计算仍然只有一张卡在做，其余卡只是不存），TP 是切了同时并行算。所以 ZeRO 省显存不增加算力，但通信量略涨（AllGather 权重）。

### 4.5 三档对比（7B 模型，DP=8，不含激活值）

| 配置 | 每卡显存公式 | 7B 实例 | 通信 | 适用 |
|---|---|---|---|---|
| 普通 DP | 16Ψ | 112 GB | 每步全量梯度 | 小模型 |
| ZeRO-1 | 4Ψ + 12Ψ/P | 38.5 GB | 不变 | 优化器状态是大头的场景 |
| ZeRO-2 | 2Ψ + 2Ψ/P + 12Ψ/P | 26.25 GB | 不变 | **微调/中小模型默认选择** |
| ZeRO-3 | 16Ψ/P | 14 GB | +权重 AllGather | 单卡放不下权重的大模型 |

> 记忆点：**Stage 数字 = 连切几样东西**。1 只切优化器状态，2 加切梯度，3 加切权重。Stage 3 通信开销最大，能用 Stage 2 就别用 3。

### 4.6 再进一步：Offload

- **offload_optimizer=cpu**：优化器状态（12Ψ）挪到 CPU 内存，显存几乎只剩参数+梯度 → 能跑更大模型，但每步更新参数要走 PCIe，**速度明显变慢**
- **offload_param=cpu**：连权重也放 CPU，用时再搬上 GPU → 极端省显存，极度变慢
- **NVMe offload**：把 CPU 都放不下的状态写到 NVMe 硬盘，最慢，只用于"卡太小又非要训"的兜底

---

## 5. 项目速览

| 项目 | Stars | 定位 | 亮点 |
|---|---|---|---|
| DeepSpeed | ⭐35k | 微软，训练优化框架 | ZeRO 显存优化、CPU Offload、接入简单 |
| Megatron-LM | ⭐17.6k | NVIDIA，大规模训练框架 | TP/PP 工程化最成熟，预训练主力 |
| LLaMA-Factory | ⭐40k | 开源微调全家桶 | 开箱即用，LoRA/全参微调 LLaMA 系模型 |
| ColossalAI | ⭐41k | 潞晨科技，一体化并行框架 | 一键式并行策略、ColossalGPT 大模型训练 |

**选型建议**：小集群微调/上手练手用 LLaMA-Factory；百亿级预训练用 Megatron-LM + DeepSpeed 组合（Megatron 管并行，DeepSpeed 管显存）。

**昇腾（NPU）生态对应物**：GPU 侧的每个角色在昇腾上都有等价方案，一一对应：

| 昇腾方案 | 对应 GPU 侧 | 定位 |
|---|---|---|
| **ModelLink**（华为官方，Gitee） | Megatron-Core | 昇腾大模型训练框架：TP/PP/DP/CP/EP 全并行，预训练/微调一条龙 |
| **MindSpeed**（华为官方） | DeepSpeed 加速组件 | 昇腾训练加速库：显存优化（ZeRO）、FlashAttention、重计算，与 ModelLink 配套使用 |
| **AscendSpeed**（昇腾社区） | Megatron-DeepSpeed | 社区版训练框架，Megatron-DeepSpeed 的昇腾适配，含 ZeRO-Offload |

> 昇腾选型口诀：预训练走 **ModelLink + MindSpeed**（对应 GPU 侧 Megatron + DeepSpeed 组合）；PyTorch 微调生态（LLaMA-Factory、DeepSpeed 原生版）通过 torch_npu 直接跑在 NPU 上，见 6.4。

---

## 6. 代码：DeepSpeed 完整配置示例 ⭐⭐

DeepSpeed 用 JSON 配置声明训练策略。核心公式：

> **train_batch_size（全局批大小）= train_micro_batch_size_per_gpu × gradient_accumulation_steps × world_size（卡数）**

### 6.1 完整配置（ZeRO-2 + CPU Offload + 混合精度）

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 2,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 3e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1 }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": 3e-5, "warmup_num_steps": 500 }
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 6.2 关键参数解读

| 参数 | 含义 | 坑 |
|---|---|---|
| `train_micro_batch_size_per_gpu` | 每卡一次前向+反向的批大小 | 显存不够时调小它 |
| `gradient_accumulation_steps` | 累积多少步再更新一次参数 | 模拟大 batch 的"时间换显存"，数值=全局batch÷(micro×卡数) |
| `zero_optimization.stage` | ZeRO 等级 1/2/3（见第 4 节） | 3 通信大，能 2 别 3 |
| `offload_optimizer` | 优化器状态放 CPU | 省显存但变慢，小卡兜底用 |
| `bf16.enabled` | 半精度训练 | bf16 动态范围和 fp32 一致，比 fp16 稳，**新卡优先 bf16** |
| `overlap_comm` | 通信与计算重叠 | 开着，白赚速度 |
| `gradient_clipping` | 梯度裁剪上限 | **NaN/发散的第一道防线**，见第 9 节 |
| `scheduler` | 学习率调度 | 训练初期必须 warmup（见第 9 节） |

**如果想用 fp16 而非 bf16**（老卡不支持 bf16 时），把 `"bf16"` 换成：

```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "min_loss_scale": 1,
  "initial_scale_power": 16
}
```

fp16 需要动态 loss scaling（梯度下溢保护），bf16 不需要——这就是 bf16 更省心的原因。

### 6.3 启动方式

```bash
# deepspeed 作为 torchrun 的启动器，8 卡单机
deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json

# 多机（2 台 × 8 卡）：需要 deepspeed 的 hostfile
deepspeed --hostfile hostfile train.py --deepspeed_config ds_config.json
```

### 6.4 昇腾（NPU）环境适配：torch_npu + HCCL

DeepSpeed 的 ZeRO 逻辑与硬件无关，昇腾上跑只需换**执行后端**：PyTorch 侧用 **torch_npu**（昇腾的 PyTorch 适配层，设备名 `npu`）接管算子，多卡通信后端从 NCCL 换成 **HCCL**（华为集合通信库，对应 NVIDIA 的 NCCL），JSON 配置一行不用改：

```bash
# 安装适配层（版本必须与 torch、CANN 严格对应）
pip install torch_npu

# 指定可见 NPU（对应 CUDA_VISIBLE_DEVICES）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# deepspeed 启动：--num_gpus 填 NPU 卡数，通信自动走 HCCL
deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json
```

```python
# 代码里只需把 device 从 cuda 换成 npu
import torch
import torch_npu        # 必须先 import，注册 npu 后端
model = model.to("npu")
```

| GPU 侧 | 昇腾侧 | 说明 |
|---|---|---|
| `torch.cuda` / `device="cuda"` | `torch_npu` / `device="npu"` | 昇腾 PyTorch 适配层，API 几乎一一对应 |
| NCCL | **HCCL** | 昇腾集合通信库，torch_npu 自动选择，无需手写 |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 指定可见设备 |
| `nvidia-smi` | `npu-smi info` | 查看 NPU 状态/显存 |

> bf16 在昇腾上同样推荐（910B 原生支持）；若初始化就报通信错误，先查 torch_npu 与 CANN 版本是否配套（见 9.4）。

---

## 7. 代码：Megatron-LM 张量并行示例 ⭐⭐

### 7.1 张量并行的本质（30 秒看懂）

一个线性层 `Y = X·W`（W 是 h×h 矩阵），切成 TP=2：

```
列并行（ColumnParallel）：W 竖着切 → 每卡算 Y 的一部分 → 拼接
行并行（RowParallel）：   W 横着切 → 每卡用完整 Y_part 算 → AllReduce 求和
前向：X → [ColumnParallel(W1)  ‖  ColumnParallel(W2)] → Y 拼接
反向：梯度 → AllReduce → 每卡拿到完整 dW
```

关键点：**每算完一层就要一次 AllReduce**（行并行那步），这就是 TP 通信频繁、必须 NVLink 的原因。

### 7.2 Megatron-LM 实际用法（框架视角）

Megatron-LM 通过命令行参数声明并行，无需手写通信：

```bash
# TP=8 单机训练（GPT-3 风格）
python pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 1 \
  --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
  --micro-batch-size 2 --global-batch-size 512 \
  --bf16 --lr 3e-4 --lr-warmup-iters 500 \
  --train-iters 100000 --save-interval 1000 \
  --data-path /data/mydata --save /checkpoints --load /checkpoints
```

关键参数：

- `--tensor-model-parallel-size`：TP 切分数（必须 ≤ 单机卡数，跨机 NVLink 不够快）
- `--pipeline-model-parallel-size`：PP 段数（多机时用）
- 两者相乘 = 总卡数：如 TP=4 × PP=4 = 16 卡

### 7.3 手写一个最小张量并行（理解原理）

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化（假设 TP=2，rank 0/1 各持一半权重）
dist.init_process_group("nccl")
tp_rank, tp_size = dist.get_rank(), dist.get_world_size()

torch.manual_seed(0)
H = 4096
# 每卡只持有 W 的一半列（列并行）
W_shard = torch.randn(H, H // tp_size, device="cuda", dtype=torch.bfloat16)

def column_parallel_forward(x):
    # 每卡用自己那 1/tp_size 列算部分输出
    y_shard = x @ W_shard                 # (b, s, H/tp_size)
    # 把各卡的部分输出拼起来，得到完整 Y —— 这就是 AllGather
    y_full = [torch.empty_like(y_shard) for _ in range(tp_size)]
    dist.all_gather(y_full, y_shard)
    return torch.cat(y_full, dim=-1)      # (b, s, H)

x = torch.randn(2, 128, H, device="cuda", dtype=torch.bfloat16)
y = column_parallel_forward(x)
print("TP 输出形状:", tuple(y.shape), "| 每卡显存:", W_shard.numel() * 2 / 1e9, "GB")
```

> 这只是一个 `all_gather` 就能讲清 TP 的原理：**权重切开、输出聚合**。真实 Megatron 里还有行并行 + 反向的 AllReduce，但骨架就是上面这几行。完整示例可运行版本见第 10 节动手实验。

### 7.4 昇腾示例：torch_npu 分布式训练

把 7.3 的代码搬到昇腾 NPU，只改三处：`import torch_npu`、设备换 `npu`、通信后端换 `hccl`：

```python
import torch
import torch_npu                          # ① 注册 npu 后端（先于一切 cuda 调用）
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# ② 通信后端：nccl → hccl（HCCL 集合通信库）
dist.init_process_group("hccl")
rank, world_size = dist.get_rank(), dist.get_world_size()

torch.manual_seed(0)
# ③ 设备：cuda → npu
model = torch.nn.Linear(4096, 4096).to("npu").to(torch.bfloat16)
ddp = DistributedDataParallel(model, device_ids=[rank])

optimizer = torch.optim.AdamW(ddp.parameters(), lr=3e-5)
for step in range(100):
    x = torch.randn(2, 128, 4096, device="npu", dtype=torch.bfloat16)
    loss = ddp(x).float().square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0 and rank == 0:
        print(f"step {step} | loss {loss.item():.4f}")
```

```bash
# 启动：torchrun 照常，卡号用 ASCEND_RT_VISIBLE_DEVICES 控制
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_npu.py
```

> DDP 本身不关心底层是 GPU 还是 NPU——换后端、换 device，同一套数据并行代码就跑在昇腾上。这正是第 3 章说的"框架层写一次、多硬件跑"。

---

## 8. 代码：LLaMA-Factory 微调完整示例（LoRA）⭐⭐

### 8.1 LoRA 一句话

冻结原模型权重（0 梯度），只训练注入的**低秩矩阵**（如 rank=8 的 A·B 两个小矩阵）。可训练参数从 7B 降到 ~0.1%，优化器状态从 84GB 降到 ~0.1GB——**这就是为什么 24GB 消费级卡也能微调 7B 模型**。

### 8.2 数据集格式（每条样本一个 JSON 对象）

```json
[
  {
    "instruction": "解释什么是反向传播",
    "input": "",
    "output": "反向传播是神经网络训练时，从输出层向输入层逐层计算梯度的方法……"
  },
  {
    "instruction": "翻译成英文：你好，世界",
    "input": "",
    "output": "Hello, world."
  }
]
```

### 8.3 LoRA 配置（YAML，`examples/train_lora/llama3_lora_sft.yaml`）

```yaml
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct   # 基座模型
dataset: alpaca_zh, self_cognition                    # 数据集（data/dataset_info.json 里注册）
template: llama3
finetuning_type: lora                                 # LoRA 微调
lora_rank: 8                                          # 低秩矩阵的秩：越大越强越费显存
lora_alpha: 16                                        # 缩放系数，一般取 rank 的 2 倍
lora_target: all                                      # 注入哪些层：all = 所有线性层
output_dir: saves/llama3-8b/lora/sft                  # 输出目录
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1                                     # 10% 步数做 warmup
bf16: true
optim: adamw_torch
max_length: 1024
logging_steps: 10
save_steps: 100
```

### 8.4 训练与推理命令

```bash
# 单卡训练（消费级显卡也能跑）
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# 多卡训练：LLaMA-Factory 底层是 transformers，直接 --multi_gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
  examples/train_lora/llama3_lora_sft.yaml \
  --deepspeed ds_config.json

# 合并 LoRA 权重回基座模型（部署推理前必须做）
llamafactory-cli export \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --adapter_name_or_path saves/llama3-8b/lora/sft \
  --template llama3 --finetuning_type lora \
  --export_dir saves/llama3-8b/lora/merged \
  --export_size 4 --export_legacy_format false
```

> **LLaMA-Factory 训练过程完整记录**（含数据集制作、训练曲线、推理验证）：见本教程配套笔记 `LLaMA-Factory 微调实战`。

---

## 9. 排障参考：训练不稳定详解 ⭐⭐

训练层故障分两大类，对应 ml-engineering 仓库的两个目录：

**容错（`training/fault-tolerance/`）—— 机器层面**：节点掉线、显存 OOM、机器重启。关键实践：定期存 checkpoint、异步保存、心跳检测、断点续训。记住：**训练中断不可怕，可怕的是没有 checkpoint。**

**训练不稳定（`training/instabilities/`）—— loss 层面**：下面重点讲。

### 9.1 Loss 曲线形状诊断

```
正常收敛                  发散                    尖峰后恢复
loss │╲                    loss │╱╲                 loss │╲ ╱
     │ ╲                        │  ╲╱╲                   │ ╲╱
     │  ╲                       │     ╲                  │  ╲
     └──────→ step              └──────→ step            └──────→ step
     ↓ 单调下降，斜率渐缓        ↓ 不降反升或剧烈震荡       ↓ 单步尖峰后回到原轨迹
     ✅ 正常                    ❌ 发散                   ⚠️ 单点异常，观察即可
```

| 曲线形状 | 含义 | 首要怀疑对象 | 处理 |
|---|---|---|---|
| 稳步下降 | 正常 | — | 不用动 |
| 一路发散（loss ↑） | 学习率过高 / 数据异常 | ① lr ② 数据 pipeline 混入脏样本 | 砍 lr（÷10）、检查数据 |
| 某个 step 尖峰后恢复 | 单点数据异常或 NaN 样本 | 数据批次 | 定位该 batch 清洗数据；若频繁出现，降 lr |
| 变 NaN 后不恢复 | **梯度爆炸 / 精度下溢** | ① gradient_clipping ② 混合精度 | 开裁剪=1.0、换 bf16、检查 loss scale |
| 长期平台期（loss 不变） | 学习率过低 / 模型没在学 | ① lr ② 梯度是否为 0 | 看梯度范数（见下）、提 lr、检查是否有冻结层误伤 |
| 震荡不收敛（周期性波动） | 全局 batch 太小，噪声大 | batch size / lr | 增大梯度累积、降 lr |

### 9.2 排查顺序（三步定位法）

1. **先看 loss 曲线形状** → 判断是哪一类（上表）
2. **再看梯度范数**（比 loss 更早暴露问题）：

```python
# 每 N 步打印一次全局梯度范数
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.detach().norm(2).item() ** 2
total_norm = total_norm ** 0.5
if step % 10 == 0:
    print(f"step {step} | loss {loss.item():.4f} | grad_norm {total_norm:.3f}")
```

   - 梯度范数持续 > 10 且增长 → 梯度爆炸前兆，开 `gradient_clipping`
   - 梯度范数突然变 NaN → 混合精度问题（loss scale 下溢），换 bf16 或调 loss scale
   - 梯度范数恒为 0 → 模型没在学：lr 太低、或参数被意外冻结

3. **最后定位到具体层**：用 `torch.autograd.detect_anomaly()` 找到第一个产生 NaN 的层，八成是 embedding 或 attention 的 softmax 附近。

### 9.3 高频故障速查表

| 症状 | 根因 | 解法 |
|---|---|---|
| loss → NaN | lr 过高 / 梯度爆炸 / fp16 下溢 | gradient_clipping=1.0，换 bf16 |
| 训练 5 分钟就 OOM | 激活值撑爆 / 别进程占显存 | 开激活检查点、减 micro-batch、`nvidia-smi`/`npu-smi` 清场 |
| 显存够但跑不起来 | 数据加载瓶颈（GPU 空转） | `num_workers` 调大、pin_memory=True |
| 多卡 loss 比单卡高 | 全局 batch 变大但 lr 没调 | 按 batch 比例放大 lr（linear scaling rule） |
| 重启后 loss 跳变 | checkpoint 加载不完整 / 数据顺序变了 | 核对 checkpoint 与 tokenizer、固定数据 shuffle 种子 |

> 📖 权威排障手册：https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md

### 9.4 HCCL 排障（昇腾多卡通信）

昇腾多卡训练的大头故障集中在 HCCL（对应 NCCL）通信上，症状与解法一一对应：

| 症状 | 根因 | 解法 |
|---|---|---|
| 初始化报 `hccl` backend 不存在 | 没装 / 没 import torch_npu，或版本不匹配 | `pip install torch_npu`，版本与 torch、CANN 严格配套 |
| 多机 rank 卡在初始化、超时 | 通信端口不通 / 防火墙拦截 | 放行 HCCL 通信端口，调大 `HCCL_CONNECT_TIMEOUT` |
| 通信速度远低于预期 | 网络拓扑没识别 / 走了以太网回退 | 单机内确认 HCCS 高速互联；多机确认 RoCE/IB 网卡 |
| 训练中断报设备错误 | NPU 驱动 / CANN 版本问题 | `npu-smi info` 检查卡状态，升级 CANN 到配套版本 |
| 部分卡 OOM，其余正常 | 分片不均 / 显存碎片 | 调小 micro-batch，`npu-smi info` 对比各卡占用 |

**排查顺序**（对应 9.2 的三步定位法，把"卡"换成"机"）：
1. 先跑通**单机单卡**——排除算子、显存问题
2. 再测**单机多卡**——暴露 HCCL 拓扑、驱动问题
3. 最后上**多机**——暴露网络问题。每上升一层，故障面减半

---

## 10. 动手实验 ⭐

按顺序做，每步都有明确产出：

### 实验 1：估算任意模型的显存（5 分钟，无需 GPU）

```python
# estimate_vram.py —— 用第 2 节的公式估算训练显存
def estimate_vram(num_params: float, dp: int = 1, bytes_per_param: int = 2):
    psi = num_params * 1e9
    total = 16 * psi                       # 参数+梯度+Adam（12Ψ）
    per_gpu = total / dp
    print(f"参数 {num_params:.1f}B | DP={dp}")
    print(f"  单卡朴素训练: {per_gpu/1e9:.1f} GB")
    print(f"  ZeRO-2 (P={dp}): {(2*psi + 2*psi/dp + 12*psi/dp)/1e9:.1f} GB")
    print(f"  LoRA (≈0.1% 可训): {2*psi/1e9:.1f} GB + 优化器≈0")

estimate_vram(7)    # 7B
estimate_vram(70)   # 70B
```

**产出**：能回答"这张卡能不能训这个模型、该用 ZeRO 还是 LoRA"。

### 实验 2：DeepSpeed 显存实测对比（需 1 张 GPU）

```bash
pip install deepspeed
# 跑一遍不带 ZeRO 的基线，再给 train.py 加上 --deepspeed_config ds_config.json
# （第 6 节的配置），对比 nvidia-smi 的显存占用
nvidia-smi --query-gpu=memory.used --format=csv   # 昇腾环境：npu-smi info
```

**产出**：亲眼看 ZeRO-2 把显存压下来多少。

### 实验 3：LLaMA-Factory LoRA 微调全流程（需 1 张 ≥16GB 显卡）

```bash
pip install llama-factory[torch]
llamafactory-cli create --model_name Qwen2.5-7B-Instruct --dataset alpaca_zh
# 用第 8.3 节的 YAML，把模型换成 Qwen2.5-7B-Instruct
llamafactory-cli train examples/train_lora/qwen_lora_sft.yaml
# 训练完看 loss 曲线：output_dir 里的 training_loss.png
llamafactory-cli chat --model_name_or_path saves/qwen/lora/sft --template qwen
```

**产出**：一份真实跑通的 LoRA 微调 + 一张可分析的 loss 曲线图（对应第 9 节诊断表）。

> 若本地无 GPU：实验 1 纯 CPU 可跑，实验 2/3 可看配套笔记 `LLaMA-Factory 微调实战` 的记录。昇腾 NPU 环境：实验 2/3 用 `ASCEND_RT_VISIBLE_DEVICES` 指定卡即可，命令不变。

---

## 11. 自测题

**第 1 题** ⭐：一张 H100（80GB）训练 100B 模型，bf16 下光权重就要 200GB。三大并行中，哪种能让每张卡只存 1/N 的权重？它和数据并行最本质的区别是什么？

<details>
<summary>参考答案</summary>

**张量并行**（流水线并行按层切也能少存，但张量并行切的是权重矩阵）。数据并行每张卡存完整模型副本，张量并行把权重切块、每卡只算一块。
</details>

**第 2 题** ⭐⭐：DeepSpeed 公式：train_batch_size = micro_batch × gradient_accumulation_steps × 卡数。8 卡、micro-batch=2、目标全局 batch=32，gradient_accumulation_steps 应设多少？

<details>
<summary>参考答案</summary>

32 ÷ (2 × 8) = **2**。
</details>

**第 3 题** ⭐⭐：训练 7B 模型（bf16, Adam），不含激活值需要多少显存？ZeRO-3 用 8 卡后每卡降到多少？写出公式。

<details>
<summary>参考答案</summary>

完整训练 = 2Ψ（参数）+ 2Ψ（梯度）+ 12Ψ（Adam 优化器状态）= **16Ψ = 112 GB**。ZeRO-3 把三类全部按 8 切分：**16Ψ/8 = 14 GB/卡**（不含激活值）。
</details>

**第 4 题** ⭐⭐：训练到第 1000 步，loss 突然变成 NaN。按第 9 节的排障思路，你最优先检查哪两项配置？

<details>
<summary>参考答案</summary>

**学习率**（是否过高或出现尖峰导致发散）和 **gradient_clipping / 混合精度**（是否梯度爆炸或 loss 缩放异常）。正确顺序：先看 loss 曲线形状 → 再看梯度范数 → 最后定位具体层。
</details>

**第 5 题** ⭐⭐：ZeRO Stage 2 和 Stage 3 都"省显存"，为什么大模型预训练首选 Stage 3，而微调默认 Stage 2？

<details>
<summary>参考答案</summary>

Stage 3 连权重都切分（每卡 16Ψ/P），能跑单卡完全放不下的模型，代价是每层计算前要 AllGather 权重、通信量增加；Stage 2 保留完整权重副本、通信量不变，只省优化器状态和梯度，对"单卡放得下但优化器状态吃不下"的微调场景性价比最高。微调若用 Stage 3，反而因频繁 AllGather 拖慢速度。
</details>

**第 6 题** ⭐⭐：在昇腾 NPU 上做多卡数据并行，PyTorch 代码相比 GPU 需要改哪三处？通信后端叫什么？

<details>
<summary>参考答案</summary>

① `import torch_npu` 注册 npu 后端；② 设备从 `cuda` 换成 `npu`；③ 通信后端从 `nccl` 换成 `hccl`（**HCCL**，华为集合通信库，对应 NVIDIA 的 NCCL）。环境变量 `CUDA_VISIBLE_DEVICES` 换成 `ASCEND_RT_VISIBLE_DEVICES`。
</details>

**第 7 题** ⭐⭐：昇腾生态里 ModelLink、MindSpeed、AscendSpeed 分别对应 GPU 侧的哪个组件？昇腾上预训练的主力组合是什么？

<details>
<summary>参考答案</summary>

ModelLink ≈ Megatron-Core（并行训练框架），MindSpeed ≈ DeepSpeed 加速组件（ZeRO、FlashAttention），AscendSpeed ≈ Megatron-DeepSpeed（社区昇腾适配）。主力组合是 **ModelLink + MindSpeed**，对应 GPU 侧的 Megatron + DeepSpeed。
</details>

---

## 小结

- **问题**：训练显存 = 参数(2Ψ) + 梯度(2Ψ) + Adam 优化器状态(12Ψ) + 激活值 ≈ 16Ψ，7B 就要 112GB，单卡放不下。
- **方案**：三大并行——DP 拆数据（通信随模型涨）、TP 拆权重（通信最狠、必须 NVLink）、PP 拆层（有气泡）；实际用 3D 混合并行。
- **显存**：ZeRO 1/2/3 依次切优化器状态、梯度、权重，配合 CPU Offload 极限压显存；LoRA 从源头绕开 12Ψ。
- **工程**：DeepSpeed 管 ZeRO/Offload/混合精度，Megatron 管 TP/PP，LLaMA-Factory 管微调，三件套各司其职。
- **昇腾**：NPU 上三件套变成 ModelLink（并行）+ MindSpeed（加速）+ torch_npu/HCCL（执行与通信），代码只改 device 和后端，ZeRO 配置原样复用。
- **排障**：先看 loss 曲线形状 → 再查梯度范数 → 最后定位具体层；NaN 先查 lr 和 gradient_clipping。
- **下一步**：第 5 章进入推理层——训练完的模型怎么高效对外提供服务（vLLM、KV cache、量化）。

> 导航：[← 第3章 框架层](../03-框架层/README.md) | **第4章 训练层** | [第5章 推理层 →](../05-推理层/README.md)

## 参考链接

- [DeepSpeed ZeRO 论文（原理解读）](https://arxiv.org/abs/1910.02054)
- [stas00/ml-engineering — training/instabilities](https://github.com/stas00/ml-engineering/blob/master/training/instabilities/README.md)
- [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)
- [LLaMA-Factory 官方仓库](https://github.com/hiyouga/LLaMA-Factory)
- [ModelLink（昇腾大模型训练框架，Gitee）](https://gitee.com/ascend/ModelLink)
- [MindSpeed（昇腾训练加速库）](https://github.com/Ascend/MindSpeed)
- [AscendSpeed（昇腾社区训练框架）](https://github.com/Ascend/AscendSpeed)
- [torch_npu 官方仓库（昇腾 PyTorch 适配层）](https://gitee.com/ascend/pytorch)
- [昇腾 HCCL 集合通信文档](https://www.hiascend.com/document)

---

**上一章**：[第3章 框架层](../03-框架层/README.md) | **下一章**：[第5章 推理层](../05-推理层/README.md)
