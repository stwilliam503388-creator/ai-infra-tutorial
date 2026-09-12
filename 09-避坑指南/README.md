---
title: 第09章 避坑指南
tags: ['ai-infra', '教程', '排障', '调试']
章节: 09
状态: 完成
---

# 第9章：避坑指南

> 生产环境常见错误和排障方法

## 9.1 避坑总表

| 坑 | 现象 | 根因 | 解法 | 参考 |
|---|------|------|------|------|
| **显存不够** | CUDA out of memory | 模型太大/批量太大 | 量化、LoRA、换小模型 | ml-engineering `training/instabilities/` |
| **速度太慢** | 推理 10s 才出结果 | 没用 Continuous Batching | 换 vLLM | ml-engineering `inference/` |
| **多卡不工作** | 只有 1 卡在跑 | HCCL/NCCL 没配好 | 检查网络、环境变量 | ml-engineering `network/debug/` |
| **版本不匹配** | import 报错 | 依赖版本冲突 | 看 requirements.txt、用虚拟环境 | openai-cookbook AGENTS.md 规范 |
| **分布式挂起** | 训练卡住不动 | 通信死锁 | 用 `torch-distributed-npu-test.py` 诊断 | ml-engineering `debug/torch-distributed-gpu-test.py`（改用 hccl 后端） |
| **训练不稳定** | Loss 爆炸/NaN | 学习率/精度问题 | 梯度裁剪、混合精度 | ml-engineering `training/instabilities/training-loss-patterns.md` |
| **API 调用失败** | 401/403/404 | API Key 错误/过期/权限不足 | 检查环境变量、重新生成 Key | openai-cookbook 规范 |
| **模型下载慢** | HuggingFace 下载卡住 | 网络问题 | 用国内镜像（hf-mirror.com） | 国内网络环境 |
| **NPU 显存不够** | `RuntimeError: NPU out of memory` | 模型/批量太大、NPU 内存碎片 | 减小 batch、`npu-smi` 清场、碎片整理 | CANN 文档 out of memory 章节 |
| **HCCL 报错** | `HCCL error` / rank 卡死 | HCCL 配置错误、网络不通、连接超时 | 调大 `HCCL_CONNECT_TIMEOUT`、查网络、跑诊断脚本 | CANN HCCL 排障指南 |
| **CANN 版本不匹配** | import 报 `libascendcl.so` 找不到 | torch_npu 与 CANN 版本不配套 | 按 CANN 版本配套表安装匹配的 CANN + torch_npu | CANN 版本配套表 |

## 9.2 逐坑详解

### 坑 1：显存不够（CUDA/NPU out of memory）

**典型场景**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**根因分析**：
- batch size 太大，激活值撑爆显存
- 模型本身太大，单卡放不下
- 其他进程占着显存没释放

**解法（按优先级）**：
1. **清场**：`nvidia-smi`（昇腾：`npu-smi info -t process`）找到占用进程，`kill -9 <pid>`
2. **减小 batch size**：从 32 → 16 → 8，立竿见影
3. **梯度累积**：`gradient_accumulation_steps=4`，用时间换空间
4. **梯度检查点**：`model.gradient_checkpointing_enable()`
5. **量化**：FP16 → INT8 → INT4
6. **换小模型**：7B → 1.5B → 0.5B

**预防**：
- 训练前先用小批量测试
- 监控显存使用：`nvidia-smi -l 1`（昇腾：`npu-smi info -l 2`）
- NPU 特有：内存碎片也会触发 OOM，必要时重启进程或查 CANN 文档做碎片整理

### 坑 2：速度太慢

**典型场景**：
- 推理 10 秒才出结果
- 训练一步要 5 分钟

**根因分析**：
- 没用 Continuous Batching（推理）
- 数据加载是瓶颈（训练）
- 没用混合精度（训练/推理）

**解法**：
- 推理：换 vLLM/SGLang，开启 Continuous Batching
- 训练：检查 DataLoader 的 `num_workers`，用 `pin_memory=True`
- 通用：开启 FP16/BF16

### 坑 3：多卡不工作

**典型场景**：
- `torch.cuda.device_count()` 返回 1，但实际有 4 张卡
- 只有 GPU 0 在跑，其他空闲

**根因分析**：
- `CUDA_VISIBLE_DEVICES` 环境变量设置错误
- NCCL/HCCL 没安装或版本不对
- 网络配置问题（多节点）

**解法**：
```bash
# 检查可见设备
echo $CUDA_VISIBLE_DEVICES  # GPU，应该是 "0,1,2,3"
echo $ASCEND_RT_VISIBLE_DEVICES  # 昇腾 NPU

# 检查通信库
python -c "import torch; print(torch.cuda.nccl.version())"  # NCCL
python -c "import torch_npu; print(torch_npu.hccl.version())"  # HCCL

# 诊断脚本
torchrun --nproc_per_node=4 torch-distributed-npu-test.py
```

### 坑 4：版本不匹配

**典型场景**：
```
ImportError: libcudart.so.11.0: cannot open shared object file
```

**根因分析**：
- PyTorch 版本和 CUDA 版本不匹配
- 系统 CUDA 和 PyTorch 内置 CUDA 冲突

**解法**：
```bash
# 检查版本
nvidia-smi  # 看 CUDA Version（昇腾：npu-smi info 看固件/CANN 版本）
pip list | grep torch  # 看 PyTorch 版本和 CUDA 后缀
pip list | grep -i cann  # 昇腾：看 CANN/torch_npu 版本

# GPU：重装匹配版本
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 昇腾：按 CANN 版本配套表安装匹配的 torch_npu（版本号以配套表为准）
pip install torch_npu==<配套表版本> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 坑 5：分布式挂起

**典型场景**：
- 训练卡住不动，GPU 利用率 0%
- 日志停在 "Initializing process group"

**根因分析**：
- 网络不通（多节点）
- 防火墙阻挡
- 端口被占用

**解法**：
```bash
# 检查网络连通性
ping <master_addr>
nc -zv <master_addr> 29500

# 设置 NCCL 调试（GPU）
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# 设置 HCCL 调试（昇腾）
export HCCL_CONNECT_TIMEOUT=1800   # 卡在初始化时先调大
export ASCEND_GLOBAL_LOG_LEVEL=1   # 1=INFO，看 HCCL 日志
```

### 坑 6：训练不稳定

**典型场景**：
- Loss 突然变成 NaN
- Loss 不下降，一直在震荡

**根因分析**：
- 学习率太大
- 梯度爆炸
- 数据有问题（NaN/Inf）

**解法**：
```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 混合精度
# GPU 用 torch.cuda.amp；昇腾用 torch.npu.amp
from torch.cuda.amp import autocast, GradScaler  # 昇腾：from torch.npu.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 9.3 诊断工具箱

### torch-distributed-npu-test.py

```python
#!/usr/bin/env python
# 诊断多卡通信是否正常（GPU / 昇腾 NPU 通用）
# 用法：torchrun --nproc_per_node=2 torch-distributed-npu-test.py
# 昇腾机器先 source /usr/local/Ascend/ascend-toolkit/set_env.sh

import torch
import torch.distributed as dist

def main():
    # 自动选择后端：昇腾 NPU 用 hccl，GPU 用 nccl
    backend = 'hccl' if torch.npu.is_available() else 'nccl'

    # 初始化进程组
    dist.init_process_group(backend=backend)

    # 获取 rank 和 world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置设备
    if backend == 'hccl':
        torch.npu.set_device(rank)
        device = torch.device(f'npu:{rank}')
    else:
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    
    print(f"Rank {rank}/{world_size}: 初始化成功，设备 {device}")
    
    # 测试 AllReduce
    tensor = torch.ones(10, device=device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected = sum(range(world_size))
    actual = tensor[0].item()
    
    if actual == expected:
        print(f"✅ Rank {rank}: AllReduce 正确 ({actual})")
    else:
        print(f"❌ Rank {rank}: AllReduce 错误，期望 {expected}，实际 {actual}")
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### 使用方法

```bash
# 单节点 2 卡（昇腾机器先 source set_env.sh）
torchrun --nproc_per_node=2 torch-distributed-npu-test.py

# 多节点（2 节点，每节点 2 卡）
# 节点 0：
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=192.168.1.100 torch-distributed-npu-test.py
# 节点 1：
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=192.168.1.100 torch-distributed-npu-test.py
```

### GPU 监控脚本

```bash
#!/bin/bash
# gpu_monitor.sh - 实时监控 GPU 状态

while true; do
    clear
    echo "=== GPU 监控 $(date) ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    echo "=== 进程占用 ==="
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
    sleep 2
done
```

### NPU 监控与性能分析（昇腾）

```bash
# 实时监控 NPU（等效 nvidia-smi）
npu-smi info
npu-smi info -l 2        # 每 2 秒刷新
npu-smi info -t process -i 0   # 查看 0 号卡上的进程

# msprof：昇腾性能分析器（等效 ncu/nsys）
msprof --output=/tmp/msprof_out --application="python train.py"
msprof --export=on --output=/tmp/msprof_out   # 导出分析结果
```

> msprof 可输出算子耗时、AI Core 利用率、HCCL 通信耗时，用于定位"哪一步慢"、通信是否瓶颈。
```

## 9.4 排障流程（参考 ml-engineering）

```
1. 收集信息
   └── nvidia-smi（昇腾：npu-smi info）、dmesg、日志文件

2. 定位问题
   ├── 硬件问题？→ nvidia-smi / npu-smi 查看设备状态
   ├── 驱动问题？→ 检查 CUDA 版本
   ├── 网络问题？→ ping、nc、iperf
   └── 代码问题？→ 最小复现

3. 解决问题
   └── 参考对应章节的排障指南

4. 验证修复
   └── 重新运行，确认问题解决
```

## 9.5 动手实验

> 三个实验覆盖最常见的三类坑：显存、多卡通信、版本。建议在真实 GPU 机器上按顺序做，每个 10-15 分钟。

### 实验 1：制造并修复 CUDA OOM

**目标**：完整走一遍 OOM 从发生到修复的流程，学会用 `nvidia-smi` 和 Python 脚本诊断显存占用。

**步骤**：

1. 检查环境，记录总显存：
```bash
nvidia-smi  # 记住每张卡的 memory.total（昇腾：npu-smi info）
```

2. 制造 OOM（`oom_demo.py`，故意分配超出显存的张量）：
```python
import torch

# 昇腾 NPU：device='cuda' 改 device='npu'，get_device_properties 改 torch.npu.get_device_properties
total = torch.cuda.get_device_properties(0).total_memory
print(f"GPU 总显存: {total / 1024**3:.1f} GiB")

x = torch.ones(total // 2, dtype=torch.uint8, device='cuda')  # 占一半显存
y = torch.ones(total // 2, dtype=torch.uint8, device='cuda')  # 又占一半
z = torch.ones(total // 2, dtype=torch.uint8, device='cuda')  # 1.5 倍 → 必然爆
print(z.sum())
```

3. 运行并观察报错：
```bash
python oom_demo.py
```
你会看到 `RuntimeError: CUDA out of memory. Tried to allocate ...`，这是最经典的 OOM 报错。

4. 诊断：
```bash
# 看当前显存占用和残留进程
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
# 昇腾：npu-smi info -t process -i 0
```

5. 修复：
   - 有残留进程占显存 → `kill -9 <pid>` 清场
   - 代码层面：减小 batch size / 分配量；用完的变量及时 `del x; torch.cuda.empty_cache()`

**预期输出**：第 3 步看到带 `Tried to allocate` 的 OOM traceback；第 4 步 `nvidia-smi`（昇腾：`npu-smi`）显示显存被占满、能看到占用进程；修复后程序跑完并打印出 `z.sum()`。

**常见错误**：
- 以为 `torch.cuda.empty_cache()` 能释放正在使用的张量——它只释放缓存池，不释放活跃引用，`del` 才是关键
- 多个终端同时跑训练脚本互相抢显存，`nvidia-smi` 一眼就能看到谁在占

### 实验 2：诊断多卡通信问题

**目标**：用 `torch-distributed-npu-test.py` 验证进程组能否初始化、AllReduce 是否算对，复现"只有 1 卡在跑"。

**步骤**：

1. 拿到脚本：直接用 9.3 节的 `torch-distributed-npu-test.py` 代码存成文件（GPU/NPU 通用）。

2. 单卡先跑通基线：
```bash
torchrun --nproc_per_node=1 torch-distributed-npu-test.py
```

3. 多卡跑：
```bash
torchrun --nproc_per_node=2 torch-distributed-npu-test.py
```

4. 人为制造故障——只给 1 张卡却要 2 个进程：
```bash
# GPU：CUDA_VISIBLE_DEVICES=0；昇腾：ASCEND_RT_VISIBLE_DEVICES=0
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 torch-distributed-npu-test.py
```
这一步会报错，正是"多卡不工作"的典型原因。

5. 观察输出：每个 rank 打印初始化成功和 `✅ Rank N: AllReduce 正确`；如果 NCCL/HCCL 有问题，会卡在初始化或报 `RuntimeError: NCCL/HCCL error`。

**预期输出**：你会看到两个 rank 的日志：`Rank 0/2: 初始化成功`、`Rank 1/2: 初始化成功`，然后是 `✅ Rank 0: AllReduce 正确 (1)`、`✅ Rank 1: AllReduce 正确 (1)`（各卡 tensor=rank，AllReduce SUM = 0+1 = 1）。第 4 步看到 rank 1 初始化失败或找不到设备。

**常见错误**：
- `--nproc_per_node` 大于可见设备数 → 报找不到设备，先查 `echo $CUDA_VISIBLE_DEVICES`（昇腾：`echo $ASCEND_RT_VISIBLE_DEVICES`）
- 卡在 `Initializing process group` 不动 → 网络/防火墙问题，加 `NCCL_DEBUG=INFO`（昇腾：调大 `HCCL_CONNECT_TIMEOUT`、`ASCEND_GLOBAL_LOG_LEVEL=1`）看卡在哪一步

### 实验 3：修复版本不匹配

**目标**：检查系统 CUDA 与 PyTorch 的 CUDA 版本，学会对齐版本并重装 PyTorch。

**步骤**：

1. 看系统驱动支持的最高 CUDA 版本：
```bash
nvidia-smi  # 右上角 CUDA Version 字段
```

2. 看 PyTorch 的 CUDA 版本：
```bash
python -c "import torch; print(torch.__version__)"        # 如 2.1.0+cu118
python -c "import torch; print(torch.version.cuda)"       # 如 11.8
python -c "import torch; print(torch.cuda.is_available())" # 必须 True
```

3. 判定：`torch.cuda.is_available()` 为 False，或 import 报 `libcudart.so: cannot open shared object file` → 版本不匹配。昇腾同理：`torch.npu.is_available()` 为 False、import 报 `libascendcl.so` 找不到 → torch_npu 与 CANN 版本不配套，查 CANN 版本配套表。

4. 先卸载再装匹配版本（到 pytorch.org 查当前稳定版对应的 CUDA 版本）：
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. 重新验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**预期输出**：重装后你会看到 `torch.__version__` 带 `+cu121` 后缀，`torch.cuda.is_available()` 打印 `True`。

**常见错误**：
- `pip install torch` 默认装了 CPU 版（版本号无 `+cu` 后缀，`is_available()` 为 False）
- 系统 CUDA 版本（驱动支持）比 PyTorch 要求的低 → 必须升级驱动，而不是换更老的 PyTorch
- 重装前没先卸载 → 残留旧文件，import 仍报错

## 9.6 自测题

1. 遇到 CUDA OOM，第一步应该做什么？（提示：减小 batch size）

<details>
<summary>参考答案</summary>

第一步：减小 batch size（从 32 → 16 → 8）。第二步：检查是否有其他进程占用显存（`nvidia-smi`）。第三步：考虑梯度累积或梯度检查点。
</details>

2. 多卡训练只有 1 卡在跑，可能是什么原因？（提示：NCCL/HCCL 配置）

<details>
<summary>参考答案</summary>

可能原因：①`CUDA_VISIBLE_DEVICES`/`ASCEND_RT_VISIBLE_DEVICES` 环境变量只设置了 1 张卡；②NCCL/HCCL 没安装或版本不对；③多节点网络不通。诊断：运行 `torch-distributed-npu-test.py`。
</details>

3. `torch-distributed-npu-test.py` 的作用是什么？（提示：诊断多卡通信）

<details>
<summary>参考答案</summary>

诊断多卡通信是否正常。通过 AllReduce 操作测试：①进程组能否初始化；②各卡能否互相通信；③计算结果是否正确。用于排查分布式训练挂起问题（GPU 用 nccl 后端，昇腾 NPU 用 hccl 后端）。
</details>

4. NPU 上遇到 `RuntimeError: NPU out of memory`，第一步应该做什么？（提示：清场）

<details>
<summary>参考答案</summary>

第一步：用 `npu-smi info -t process -i 0` 查看是否有残留进程占用 NPU 内存，有则 `kill -9 <pid>` 清场。第二步：减小 batch size。第三步：检查是否内存碎片导致，必要时重启进程或查 CANN 文档做碎片整理。
</details>

5. 分布式训练报 `HCCL error`，可能的原因和排查步骤？（提示：超时/网络/版本）

<details>
<summary>参考答案</summary>

可能原因：①`HCCL_CONNECT_TIMEOUT` 太短导致连接超时；②多节点网络不通或防火墙阻挡；③CANN/HCCL 版本不匹配。排查：先调大超时（`export HCCL_CONNECT_TIMEOUT=1800`），再检查网络连通性（`ping`、`nc -zv <master_addr> 29500`），最后用 `torch-distributed-npu-test.py` 验证通信。
</details>

6. import torch_npu 报 `libascendcl.so: cannot open shared object file`，怎么修？（提示：CANN 版本配套）

<details>
<summary>参考答案</summary>

torch_npu 与 CANN 版本不配套。解法：查 CANN 版本配套表，确认当前 CANN 版本对应的 torch_npu 版本，重装匹配的 torch_npu（或升级/降级 CANN），再重新 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
</details>

7. `npu-smi` 和 `msprof` 分别用来做什么？（提示：监控/性能分析）

<details>
<summary>参考答案</summary>

`npu-smi` 是昇腾设备监控工具，查看 NPU 状态、内存占用、运行进程（等效 nvidia-smi）。`msprof` 是昇腾性能分析器，分析算子耗时、AI Core 利用率、HCCL 通信耗时（等效 ncu/nsys），用于定位性能瓶颈。
</details>

---

**上一章**：[第8章：实战路径](../08-实战路径/README.md) | **下一章**：[第10章：开源框架昇腾适配](../10-开源框架昇腾适配/README.md)
