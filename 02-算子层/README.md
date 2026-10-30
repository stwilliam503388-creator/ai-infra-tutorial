---
title: 第2章 算子层——把数学变成机器的肌肉记忆
tags: [ai-infra, 教程, 算子, cuda, triton, flashattention]
difficulty: ⭐~⭐⭐⭐
章节: 02
状态: 完成
---

# 第2章 算子层

> 导航：[第1章 全景图](../01-全景图/README.md) → **第2章 算子层** → [第3章 框架层](../03-框架层/README.md)

> 一句话：框架是菜谱，算子是刀工。矩阵乘法太慢？别改菜谱，先磨刀。

---

## 🥢 类比开场：算子 = 切菜 / 炒菜的具体动作

把训练一个大模型比作开一家餐厅：

- **菜谱（模型结构）**：Transformer 有 12 层、8 个头、隐藏维度 768……这是"做什么菜"。
- **算子（Kernel）**：`matmul`（矩阵乘法）、`attention`（注意力）、`layernorm`（归一化）就是"**切菜、炒菜、颠勺**"这些**具体动作**。
- **框架（PyTorch）**：是后厨主管，把菜谱拆成一道道工序，喊一声"上 matmul"，算子就去执行。

| 厨房概念 | AI Infra 概念 | 说明 |
|---|---|---|
| 切菜这个动作本身 | **算子（Operator/Kernel）** | 输入是菜（张量），输出是加工好的菜 |
| 刀工好坏 | **Kernel 优化程度** | 同样的菜，好刀工快 10 倍 |
| 一把刀切所有菜 | **通用 Kernel** | PyTorch 默认实现，啥都能干，但都不极致 |
| 专用厨具（切葱器） | **专用 Kernel** | 为特定形状/硬件特调，快但挑场景 |

本章要回答三个问题：**刀（NPU/GPU）长什么样？怎么磨刀（写 kernel）？磨好的刀有哪些（生态）？**

---

## 1. NPU 与 GPU：昇腾核心 vs 参考对比 ⭐⭐

### 1.1 昇腾 NPU：国产 AI 芯片主力

| 维度 | 昇腾 NPU (Ascend) | NVIDIA GPU（参考） |
|------|-------------------|-------------------|
| **定位** | 国产替代主力 | 事实标准 |
| **代表型号** | 昇腾 910B / 310P | A100 / H100 |
| **软件栈** | CANN + AscendCL | CUDA + cuDNN |
| **算子开发** | Ascend C / Triton-Ascend | CUDA / Triton |
| **通信库** | HCCL | NCCL |
| **适用场景** | 国产化要求、推理 | 训练+推理全覆盖 |

**关键认知**：昇腾 NPU 的编程模型与 GPU 类似，但接口不同。`torch.npu` 对应 `torch.cuda`，`npu-smi` 对应 `nvidia-smi`。

### 1.2 算子开发：Triton → Triton-Ascend

| 层级 | GPU 路线 | 昇腾路线 |
|------|---------|---------|
| DSL | Triton | Triton |
| 编译器后端 | CUDA | triton-ascend |
| 目标 IR | PTX | AscendNPU-IR |
| 运行时 | CUDA Runtime | CANN Runtime |

**你已有的优势**：如果你会 Triton，迁移到昇腾只需要换后端（triton-ascend），DSL 语法不变。

---

## 2. CUDA 核心概念：GPU 的"组织架构"（参考）⭐⭐

### 2.1 CPU vs GPU：一个厨师长 vs 一千个帮厨

| 维度 | CPU | GPU |
|---|---|---|
| 核心数 | 8~64 个"全能大厨" | 几千~上万个"简单帮厨" |
| 单核能力 | 强（复杂逻辑、分支预测） | 弱（只会简单算术） |
| 擅长的活 | 串行、控制流 | 大量并行、同构计算 |
| 对应矩阵乘法 | 一个人慢慢切整桌菜 | 每人切一片，同时切 |

GPU 之所以适合矩阵乘法，是因为矩阵乘法是**"同一动作重复百万次"**——帮厨模式完美匹配。

### 2.2 线程层级：Thread → Warp → Block → Grid ⭐⭐

GPU 上干活的最小单位是**线程（Thread）**，但线程不是散兵游勇，它们按四级结构组织：

| 层级 | 是什么 | 类比 | 关键约束 |
|---|---|---|---|
| **Thread**（线程） | 执行一条指令的最小单位 | 一个帮厨 | 一个线程算一个/几个元素 |
| **Warp**（束） | 32 个线程一组，**同进同出**（SIMT，单指令多线程） | 32 人一排同时切 | 32 是硬件定的，不可改 |
| **Block**（线程块） | 若干 warp 组成，可共享一块**共享内存（Shared Memory）** | 一个切菜小组 | 同一 block 内可通信，跨 block 不行 |
| **Grid**（网格） | 所有 block 的集合，覆盖整个任务 | 整个后厨团队 | block 之间互不可见、无同步 |

> 类比：**餐厅 = Grid，切菜小组 = Block，小组内 32 人一排 = Warp，每个人 = Thread。**
> 小组之间不串门（不能访问对方的共享内存），这是 GPU 并行模型最重要的纪律。

### 1.3 存储层级：数据放得越近，拿得越快 ⭐⭐

| 层级 | 大小 | 速度 | 谁用 |
|---|---|---|---|
| **Register**（寄存器） | 每个线程几十个 | 最快 | 线程私有，算中间结果 |
| **Shared Memory**（共享内存） | 每 block 几十~几百 KB | 快（~10TB/s 级） | block 内共享，手动管理 |
| **L2 Cache** | 几十 MB | 中 | 自动缓存 |
| **VRAM**（显存/HBM） | 80GB（H100） | 慢（~3TB/s） | 所有数据的"大仓库" |

回到第 1 章的数字：**算 1 次的时间，显存能搬 200 次数据。** 所以 kernel 优化的本质是——把数据从"仓库"（VRAM）搬到"案板"（Shared Memory/Register）上反复用，别一趟一趟回仓库取。

### 1.4 一个 Kernel 的完整执行流程

```cuda
// CUDA 矩阵加法（伪代码级别，示意层级关系）
__global__ void vec_add(float* x, float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程编号
    if (i < n) out[i] = x[i] + y[i];                // 越界保护（就是 Triton 的 mask）
}

vec_add<<<grid_size, block_size>>>(x, y, out, n);   // <<<Grid, Block>>> 两级并行
```

| 代码要素 | 对应层级 | 说明 |
|---|---|---|
| `blockIdx` / `blockDim` / `threadIdx` | Grid / Block / Thread | 三级编号定位一个线程 |
| `<<<grid, block>>>` | 启动参数 | 告诉 GPU 开多少 block、每 block 多少线程 |
| `if (i < n)` | 边界保护 | 数组长度不是 block 整数倍时防越界 |

> **为什么学 CUDA 概念？** 因为 Triton 把 `threadIdx` 藏起来了，但 `blockIdx`（program_id）和边界保护（mask）你仍然要亲手写。不懂硬件层级，Triton 代码出错了都不知道错在哪。

---

## 3. Triton 核心概念：用 Python 写 Kernel ⭐⭐

### 2.1 Triton 是什么

**Triton** 是 OpenAI 出的 GPU kernel 编程语言：**用 Python 语法写 kernel，自动做分块、调度、寄存器分配**，性能接近手写 CUDA，但代码量只有十分之一。它是当前**写算子的事实标准**（FlagGems 用它、PyTorch 的 `torch.compile` 后端也用它）。

### 2.2 四个核心概念

| Triton 概念 | 对应 CUDA | 类比 | 说明 |
|---|---|---|---|
| **Program**（程序） | Block | 一个切菜小组 | 一次 kernel 调用中，每个 program 独立跑同一段代码 |
| **Grid**（网格） | Grid | 整个后厨团队 | `grid = (M, N)` 表示开 M×N 个 program |
| **Block**（数据块） | —（新概念） | 小组要切的"那片菜" | 每个 program 处理的一块连续数据，如 `BLOCK_M×BLOCK_N` |
| **Tile**（瓦片） | 分块策略 | 案板上的一块豆腐 | 大矩阵切成的小块，Triton 自动把它放进寄存器/共享内存 |

> **关键心智模型**：写 Triton 时你只回答两个问题——**"我负责哪一块数据？"（program_id + offsets）** 和 **"这块数据怎么算？"（块内张量运算）**。至于"32 个线程怎么排、谁用哪个寄存器"，Triton 全包了。

### 2.3 CUDA vs Triton：同一个加法

| | CUDA（手写） | Triton（半自动） |
|---|---|---|
| 线程编号 | `threadIdx.x` 手动算 | `tl.program_id(0)` 一行 |
| 数据分块 | 自己写循环+指针偏移 | `tl.arange` 自动生成偏移 |
| 边界保护 | `if (i < n)` | `mask=` 参数 |
| 调度到 warp/寄存器 | 全靠自己 | **自动** |
| 代码量 | 50~100 行 + 编译配置 | 10~20 行 |

```python
# Triton 向量加法（完整可运行，见 code/vector-add/vector_add.py）
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                   # ① 我是第几个 program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)    # ② 我负责哪一段数据（Tiling）
    mask = offsets < n_elements                              # ③ 尾部不够一块？掩码保护
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)                           # Grid = 需要的 program 数
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)   # ① Tiling：每个块管 1024 个元素
    return out


if __name__ == "__main__":
    x = torch.randn(10000, device="cuda")
    y = torch.randn(10000, device="cuda")
    out = vector_add(x, y)
    assert torch.equal(out, x + y)                           # 自检：和 PyTorch 结果一致
    print("OK: Triton vector add 与 torch 结果一致")
```

> 别小看这个加法：**tiling + mask + 融合这三段式，就是 FlashAttention 的全部骨架**，只是块内操作从"加"变成了"QK^T→softmax→×V"。

### 3.4 昇腾注意点：Triton-Ascend 的差异

Triton 在昇腾上通过 **triton-ascend** 后端运行，DSL 语法基本一致，但有几点差异：

| 差异点 | GPU (CUDA) | 昇腾 (Ascend) |
|--------|-----------|--------------|
| 设备指定 | `device="cuda"` | `device="npu"` |
| 后端选择 | 默认 CUDA | `export TRITON_DEVICE=ascend` |
| 编译目标 | PTX | AscendNPU-IR |
| 性能调优 | `num_warps`/`num_stages` | 需考虑 UB（Unified Buffer）大小 |
| 内存层级 | Shared Memory / L2 | UB / L1 / L2 / GM |

**关键认知**：昇腾的 **UB（Unified Buffer）替代了 GPU 的 Shared Memory**，但容量和访问模式不同。写 Triton 算子时，tile 大小要适配 UB 容量，否则会溢出。

```python
# 昇腾版：注意 tile 大小适配 UB
import torch
import torch_npu  # 昇腾适配层

x = torch.randn(10000, device="npu")
y = torch.randn(10000, device="npu")
out = vector_add(x, y)  # Triton kernel 在 NPU 上执行
assert torch.equal(out, x + y)
```

### 3.5 Ascend C：昇腾原生算子开发

如果 Triton 无法满足性能要求，可以用 **Ascend C** 写原生算子（相当于 CUDA C++ 的角色）。

| 维度 | CUDA C++ | Ascend C |
|------|----------|----------|
| 语言 | C++ 扩展 | C++ 扩展 |
| 编译器 | nvcc | ccec (毕昇编译器) |
| 核函数标记 | `__global__` | `__global__ __aicore__` |
| 线程索引 | `threadIdx.x` | `GetBlockIdx()` |
| 共享内存 | `__shared__` | `TPipe` + `TBuf` |
| 同步原语 | `__syncthreads()` | `SyncAll()` |

**Ascend C 算子骨架**（vector add 示例）：

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        // 计算每个核处理的数据量
        this->blockLength = totalLength / GetBlockNum();
        this->tileLength = 8 * 1024 / sizeof(float);  // UB tile 大小
        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + this->blockLength * GetBlockIdx(), this->blockLength);
        // 分配 UB 空间
        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength, tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
```

**Ascend C vs Triton 选型**：

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速原型 | Triton-Ascend | Python 写，迭代快 |
| 极致性能 | Ascend C | 可精细控制 UB 和流水 |
| 通用算子 | FlagGems | 现成的 Triton 算子库 |
| 注意力算子 | FlagAttention | 昇腾优化的 FlashAttention |

**关键认知**：FlagGems 里的算子都是 Triton 写的（跨硬件），FlagAttention 是 Ascend C 写的（昇腾专用，性能更好）。这就是"通用 vs 专用"的权衡。

### 3.6 TileLang：DeepSeek 开源的 tile 级编程语言 ⭐⭐

如果 Triton 是"半自动"，**TileLang**（[tile-ai/tilelang](https://github.com/tile-ai/tilelang)，DeepSeek 团队开源）就是再往上一层：**你只描述 tile 之间的数据流，编译器自动生成调度、同步和设备代码**。DeepSeek V4 系列模型训练中的大部分算子就是用 TileLang 实现的（官方口径）。

**四种算子开发路径对比**：

| 维度 | CUDA C++（参考） | Triton | TileLang | Ascend C |
|------|-----------------|--------|----------|----------|
| 抽象层次 | 线程级 | block/tile 级，手写调度 | tile 级，**编译器自动调度** | 指令级，全手动 |
| 性能上限 | 最高 | 高（依赖手写 tiling） | 高（自动调度接近上限） | 最高 |
| 学习成本 | 高 | 中 | 低-中 | 高 |
| 昇腾支持 | N/A | Triton-Ascend | **tilelang 原生 Ascend 950 后端** | 原生 |

**昇腾支持现状**（2026-09-30 DeepSeek 开源昇腾基础组件，含 TileLang 昇腾版）：

- 上游 tilelang 已合入**原生 Ascend 950 后端**（[PR #3308](https://github.com/tile-ai/tilelang/pull/3308)）：`target="ascend"` 即可面向 950（架构代号 `dav-3510`）开发，端到端走「Ascend dialect → 硬件 lowering → 自动调度同步 → 毕昇编译器生成设备码 → NPU runtime」全链路
- 编程模型：**显式 UB/L1/L0 存储**、Cube（矩阵）+ Vector（向量）混合编程（`T.SimdVF` / `T.SimtVF` 区域）、支持 MXFP8/MXFP4 块缩放数据类型
- DeepSeek 官方表态：V4 训练中用到的**每一个 TileLang 算子，在昇腾上都有对应的高性能实现**；TileLang 昇腾版对 Ascend C 底层指令做了封装，"高级编程方式且不损失硬件性能"
- 注意：原生后端目前**只覆盖 950**；A2/A3 走社区维护的实现（见仓库 Ascend README）

**最小示例**：同一个 vector add，对比 §3.2 的 Triton 写法——

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="ascend")  # GPU 上是 target="cuda"
def vector_add(N: int, blk: int = 1024):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"),
             Y: T.Tensor((N,), "float32"),
             Z: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, blk), threads=128) as bx:
            for i in T.Parallel(blk):          # 描述 tile 内并行
                Z[bx * blk + i] = X[bx * blk + i] + Y[bx * blk + i]
    return main
```

差异一眼可见：Triton 里你要自己算 `pid * BLOCK + offsets`、手写 mask；TileLang 里 `T.Parallel` 把"tile 内并行"交给编译器，越界处理、调度、同步都自动完成。**少写调度代码 = 少出调度 bug**，代价是对极端优化的控制粒度让渡给编译器。

> 第 18 章会讲到：DeepSeek 的 TileKernels 算子库和 DeepInfer 推理框架里的大量自定义 kernel 都是 TileLang 写的——学会 TileLang，就拿到了读懂 DeepSeek 推理栈源码的钥匙。

---

## 4. 方案：手写优化 Kernel 的三板斧 ⭐⭐

框架默认算子是"通用做法"（正确、易用、但中庸）。要榨干硬件，套路就三板斧：

1. **Tiling（分块）**：把大矩阵切成小块（如 128×128），数据留在片上（Shared Memory/寄存器）反复用，而不是每次回显存搬。
2. **向量化 + 并行化**：每个线程一次算 4/8 个元素（向量化），同时把任务分给成千上万个线程（并行化）。
3. **算子融合（Fusion）**：把 `QK^T → softmax → ×V` 合成一个 kernel，中间结果**不出显存**，省掉 O(n²) 的读写。

这三个套路是所有高性能 kernel 的共同语言：**CUDA 手写、Triton 半自动、FlashAttention 把套路做到极致。**

---

## 5. FlashAttention 原理详解 ⭐⭐⭐

### 4.1 标准注意力为什么慢

标准注意力 = 3 个独立 kernel，中间结果全部落显存：

```text
        ┌─ 写回显存 ─┐  ┌─ 写回显存 ─┐
QK^T ──▶ S (n×n) ──▶ softmax(S) (n×n) ──▶ P @ V ──▶ O
        中间矩阵 O(n²)   又是 O(n²)
```

| 问题 | 严重程度 | 后果 |
|---|---|---|
| 中间矩阵 `S` 是 n×n | 序列长 n=4096 → 16M 个元素 | 显存 O(n²)，长序列直接爆炸 |
| 中间结果写两次显存 | 每次读写都是带宽 | 慢 2~3 倍 |
| softmax 需要全局 max | 必须等整行算完 | 无法分块并行（后续解决） |

### 4.2 三个关键思想

**① IO 感知（IO-aware）**：传统优化只看"算得快不快"（FLOPs），FlashAttention 反过来说——**瓶颈不是算力，是显存带宽**（第 1.3 节：算 1 次能搬 200 次）。所以目标从"减少计算"变成"**减少 HBM 访问**"。

**② Tiling（分块）**：把 `Q、K、V` 切成小块，每次只把一个块读进 Shared Memory，在片上完成"这个块的 QK^T→softmax→×V"，结果累加到输出。中间 `n×n` 矩阵**从头到尾不存在**——这就是显存从 O(n²) 降到 O(n) 的原因。

**③ Online Softmax（在线 softmax）**：普通 softmax 要先看完整行才能算（要全局 max），分块后没看完全部数据怎么办？——**用"跑了多少算多少"的增量公式**：

```text
每处理一个块：
  m_new = max(m_old, 本块 S 的行最大值)      # 更新运行最大值
  P = exp(S - m_new)                          # 按新最大值重新缩放
  O = O * exp(m_old - m_new) + P @ V          # 旧结果打个"折扣"再累加
  l = l * exp(m_old - m_new) + P.sum(dim=1)   # 同步更新归一化分母
最后：O / l                                   # 统一除分母，就是标准 softmax 的结果
```

> 类比：你边吃边给自助餐打分，中途来了更好吃的——不用重吃，**旧分数打个折扣加上新分数**就行。这就是 online softmax。

### 4.3 效果对比

| 指标 | 标准注意力 | FlashAttention |
|---|---|---|
| 中间矩阵显存 | O(n²) | **O(n)**（只剩输入输出） |
| HBM 访问量 | O(n² + n²d) | O(n²d / M)（M 是片上缓存大小） |
| 速度 | 基准 | 2~4 倍（长序列更夸张） |
| 数值精度 | 标准 | **不降反升**（块内逐块归一，避免大数相减） |

> 参考文献：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（NeurIPS 2022，用算法改变了整个 LLM 的序列长度上限）。

---

## 6. 生态项目地图：磨好的刀 ⭐⭐

### 5.1 总览

| 项目 | 是什么 | 定位 |
|---|---|---|
| **CUDA** | NVIDIA 的 GPU 编程语言/平台 | 最底层，手写 kernel 的原点。上限最高，成本也最高 |
| **Triton** | OpenAI 的 kernel 编程语言，Python 语法 | 用 Python 写 kernel，性能接近手写，**当前事实标准** |
| **FlashAttention** | 注意力融合算子（论文+开源） | 一个算子改变了 LLM 的序列长度上限 |
| **FlashInfer** | 面向 LLM **推理**的算子库 | 分页注意力、KV cache，vLLM 的底层依赖 |
| **FlagGems** | 国产通用算子库（Triton 实现） | 一个算子库跑遍 GPU/NPU，跨硬件 |

### 5.2 FlagGems vs FlashInfer：正面对比 ⭐⭐

两者都是"算子库"，但**服务场景完全不同**——一个是"通用菜刀套装"，一个是"刺身专用刀"：

| 维度 | **FlagGems** | **FlashInfer** |
|---|---|---|
| 出身 | 旷视/算力生态，Triton 生态 | 字节/港科大系，CUDA/CUTLASS |
| 定位 | **训练+推理通用**算子库（全能） | **LLM 推理专用**（专精） |
| 覆盖范围 | 全算子：matmul/attention/norm/reduction…… | 只做推理热点：分页注意力、KV cache、采样 |
| 核心卖点 | **跨硬件**：一套代码跑 GPU/NPU | **极致性能**：为推理场景特调（变长序列、缓存复用） |
| 技术栈 | Triton（可移植性优先） | CUDA 手写（性能上限优先） |
| 典型用户 | 想国产化/跨硬件的训练团队 | vLLM、SGLang 等推理引擎 |
| 类比 | 超市"全套刀具" | 日料店"刺身刀" |
| 怎么选 | 要通用、要跨硬件 → 用它 | 推理性能敏感、用 vLLM → 它已在里面 |

> **选型逻辑**：学习用 Triton；训练要提速看 FlagGems 的通用算子；推理压性能，FlashInfer 的思路（分页注意力、KV cache）是必读课。

---

## 7. 代码示例：从加法到矩阵乘法到注意力 ⭐⭐

### 6.1 Triton 矩阵乘法（完整可运行）

矩阵乘法是 AI 的"米饭"，下面这个 kernel 是第 3 节三板斧的完整示范：

```python
# matmul_triton.py —— 需要 NVIDIA GPU + triton
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)                              # ① Grid: 我是 (M 方向第几个, N 方向第几个)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # ② Tiling: 负责的 M 行、N 列
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 累加器放寄存器，不回显存

    for k in range(0, K, BLOCK_K):                        # ③ K 方向循环：每次搬一个 K 块
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)                               # ④ 块内矩阵乘（硬件张量核）

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))   # Grid 大小 = 输出块数
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = matmul(a, b)
    ref = a @ b
    assert torch.allclose(c, ref, atol=1e-2), f"最大误差 {(c - ref).abs().max():.4f}"
    # 性能对照：Triton kernel vs torch.matmul
    import time
    for fn, name in [(lambda: matmul(a, b), "Triton matmul"),
                     (lambda: a @ b, "torch.matmul")]:
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms")
    print("OK: 结果一致，且能对比性能")
```

### 6.2 简化版 FlashAttention（在线 softmax 最小实现）

下面这段**纯 PyTorch** 代码不需要 Triton/GPU 也能跑，专门演示 4.2 节的 online softmax 公式：

```python
# flash_attention_lite.py —— 演示核心思想，非性能实现（可 CPU 运行）
import torch


def flash_attention_lite(Q, K, V, block_size=16):
    """FlashAttention 核心思想的玩具版：分块 + 在线 softmax，逐块扫描 K/V。"""
    seq_len, d = Q.shape
    out = torch.zeros_like(Q)
    m = torch.full((seq_len,), float("-inf"))   # 运行最大值（初始 -inf）
    l = torch.zeros(seq_len)                    # 归一化分母（exp 和）

    for start in range(0, seq_len, block_size):          # ① Tiling: 按块扫 K/V
        end = min(start + block_size, seq_len)
        S = Q @ K[start:end].T                            # 本块的 QK^T
        m_new = torch.maximum(m, S.max(dim=1).values)     # ② 更新最大值
        P = torch.exp(S - m_new[:, None])                 # 按新最大值缩放（数值稳定）
        alpha = torch.exp(m - m_new)                      # 旧结果的"折扣系数"
        out = out * alpha[:, None] + P @ V[start:end]     # ③ 打折 + 累加
        l = l * alpha + P.sum(dim=1)
        m = m_new

    return out / l[:, None]                               # 最后统一归一化


if __name__ == "__main__":
    torch.manual_seed(0)
    seq, d = 64, 32
    Q, K, V = (torch.randn(seq, d) for _ in range(3))

    mine = flash_attention_lite(Q, K, V, block_size=16)
    ref = torch.softmax(Q @ K.T, dim=1) @ V               # 标准注意力
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()
    print("OK: online softmax 结果与标准 attention 一致")
```

> 玩具版正确性等于 FlashAttention，区别只在性能：真实版用 Triton/CUDA 把"块"放进 Shared Memory 并行算；玩具版在 Python 里串行循环。**先看懂公式，再看懂性能。**

---

## 8. 动手实验：自己跑一遍 ⭐⭐

> 环境要求：NVIDIA GPU（或可用的 Triton 后端），`pip install torch triton`。

### 实验 1：跑通第一个 Triton kernel

```bash
cd code/vector-add
pip install -r requirements.txt   # torch>=2.1, triton>=2.1
python vector_add.py              # 输出 OK 即成功
```

**任务**：把 `BLOCK_SIZE` 从 1024 改成 1000（非 2 的幂）再跑，看报什么错，猜猜为什么——答案在第 8 节案例 1。

### 实验 2：矩阵乘法正确性 + 性能对比

把 6.1 节代码存成 `matmul_triton.py` 运行，观察：

- 结果断言是否通过？`atol=1e-2` 改成 `atol=1e-5` 会不会失败？（提示：TF32 精度，见第 8 节案例 2）
- Triton kernel 和 `torch.matmul` 谁快？改变矩阵形状（如 4096×4096）再比一次。

### 实验 3：调参看性能曲线

在实验 2 基础上，把 `BLOCK_M/BLOCK_N` 改成 16、32、128、256，记录每档耗时：

| BLOCK 大小 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 耗时（ms） | ？ | ？ | ？ | ？ | ？ |

思考：为什么太小和太大都慢？（太小→复用差、调度开销占比高；太大→寄存器不够、占用率下降。见第 8 节案例 3）

### 实验 4：验证 FlashAttention 玩具版

运行 6.2 节代码，然后做两个修改验证你理解了 online softmax：

1. 把 `block_size` 改成 4、8、64——结果应该始终一致（这正是"分块不影响正确性"的证明）。
2. 把 `m` 的初始值从 `-inf` 改成 `0`——结果会错（想想为什么：`alpha = exp(m - m_new)` 就不是"折扣"了）。

### 实验 5（进阶）：用 `torch.compile` 感受自动融合 ⭐⭐⭐

```bash
python -c "
import torch
torch.manual_seed(0)
x = torch.randn(4096, 4096, device='cuda')
# 一个典型的三算子链：矩阵乘 + ReLU + 矩阵乘
def f(a, b, c):
    return (a @ b).relu() @ c
f(x, x, x); torch.cuda.synchronize()
# 对比原生 vs torch.compile（会自动做算子融合）
fc = torch.compile(f)
fc(x, x, x); torch.cuda.synchronize()
import time
for fn, name in [(f, 'eager'), (fc, 'compiled')]:
    fn(x, x, x); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): fn(x, x, x)
    torch.cuda.synchronize()
    print(f'{name}: {(time.perf_counter()-t0)/50*1e3:.2f} ms')
"
```

**预期收获**：`compiled` 通常比 `eager` 快——因为 `torch.compile` 把三个 kernel 融合成了一个，中间结果不再落显存。这就是第 3 节"算子融合"的自动化版本。

---

## 9. 排障实战：三个真实场景 ⭐⭐⭐

> 参考 [ml-engineering debug/pytorch.md](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md)，以下案例按"症状 → 根因 → 解法"组织。

### 案例 1：编译失败 —— `tl.arange` 尺寸必须是 2 的幂

**症状**：实验 1 把 `BLOCK_SIZE=1024` 改成 `1000` 后报错，形如：

```text
AssertionError: tl.arange() requires the size to be a power of two, got 1000
```

**根因**：Triton 的 `tl.arange(0, BLOCK_SIZE)` 生成的偏移量最终要分配到 warp 上（第 1.2 节：warp 固定 32 线程），**非 2 的幂无法均匀切分**，编译器直接拒绝。

**解法**：`BLOCK_SIZE` 一律用 2 的幂（128/256/512/1024）；数据长度不是整数倍时，用 `mask` 兜底——**这也是 mask 必须存在的根本原因**。

> 排查口诀：Triton 编译报错，先检查**所有 `tl.constexpr` 是不是 2 的幂**；再把 `if` 条件里的运行时值移到 host 端（kernel 里动态分支不支持）。

### 案例 2：结果不正确 —— 精度模式对不上

**症状**：矩阵乘法断言 `atol=1e-5` 失败，最大误差 ~1e-3，但 `atol=1e-2` 能过。

**根因**：`tl.dot` 对 float32 默认走 **TF32**（Tensor Float 32，牺牲精度换 2 倍速度）；而 `torch.matmul` 默认关闭 TF32。两边精度模式不一致，误差必然大于 1e-5。

**解法**（按需选一）：

```python
# 方案 A：放宽容差（工程上常用，LLM 训练本身有噪声）
assert torch.allclose(c, ref, atol=1e-2)

# 方案 B：关掉 Triton 的 TF32，用 IEEE 精度（追求精确）
# 在 tl.dot 调用处加参数：tl.dot(a, b, input_precision="ieee")

# 方案 C：两边都开 TF32（追求速度）
torch.backends.cuda.matmul.allow_tf32 = True
```

> 排查口诀：**结果"差不多对"= 精度模式问题**；结果"完全不对"= 逻辑/mask/指针问题。另外别忘了检查输入是不是真的在 GPU 上——CPU 张量混进 kernel 会得到神秘错误。

### 案例 3：性能不达标 —— kernel 没吃满 GPU

**症状**：`nvidia-smi` 里 GPU-Util 只有 30%，Triton kernel 比预期慢 5 倍，功耗上不去。

**根因**（按概率排序）：

| 嫌疑 | 表现 | 验证方法 |
|---|---|---|
| BLOCK 太小（如 16×16） | 每块算太少，调度开销占比高 | 实验 3 调大 BLOCK 对比 |
| Grid 太小（矩阵小但分块大） | 只有几个 program，几千个 SM 闲着 | `grid` 打印出来看数量 |
| 数据没在显存 | 每次计算前走 PCIe 拷贝 | 检查是否漏了 `.cuda()` |
| kernel 太短 | 启动开销 > 计算时间 | 循环 100 次测平均，别测单次 |
| 精度模式拖累 | 用了 IEEE 而非 TF32/FP16 | 看案例 2 |

**解法**：先用 `torch.cuda.synchronize()` + 多次计时测准（单次计时会被异步启动骗了）；再用 **NVIDIA Nsight Compute（`ncu`）** 看 memory/compute throughput 谁先到 100%——**哪个到顶，瓶颈就是哪个**。

```bash
# 测准：synchronize 后计时，跑多次取平均（代码见实验 2）
# 深度分析（需要 root 权限）：
# ncu --set full python matmul_triton.py
```

> 排查口诀：**性能问题先测准（synchronize + 多次平均），再问 ncu，最后才改代码。** 别凭感觉调 BLOCK——用数据说话。

---

## 10. 自测题

**Q1** ⭐⭐：GPU 有 Thread/Warp/Block/Grid 四级结构。请解释：为什么 Triton 代码里看不到 Warp，却仍然受它约束（提示：联系 `BLOCK_SIZE` 为什么必须是 2 的幂）？

<details>
<summary>答案</summary>

Warp 是硬件层概念（固定 32 线程一组、同进同出），Triton 把"线程怎么排进 warp"自动处理了，所以源码里看不到。但约束还在：`tl.arange` 的尺寸要能被 warp 均匀切分，所以必须是 2 的幂；`BLOCK_SIZE=1000` 会被编译器直接拒绝。理解 Warp，才能理解 Triton 的"神秘"编译错误。
</details>

**Q2** ⭐⭐⭐：FlashAttention 用了 online softmax 解决了什么问题？如果不用它，直接对每个块单独做标准 softmax（`exp(S) / sum(exp(S))`），结果为什么不对？

<details>
<summary>答案</summary>

标准 softmax 需要整行的全局 max 才能数值稳定地归一化。分块后每个块只能看到局部数据：若每块单独做 softmax，各块的归一化分母（sum）不统一，累加时权重错误；且直接 `exp(S)` 在大数值时可能溢出（inf）。online softmax 用"运行最大值 m + 折扣系数 exp(m_old - m_new)"的增量公式，无论怎么分块、分多少块，最终结果都和全局 softmax 完全一致（玩具版代码验证过）。它让"分块"和"softmax 的全局性"这对矛盾得以共存。
</details>

**Q3** ⭐⭐：你的 Triton matmul 比 torch.matmul 慢 5 倍，GPU-Util 只有 30%。按什么顺序排查？（写出至少 4 步）

<details>
<summary>答案</summary>

1. 测准：加 `torch.cuda.synchronize()`，循环 20+ 次取平均（异步执行会让单次计时虚低）；2. 看输入是否在 GPU 上（漏 `.cuda()` 会走 PCIe）；3. 检查 BLOCK 大小是否过小/过大（实验 3 扫 16~256），检查 grid 数量是否覆盖全卡；4. 用 `ncu` 看 memory/compute throughput，判断瓶颈在带宽还是算力；5. 检查精度模式（是否误开了 IEEE 拖慢速度）。原则：先用数据定位瓶颈，再改代码。
</details>

---

## 小结

- **硬件**：GPU 按 Thread→Warp→Block→Grid 四级组织，数据按 Register→Shared→L2→VRAM 四级存放；**谁离算得快谁就贵**，kernel 优化 = 让数据少回"仓库"。
- **Triton**：Python 写 kernel 的事实标准，只回答"我负责哪块数据 + 这块怎么算"，调度交给编译器；`program_id` 对应 CUDA 的 `blockIdx`。
- **FlashAttention**：IO 感知（瓶颈是带宽不是算力）+ Tiling（中间矩阵不落显存）+ Online Softmax（分块不影响正确性），显存 O(n²)→O(n)，速度 2~4 倍。
- **生态**：学习用 Triton；训练提速看 FlagGems（通用、跨硬件）；推理压性能看 FlashInfer（专用、极致）。
- **排障**：编译错查 constexpr/2 的幂；结果"差不多对"查 TF32 精度；性能差先测准（synchronize）再上 ncu。
- **下一步**：第 3 章进入框架层，看 PyTorch 怎么把这些算子组织成训练/推理的流水线。

---

**上一章**：[第1章 全景图](../01-全景图/README.md) | **下一章**：[第3章 框架层](../03-框架层/README.md)
