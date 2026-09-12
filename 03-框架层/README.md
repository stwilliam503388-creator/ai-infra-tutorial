---
title: 第3章 框架层——用 Python 定义模型，把底层留给框架
tags: [ai-infra, 教程, 框架, pytorch, autograd, mindspore, torch_npu]
章节: 03
状态: 完成
---

# 第3章 框架层：用 Python 定义模型，把底层留给框架

> 导航：[第2章 算子层](../02-算子层/README.md) → **第3章 框架层** → [第4章 训练层](../04-训练层/README.md)

> 一句话：框架是菜谱，算子是刀工，硬件是灶台。你只需要写"做什么菜"，剩下的自动微分、kernel 调度、显存管理全交给框架。

---

## 🥢 类比开场：框架 = 菜谱和厨具

延续餐厅类比：

| 层 | 餐厅角色 | 对应本章 |
|---|---|---|
| 硬件层（第2章） | 灶台和锅 | NPU、GPU、CANN kernel |
| 算子层 | 切菜、炒菜的具体动作 | GEMM、FlashAttention |
| **框架层（本章）** | **菜谱 + 后厨主管** | **PyTorch 定义模型、自动求梯度、调度算子** |

没人会为了炒一盘菜先去学打铁（写 CUDA）。框架层让你用高层 Python 描述模型，把"翻译成 kernel、管理显存、求梯度"这些脏活全部接管。**但菜谱写得再好，刀工（算子）不行菜还是慢**——这就是上一章的意义。

---

## 1. 问题：不想写 CUDA/CANN，想用 Python 定义模型

为了 `y = xWᵀ + b` 这一行数学，手写底层代码要做三件痛苦的事：

1. **内存管理**：`cudaMalloc` / `aclrtMalloc` 手动分配、搬运、释放显存
2. **kernel 启动**：grid / block / 线程坐标换算，错了就越界
3. **算子实现**：矩阵乘的 tiling、共享内存、向量化，性能还要调

更致命的是第 4 件：**求梯度**。每个算子的反向传播（backward）也得手写一份，模型结构一变，forward + backward 全部重来。

**框架层（framework layer）** 就是来解决这个问题的：

```
你的 Python 模型定义（forward 只写一遍）
    │
    ▼
框架层：自动微分（Autograd）+ 计算图 + 算子调度 + 内存管理
    │
    ▼
算子层 / 硬件层（第2章的 kernel 在 NPU/GPU 上跑）
```

**昇腾适配**：PyTorch 通过 `torch_npu` 插件支持昇腾，API 与 `torch.cuda` 几乎一致：
```python
# GPU 版
device = torch.device("cuda")
# NPU 版（昇腾）
device = torch.device("npu")
```

---

## 2. 自动微分原理详解 ⭐⭐

### 2.1 计算图（Computation Graph）

框架把你写的每一行张量运算记录成一张**有向无环图（DAG）**。节点是数据（张量），边是算子：

```
x ──┐
    ├─▶ Linear(W, b) ──▶ pred ──▶ MSE ──▶ loss
W ──┘                            ▲
b ───────────────────────────────┘
```

- **正向传播（forward）**：从左到右执行，算出结果，同时把"每个算子 + 它的输入输出"存进图里
- **反向传播（backward）**：从 loss 出发，**从右到左**用链式法则求每个叶子节点（参数）的梯度

### 2.2 链式法则：自动微分的数学心脏

复合函数 `loss = MSE(Linear(x))` 对 W 求导，就是链式法则逐层回传：

```
∂loss/∂W = ∂loss/∂pred · ∂pred/∂W
             ↑              ↑
          损失层回传    线性层局部导数
```

PyTorch 里每个算子都内置了自己的 `backward()`（局部导数），反向传播就是把所有局部导数**相乘**并沿图回传。这就是为什么"**你只写 forward，梯度全自动**"——backward 是框架替你写好的。

一个具体的数值例子（单样本、无偏置）：

```
x = 2,  W = 3,  y_true = 10
pred = x·W = 6
loss = (pred - y_true)² = 16

∂loss/∂pred = 2·(pred - y_true) = -8
∂pred/∂W   = x = 2
∂loss/∂W   = -8 × 2 = -16     ← 链式法则相乘，梯度下降将 W 往 -16 方向更新
```

### 2.3 动态图 vs 静态图对比表

| 维度 | 动态图（PyTorch 默认） | 静态图（TensorFlow 2.x graph mode / MindSpore graph） |
|---|---|---|
| 建图时机 | **define-by-run**：边执行边建图 | **define-and-run**：先构图、后执行 |
| 调试 | 像写普通 Python，print/断点随意 | 图先编译，报错定位难 |
| 灵活性 | 随意 if/for/动态形状 | 控制流需特殊语法，形状需静态 |
| 运行时开销 | 每步重建图，有开销 | 图复用，开销低 |
| 编译优化 | 受限（可用 `torch.compile` 补救） | 算子融合、内存规划空间大 |
| 部署 | 需导出（TorchScript/ONNX） | 天然适合 Serving（SavedModel） |
| 上手难度 | 最低 | 中 |

**一句话：动态图好调试，静态图好优化。** PyTorch 选了好调试，所以赢了研究圈；TensorFlow/MindSpore 的静态图在部署和 NPU 编译优化上有优势。现代框架在互相靠拢：PyTorch 有 `torch.compile`，TensorFlow 2.x 默认 eager，MindSpore 也提供 PyTorch 式动态图模式。

---

## 3. PyTorch 核心概念详解 ⭐

五个概念构成你 90% 的日常：`Tensor` → `Parameter` → `Module` → `Optimizer` → `DataLoader`。它们的关系：

```
DataLoader 供数据 → Tensor 存数据 → Module 里的 Parameter 是要学的参数
                                          ↓
                        Optimizer 拿着梯度更新 Parameter
```

### 3.1 Tensor：数据的容器

```python
import torch
x = torch.randn(3, 4)          # 3×4 随机张量（CPU）
x = x.to("cuda")               # 搬到 GPU（或者 .to("npu") 搬到昇腾）
x.requires_grad_(True)         # 告诉框架：我要对这个张量求梯度
```

要点：`requires_grad=True` 的张量会成为计算图的叶子节点，`backward()` 后它的 `.grad` 属性里就是梯度。**设备是张量的属性**——CPU/GPU 张量不能直接做运算（见 7.3 设备不匹配）。

### 3.2 Parameter：可学习的 Tensor

```python
weight = torch.nn.Parameter(torch.randn(2, 1))   # 会被 Module 自动登记
```

`Parameter` 就是 `Tensor` + "请自动登记我"标记。放在 `nn.Module` 里，`model.parameters()` 就能收集到它，优化器才知道该更新谁。普通 Tensor 不登记，`parameters()` 里没有它。

### 3.3 Module：模型的基本单元

```python
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_f, out_f))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return x @ self.weight + self.bias   # 只写 forward，backward 自动
```

`Module` 干三件事：① 递归登记子模块和 Parameter；② 调用 `model(x)` 时自动执行 `forward()` 并建计算图；③ 提供 `.to(device)`、`.train()/.eval()`、`.state_dict()` 等统一接口。**任何模型都是 Module 的嵌套组合**。

### 3.4 Optimizer：更新参数的策略

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # 随机梯度下降
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 自适应学习率，实际更常用
optimizer.zero_grad()   # ① 清空上轮梯度（不清会累加！）
loss.backward()         # ② 反向传播，算出梯度
optimizer.step()        # ③ 用梯度更新参数：W ← W - lr·grad
```

三种主流优化器对比：

| 优化器 | 原理 | 适合场景 |
|---|---|---|
| SGD | 沿梯度反方向走固定步长 | 简单任务、需要精细调 lr |
| SGD + Momentum | 累积历史方向，冲过局部极小 | 训练更稳、更快收敛 |
| Adam / AdamW | 每参数自适应学习率 + 动量 | **大模型默认选择**，几乎免调参 |

### 3.5 DataLoader：数据流水线

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)          # ① 定义"数据在哪"：样本+标签打包
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
for x_batch, y_batch in loader:        # ② 迭代即得一批数据
    ...                                # ③ num_workers>0 用多进程预取，避免 GPU 空等
```

`DataLoader` 解决三个问题：**批处理**（一次喂 32 个样本）、**打乱**（`shuffle=True` 避免模型学到顺序）、**预取**（CPU 提前加载下一批，GPU 不闲置）。

---

## 4. 三大框架详细对比 ⭐⭐

### 4.1 语法对比：同一个模型，三种写法

`y = xWᵀ + b`，训练一步：

```python
# ── PyTorch：动态图，命令式，最接近原生 Python ──
import torch, torch.nn as nn
model = nn.Linear(2, 1)
loss = nn.MSELoss()(model(X), y)
loss.backward()          # 就地更新梯度
```

```python
# ── TensorFlow 2.x：Keras 高层 API，eager + tf.function ──
import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
with tf.GradientTape() as tape:          # 显式"录音"上下文
    loss = tf.keras.losses.MSE(y, model(X))
grads = tape.gradient(loss, model.trainable_variables)
tf.keras.optimizers.SGD(0.1).apply_gradients(zip(grads, model.trainable_variables))
```

```python
# ── MindSpore：静态图为主，函数式风格 ──
import mindspore as ms
from mindspore import nn, ops

class Net(nn.Cell):                      # 基类叫 Cell，不是 Module
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(2, 1)
    def construct(self, x):              # 注意：方法名叫 construct，不是 forward
        return self.fc(x)

model = Net()
loss_fn = nn.MSELoss()
optimizer = nn.SGD(model.trainable_params(), 0.1)

def forward_fn(x, y):                    # 训练逻辑包成一个函数交给图编译
    loss = loss_fn(model(x), y)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)  # 显式求梯度
```

### 4.2 生态对比

| 维度 | PyTorch | TensorFlow | MindSpore |
|---|---|---|---|
| 出品方 | Meta | Google | 华为 |
| 研究社区 | **事实标准**，论文几乎全用它 | 逐渐退守生产 | 昇腾生态内为主 |
| 模型库 | HuggingFace `transformers` 原生支持 | 需转换权重 | 有 ModelZoo，数量少 |
| 工具链 | Lightning / DeepSpeed / vLLM 全系支持 | TFX / SavedModel | MindFormers / CANN 深度集成 |
| 硬件支持 | NVIDIA 最佳，AMD/昇腾需适配层 | NVIDIA + TPU | **昇腾 NPU 原生**（核心卖点） |
| 中文资料 | 最多 | 多 | 华为官方文档中文友好 |

### 4.3 部署对比

| 维度 | PyTorch | TensorFlow | MindSpore |
|---|---|---|---|
| 部署格式 | TorchScript / ONNX / `torch.export` | SavedModel（服务端标准） | MindIR（昇腾部署标准） |
| 推理服务 | vLLM / Triton（业界主流） | TF Serving | MindSpore Serving / MindIE |
| 硬件部署 | 先 NVIDIA，昇腾走 ONNX→CANN | NVIDIA/TPU | **昇腾 NPU 一条龙**（910B 推理性价比高） |
| 转静态图 | `torch.compile`（v2.0+） | `tf.function` | 默认就是静态图 |

### 4.4 怎么选（决策流）

```
要训大模型/跟研究前沿？  ──► PyTorch（没得选，生态在这）
国产化/昇腾 NPU 部署？  ──► MindSpore（原生支持）或 PyTorch + ONNX 迁移
已有 TF 生产管线？      ──► 留在 TensorFlow，别为换而换
```

> MindSpore 的 GitHub ⭐ 低有个背景：主仓库在华为 Gitee，GitHub 只是镜像。选它的核心理由只有一个——**你要用昇腾 NPU**。

---

## 5. LLM-Engineers-Handbook 分层架构详解 ⭐⭐

[LLM-Engineers-Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)（Packt 出品）是社区公认的 LLM 工程最佳实践参考。它的 `llm_engineering/` 包把整个工程拆成**四层**，按"职责变化频率"隔离：

```
llm_engineering/
├── domain/          # ① 领域层：业务世界的"名词"
│   ├── base/        #    实体基类、向量数据库基类（抽象接口）
│   ├── documents/   #    文档实体（Article、Podcast 等数据类型）
│   ├── exceptions/  #    自定义异常
│   └── types.py     #    共享类型
├── application/     # ② 应用层：业务世界的"动词"
│   ├── cbr/         #    案例推理（Case-Based Reasoning）用例
│   ├── llm/         #    LLM 调用封装（提示词、结构化输出）
│   ├── preprocessing/#   文档清洗/分块
│   └── rag/         #    RAG 检索增强生成编排
├── infrastructure/  # ③ 基础设施层：外部系统怎么接
│   ├── chroma/      #    向量数据库
│   ├── minio/       #    对象存储（S3 兼容）
│   ├── postgres/    #    关系数据库
│   └── qdrant/      #    向量数据库（生产级）
├── model/           # ④ 模型层：ML 模型生命周期
│   ├── evaluation/  #    评测：训完怎么打分（RAGAS 等）
│   ├── finetuning/  #    微调：LoRA 等训练逻辑
│   └── inference/   #    推理：模型怎么对外服务
├── __init__.py
└── settings.py      # 全局配置（API key、路径等）
```

### 每层职责与依赖方向

| 层 | 回答的问题 | 变化频率 | 依赖谁 |
|---|---|---|---|
| `domain` | 业务里有什么（文档、实体、接口抽象） | 低 | 不依赖其他层 |
| `application` | 业务做什么（检索、生成、预处理） | 中 | 依赖 domain + infrastructure |
| `infrastructure` | 外部系统怎么接（向量库、存储、DB） | 中 | 依赖 domain 的抽象接口 |
| `model` | 模型怎么定义/训练/部署 | 高 | 相对独立 |

**核心设计思想：依赖倒置。** `domain` 定义抽象接口（如 `VectorDB` 基类），`infrastructure` 提供实现（Chroma/Qdrant 各写一个），`application` 只面向接口编程——换向量库时只改 infrastructure，业务代码一行不动。这正是"**框架层帮我们隔离变化**"在工程上的延伸。

### 映射到本章

本章的 `linear_model.py` 只用到 `model` 层的子集（定义 + 训练），但目录结构照抄它的思路：**定义模型（model/）、训练（finetuning/）、部署（inference/）、评测（evaluation/）是四个不同生命周期，混在一个文件里，模型一改全崩。**

---

## 6. 动手：完整代码示例（自定义 Module + 训练 + 保存/加载）⭐⭐

### 6.1 模型定义：自定义 Module

```python
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):                 # 自定义 Module：两层线性 + ReLU
    def __init__(self, in_features: int, hidden: int = 16, out_features: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()                 # 激活函数：给模型非线性能力
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))
```

没有激活函数的话，两层线性叠加还是线性（`W₂(W₁x) = (W₂W₁)x`），学不了非线性规律——ReLU 就是给模型"掰弯"能力的那个弯。

### 6.2 训练循环（DataLoader + Optimizer）

```python
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
# 造数据：y = 3x₀ - 2x₁ + 1 + 噪声（噪声让任务更像真实世界）
X = torch.randn(1000, 2)
y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.1 * torch.randn(1000)

dataset = TensorDataset(X, y.unsqueeze(1))
loader = DataLoader(dataset, batch_size=64, shuffle=True)   # 3.5 节的概念上场

model = TwoLayerNet(in_features=2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)   # Adam：大模型默认

for epoch in range(100):                        # 扫完整数据 100 遍
    epoch_loss = 0.0
    for x_batch, y_batch in loader:             # 每批 64 个样本
        optimizer.zero_grad()                   # ① 清空上轮梯度
        pred = model(x_batch)                   # ② 正向：建计算图
        loss = loss_fn(pred, y_batch)           # ③ 算损失
        loss.backward()                         # ④ 反向：自动微分
        optimizer.step()                        # ⑤ 更新参数
        epoch_loss += loss.item() * len(x_batch)
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}, loss = {epoch_loss / len(dataset):.4f}")
```

**训练循环的五个固定步骤**（背下来，所有训练都长这样）：`zero_grad → forward → loss → backward → step`。

### 6.3 保存 / 加载模型

```python
# ── 保存：只存参数字典（推荐），不存整个对象 ──
torch.save(model.state_dict(), "model.pt")

# ── 加载：先建同结构模型，再灌参数 ──
loaded = TwoLayerNet(in_features=2)
loaded.load_state_dict(torch.load("model.pt", weights_only=True))
loaded.eval()                                   # 切到推理模式（关 dropout/bn 统计）

# ── 推理 ──
with torch.no_grad():                           # 不建计算图：省显存、更快
    pred = loaded(torch.tensor([[1.0, 2.0]]))
print("预测:", pred.item())
```

两个易错点：① 加载时**必须**先实例化同结构的模型，再 `load_state_dict`，没有"凭空恢复"；② 推理务必包在 `torch.no_grad()` 里，否则每步都建计算图、白占显存。

### 6.4 你会看到

```
epoch   0, loss = 2.4143
epoch  20, loss = 0.0195
epoch  40, loss = 0.0125
...
预测: -0.1234        # 接近真实值 3×1 - 2×2 + 1 = 0
```

loss 从 ~2.4 一路掉到 ~0.01，说明模型学到了 `y = 3x₀ - 2x₁ + 1`。如果 loss 是 `nan` 或完全不降，跳第 7 节。

---

## 7. 排障案例 ⭐⭐

框架层排障的权威手册是 [ml-engineering](https://github.com/stas00/ml-engineering) 的 [`debug/pytorch.md`](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md)。三个最高频问题：

### 7.1 案例一：梯度消失 / 爆炸（loss = nan 或 loss 不动）

**症状**：训练几轮后 loss 变 `nan`，或 loss 卡在某个值完全不动。

**根因**：链式法则是**连乘**——`∂loss/∂W = ∏ 各层局部导数`。层多了，每层导数 <1 就连乘趋近 0（**梯度消失**，深层权重学不动）；每层导数 >1 就连乘爆炸（**梯度爆炸**，数值溢出成 `inf/nan`）。

**排查步骤**：

```python
# ① 打印每层梯度，找出"出事的那一层"
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.abs().mean().item())
# 梯度全 0 → 消失；出现 inf/nan → 爆炸
```

**解法**（按优先级）：

| 手段 | 说明 |
|---|---|
| 调小学习率 | 最常见原因：lr 太大直接炸 |
| 梯度裁剪 `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | 把梯度范数压到上限，爆炸的保险丝 |
| 换激活函数 | ReLU 缓解消失（sigmoid/tanh 深了必消失） |
| 归一化层 | BatchNorm/LayerNorm 稳住每层分布 |
| 检查数据 | 特征有 `nan`/`inf` 或未归一化（尺度过大）也会炸 |

### 7.2 案例二：显存泄漏 / CUDA out of memory

**症状**：训练几轮后突然 `RuntimeError: CUDA out of memory`，或显存占用随训练时间**单调上涨**（泄漏）。

**根因**（按概率）：

1. **batch size 太大**，激活值撑爆——最直接
2. **别的进程占着显存**：`nvidia-smi` 看 `Memory-Usage`，常见是上一个训练任务没 kill 干净
3. **泄漏三件套**：训练循环里每步都在新建张量/模型没清理；`loss.item()` 忘调导致 Python 持有计算图；推理时没包 `no_grad()`

**排查步骤**：

```bash
nvidia-smi                              # ① 看显存被谁占
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
kill -9 <pid>                           # ② 清掉残留进程
```

```python
# ③ 代码自查：训练循环里是否有"每步新建但从不释放"的对象
# 正确姿势：把累加的数字 .item() 化（脱离计算图），别让 loss 张量链一直活着
epoch_loss += loss.item()               # 而不是 epoch_loss += loss
```

**解法**：调小 `batch_size` → 梯度累积（accumulate 几步再 step）→ `model.gradient_checkpointing_enable()`（用时间换显存）→ 换大卡。

### 7.3 案例三：设备不匹配

**症状**：报错 `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`（或 `Tensor with baseType Float does not have the correct type`）。

**根因**：模型在 GPU，输入张量在 CPU（或反过来）。CPU 和 GPU 张量不能直接运算——这和"内存与显存不能直接互换"是同一件事。

**解法**：统一用 `device` 变量，别硬编码：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                          # 模型
x = x.to(device)                                  # 输入
y = y.to(device)                                  # 标签（最容易忘！）
```

**规律**：报错说的是哪一行，就检查那一行涉及的**所有**张量（含标签 `y`）和设备。`.to(device)` 是个无脑但有效的习惯——模型、输入、标签三件套全搬。

---

## 8. 动手实验 ⭐

每实验 5 分钟，改一行看效果，建立直觉：

**实验 1：学习率过大会怎样？**
把 6.2 的 `lr=1e-2` 改成 `1e-1` → loss 变 `nan`（梯度爆炸）；改成 `1e-5` → loss 几乎不动（学太慢）。结论：**lr 是训练里最敏感的超参**。

**实验 2：没有 ReLU 会怎样？**
把 `self.relu = nn.ReLU()` 从 6.1 的 forward 里删掉（`return self.fc2(self.fc1(x))`）→ 数据是线性的所以 loss 还能降，但换成 `y = 3*x0**2 - 2*x1 + 1`（非线性数据）就会 loss 卡住不降。结论：**激活函数给模型非线性能力**。

**实验 3：shuffle 关掉会怎样？**
把 6.2 的 `shuffle=True` 改成 `False` → 数据本身随机所以影响小；换成有序数据（前 500 行 y=1，后 500 行 y=0）就会训练震荡、收敛变慢。结论：**DataLoader 打乱防止模型学到批次顺序**。

**实验 4：保存的模型能复用吗？**
跑完 6.3 后，新建一个 Python 会话，只跑"加载 + 推理"部分 → 预测结果和训练会话里一致。结论：**`state_dict` 是模型唯一的持久化形态**，部署就是"加载权重 + forward"。

**实验 5（可选，有昇腾环境）**：把 `model.to(device)` 的 device 换成 `npu`，体验框架层"写一次、多硬件跑"的承诺。

---

## 9. 自测题

**第 1 题** ⭐：用你自己的话解释"为什么说 `loss.backward()` 是框架层的核心价值"？

<details>
<summary>参考答案</summary>

因为你**只写了 forward**，反向传播（每个算子的局部导数 × 链式法则连乘）是框架根据计算图自动完成的。手写 CUDA 时 backward 要和 forward 一样再写一遍，模型一改全重来——自动微分把这个成本降到了零。
</details>

**第 2 题** ⭐⭐：动态图和静态图各有什么优缺点？为什么 PyTorch 赢了研究圈，而 MindSpore 在昇腾上有优势？

<details>
<summary>参考答案</summary>

动态图（define-by-run）边执行边建图，像写普通 Python，好调试、灵活，代价是运行时开销；静态图先构图再执行，编译优化空间大（算子融合、内存规划）、部署高效，但调试难、灵活度低。研究圈要的是快速迭代和调试体验，所以 PyTorch 赢了；MindSpore 的静态图在昇腾 NPU 上能被深度编译优化并与 CANN 原生打通，所以目标平台是昇腾时它更合理。
</details>

**第 3 题** ⭐⭐：训练循环里 `optimizer.zero_grad()` 忘写会发生什么？为什么？

<details>
<summary>参考答案</summary>

梯度会**累加**。`backward()` 是把新梯度**加到** `.grad` 上而不是覆盖，不清零的话第二步的梯度是前两步之和，第三步是前三步之和……参数更新方向越来越偏，loss 震荡甚至发散。所以五步循环 `zero_grad → forward → loss → backward → step` 一步都不能少。
</details>

**第 4 题** ⭐⭐⭐：你的模型在 GPU 上训练，loss 在第 10 轮变成 `nan`。写出完整的排查步骤（至少 3 步，按优先级）。

<details>
<summary>参考答案</summary>

1. **检查数据**：特征或标签里有没有 `nan`/`inf`（`torch.isnan(X).any()`），数据本身脏的话后面全白查；
2. **调小学习率**：lr 太大是最常见原因，先除以 10 试跑几轮；
3. **定位出事层**：遍历 `model.named_parameters()` 打印每层 `grad` 的均值/范数，找到第一个 `inf/nan` 的层，确认是梯度爆炸（加 `clip_grad_norm_` 裁剪）还是某层数值问题；
4. **加梯度裁剪**：`nn.utils.clip_grad_norm_(model.parameters(), 1.0)`；
5. 还不行就检查损失函数和标签范围（回归任务标签未归一化、尺度过大也常见）。
</details>

---

## 🍳 类比回顾：框架 = 菜谱

- **硬件层**（第2章）= 厨房和灶台：锅（GPU）、火（CUDA kernel）都在这里
- **算子层** = 刀工火候：同样的菜，好刀工快 10 倍
- **框架层** = 菜谱 + 后厨主管：告诉你**做什么菜、按什么步骤**（定义模型、正向、反向），自动微分是"自动尝咸淡调整火候"，你不用关心灶台怎么点火
- **PyTorch 是川菜菜谱**：步骤灵活（动态图），边炒边尝、随时改配方（好调试），会做的人多、菜谱（教程/生态）最全——所以大家默认用它
- **TensorFlow 是老派宴会菜谱**：讲究提前备料、流程固定（静态图），适合大酒店（生产部署）标准化出菜
- **MindSpore 是私房菜谱**：只有自家灶台（昇腾 NPU）能完全发挥，但如果你就用这口锅，它最顺手

## 参考链接

- [PyTorch 官方 60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch autograd 机制官方文档](https://pytorch.org/docs/stable/notes/autograd.html)
- [ml-engineering — debug/pytorch.md（排障手册）](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md)
- [LLM-Engineers-Handbook（四层架构参考）](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [MindSpore 官方文档](https://www.mindspore.cn/)
- [TensorFlow 官方教程](https://www.tensorflow.org/tutorials)

---

> 下一章预告：训练层——从单机训练到多卡并行，DeepSpeed 与分布式训练是怎么把模型塞进显存的。
