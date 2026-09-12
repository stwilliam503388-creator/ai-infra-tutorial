---
title: 第14章 模型部署进阶
tags: ['ai-infra', '教程', '部署', 'kserve']
章节: 14
状态: 完成
---

# 第14章：模型部署进阶

> 模型部署 = 开连锁店。单店（本地）能赚钱，但想规模化，需要标准化流程、统一管理和自动扩缩容。

## 14.1 问题：生产环境怎么部署模型服务？

你在本地跑通了模型，但：
- 怎么让 1000 个用户同时访问？
- 服务器挂了怎么办？
- 怎么自动扩容应对流量高峰？

**核心矛盾**：本地能跑 ≠ 生产可用。

## 14.2 方案：三种部署方式

| 方式 | 一句话 | 适合场景 |
|------|--------|---------|
| **Kserve** | K8s 原生模型服务 | 云原生、自动扩缩容 |
| **Triton** | NVIDIA 推理服务器 | 高性能、多框架 |
| **Seldon Core** | 模型服务网格 | A/B 测试、灰度发布 |

## 14.3 项目

| 项目 | ⭐ | 定位 | 昇腾支持 |
|------|-----|------|---------|
| **Kserve** | 4k | K8s 模型服务 | ✅ 支持昇腾 device plugin |
| **Triton** | 11k | NVIDIA 推理服务器 | ❌ CUDA 专用（昇腾用 MindIE） |
| **Seldon Core** | 4k | 模型服务网格 | ⚠️ 需自定义 |
| **Istio** | 36k | 服务网格 | ✅ 硬件无关 |
| **MindIE** | 闭源 | 华为官方推理引擎 | ✅ 昇腾原生 |

## 14.4 代码示例

### Kserve 部署 vLLM

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-qwen
spec:
  predictor:
    containers:
    - name: vllm
      image: vllm/vllm-openai:latest
      args:
      - --model
      - Qwen/Qwen2-0.5B-Instruct
      - --port
      - "8000"
      ports:
      - containerPort: 8000
      resources:
        limits:
          nvidia.com/gpu: 1
          memory: 16Gi
        requests:
          nvidia.com/gpu: 1
          memory: 16Gi
```

### Triton 配置

```protobuf
# config.pbtxt
name: "qwen"
platform: "python"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

## 14.5 排障：生产环境常见问题

| 问题 | 现象 | 根因 | 解法 |
|------|------|------|------|
| 模型加载超时 | Pod 一直 Pending | 模型太大，下载慢 | 用 InitContainer 预下载 |
| GPU 分配失败 | 0/1 nodes available | GPU 资源不足 | 检查节点 GPU 数量 |
| 自动扩缩容失效 | 流量高峰时服务崩溃 | HPA 配置错误 | 检查指标和阈值 |
| 冷启动慢 | 第一个请求超时 | 模型加载需要时间 | 用 readinessProbe 延迟就绪 |

## 14.6 动手实验：用 Kserve 部署一个模型

**目标**：在 K8s 上部署一个 vLLM 服务。

**步骤**：

```bash
# 1. 安装 Kserve
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml

# 2. 创建 InferenceService
kubectl apply -f vllm-qwen.yaml

# 3. 查看状态
kubectl get inferenceservices
kubectl get pods

# 4. 测试
kubectl port-forward svc/vllm-qwen-predictor 8000:80
curl http://localhost:8000/v1/models
```

**预期输出**：
- Pod 状态从 `Pending` → `ContainerCreating` → `Running`
- `curl` 返回模型列表

**常见错误**：
- GPU 资源不足 → 检查 `nvidia.com/gpu` 资源配额
- 镜像拉取失败 → 检查网络、用国内镜像

## 14.7 自测题

1. Kserve 和 Triton 的区别是什么？（提示：K8s 原生 vs 独立服务器）

<details>
<summary>参考答案</summary>

Kserve 是 K8s 原生，用 YAML 声明式部署，自动扩缩容；Triton 是独立推理服务器，性能更高但需要自己管理扩缩容。Kserve 底层可以用 Triton 作为推理引擎。
</details>

2. 什么是冷启动问题？（提示：模型加载需要时间）

<details>
<summary>参考答案</summary>

模型服务启动时需要加载模型到显存，这个过程可能需要 10-60 秒。如果流量来了但模型还没加载完，请求会超时。解法：用 readinessProbe 延迟就绪状态，或用预热请求。
</details>

3. HPA（Horizontal Pod Autoscaler）的作用是什么？（提示：自动扩缩容）

<details>
<summary>参考答案</summary>

HPA 根据 CPU/内存/自定义指标自动调整 Pod 副本数。流量高峰时自动扩容，低峰时自动缩容，节省成本。
</details>

---

**上一章**：[第13章：模型评测](../13-模型评测/README.md) | **下一章**：[第15章：MLOps](../15-MLOps/README.md)
