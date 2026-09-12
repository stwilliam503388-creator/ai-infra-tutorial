---
title: 第6章 编排层——100 台服务器怎么管理？
tags: [ai-infra, 教程, 编排, kubernetes, volcano, ray]
章节: 06
状态: 完成
---

# 第6章 编排层

> 导航：[第5章 推理层](../05-推理层/README.md) → **第6章 编排层** → [第7章 应用层](../07-应用层/README.md)

> 一句话：容器解决"环境一致性"，调度器解决"资源分配"。编排层把 100 台机器变成一台"按需取用"的巨型计算机。

---

## 1. 问题：100 台服务器怎么管理？

第 5 章我们在单机（或单机多卡）上把训练跑通了。现在把镜头拉远：

- 你有 **100 台服务器**，每台 8 张 GPU，一共 800 张卡；
- 10 个团队、30 个任务同时在跑：微调、推理服务、数据预处理、评测；
- 有人提交了一个要 64 张卡的训练任务，**它该落在哪台机器上？**
- 一台机器挂了，上面跑的 5 个任务怎么办？
- 新加 20 台机器，要不要改代码？

如果靠人肉管理，答案是：一个 24 小时待命的运维团队，和一堆贴满机房的便利贴。这显然不 scale。

**核心矛盾**：计算资源（GPU/CPU/内存）是共享的、异构的、会故障的；而任务是多变的、有依赖的、有优先级的。中间缺一个"资源分配与任务调度"的大脑——这就是**编排层（Orchestration Layer）**要解决的问题。

---

## 2. 方案：容器化 + 调度器

编排层由两个半部分组成：

### 2.1 容器化（打包）

> 类比先行：容器 = 标准集装箱。货柜船能一小时装卸上千个箱子，不是靠码头工人力气大，而是因为**箱子尺寸统一、内容自包含**。

- **镜像（Image）**：把代码 + 运行环境 + 依赖打成一个只读包。参考 LLM-Engineers-Handbook 的 `Dockerfile`：`python:3.11-slim` 基础镜像 → 装 Chrome / 系统依赖 → 用 Poetry 装 Python 依赖 → `COPY` 代码。这样"在我机器上能跑"彻底消失。
- **容器（Container）**：镜像的运行实例，进程级隔离（namespace + cgroup），启动秒级。
- **为什么训练/推理也要容器**：CUDA 版本、PyTorch 版本、算子库（见第 3 章）互相打架，容器把"环境"和"机器"解耦——任务跟着镜像走，而不是跟着某台机器走。

### 2.2 调度器（分配）

调度器回答三个问题：

| 问题 | 含义 |
|---|---|
| **放哪？** | 根据资源需求（GPU 数、显存、CPU）和节点现状（已用/空闲、拓扑）选一台机器（Placement） |
| **何时跑？** | 排队策略、优先级、抢占（Scheduling） |
| **挂了怎么办？** | 健康检查、重启、迁移（Self-healing） |

### 2.3 两种调度哲学

- **K8s 系（声明式）**：你告诉系统"我要 64 张卡跑这个镜像"，系统负责让现实向声明收敛（`kubectl apply` → 控制器不断调谐 reconcile）。
- **SLURM 系（批处理）**：你提交一个脚本作业（`sbatch`），调度器按队列和分区分配节点，作业跑完即走。HPC 界（包括很多 AI 训练集群）的经典答案。

> ⚠️ 业界真实分歧（来自 ml-engineering 作者）：训练任务用 SLURM 更顺手——它基于 Unix，多机协调自然；K8s 更擅长**推理服务**这种长驻、弹性、多副本的负载，但对训练任务来说"重造轮子"、缺 Unix 特性、需要大团队伺候。选型别跟风，先想清楚你主要是**跑作业**还是**跑服务**。

---

## 3. 项目：三个档次的答案

| 项目 | Stars | 定位 | 什么时候选它 |
|---|---|---|---|
| **Kubernetes** | ⭐125k | 通用容器编排的事实标准：部署、服务发现、弹性伸缩、自愈 | 推理服务、微服务、长驻负载；集群已有 K8s 团队 |
| **Volcano** | ⭐5.9k | K8s 上的**批处理/AI 作业调度器**（华为云开源、已捐赠 CNCF）：队列、优先级、抢占、Gang Scheduling，支持 GPU/NPU 异构资源 | 在 K8s 上跑训练作业（PyTorchJob、TFJob 等）；**昇腾 NPU 集群的事实标准调度器**（华为云 ModelArts/CCE 底层就是它） |
| **Ray** | ⭐43.6k | 分布式 Python 计算框架：把单机 Python 代码变成集群分布式任务，自带调度、对象存储、容错 | 数据预处理、RL、超参搜索、Serve（推理）；不想碰 K8s 复杂度 |

**怎么理解三者的关系**：

- **K8s** 是操作系统（管容器）；**Volcano** 是 K8s 上的批处理调度器（管训练作业）；**Ray** 是更上层的应用框架（管 Python 任务图）。
- 一条常见链路：**Ray 跑在 K8s 上**（KubeRay），K8s 管 Ray 头节点和工作节点的生命周期，Ray 内部再管 Python 任务的调度。
- 纯训练集群的另一个常见形态：**SLURM 管节点 + PyTorch 的 `torch.distributed.run` 做多机协调**（见第 5 章）。

---

## 4. Kubernetes 核心概念详解

### 4.1 集群结构：控制面 + 数据面

K8s 分两个平面，先建立全局印象：

| 平面 | 组件 | 职责 |
|---|---|---|
| **控制面**（大脑） | API Server | 所有操作唯一的入口（RESTful），`kubectl` 的一切命令都打到它 |
| | etcd | 集群状态的"数据库"，所有声明和现状都存在这里 |
| | Scheduler | 决定新 Pod 落在哪个节点（就是第 2 节说的"放哪"） |
| | Controller Manager | 一堆控制器的家：Deployment 控制器、Node 控制器……不断"调谐"让现状向声明收敛 |
| **数据面**（肌肉） | kubelet | 每台节点上的代理，负责拉起/销毁本节点的 Pod，向控制面汇报心跳 |
| | kube-proxy | 实现 Service 的网络规则（VIP、负载均衡） |
| | 容器运行时 | containerd / docker，真正把容器跑起来 |

> 记忆口诀：**API Server 是唯一入口，etcd 是唯一真相，Controller 让现实对齐声明，kubelet 是节点的代言人。**

### 4.2 Pod：最小调度单元

- Pod 是 K8s 里**能调度的最小单位**：一个或多个容器共享同一个网络命名空间（同一个 IP）、同一份存储卷。
- Pod 是**临时**的：它可能因为节点故障、资源驱逐、手动删除而随时消失，IP 也会变。**永远不要直接依赖 Pod 的 IP**。
- 多容器 Pod 常见于"边车（sidecar）"模式：比如主容器是 vLLM，边车容器负责采集日志/指标。

### 4.3 Service：稳定的访问入口

Pod 会死、会重建、IP 会变，那客户端怎么找到服务？——**Service**。

- Service 通过 `selector` 选中一组 Pod（按 label 匹配），提供一个**稳定的虚拟 IP + DNS 名字**，并把流量负载均衡到这些 Pod 上。
- 类型三种，从内到外：
  - `ClusterIP`（默认）：集群内部可达，外部访问不到；
  - `NodePort`：在每个节点上开一个端口（如 30000-32767）暴露到集群外；
  - `LoadBalancer`：云厂商（AWS/阿里云）给一个公网 IP，流量先到负载均衡器再进 NodePort。

### 4.4 Deployment：无状态应用的控制器

Deployment 是最常用的工作负载类型，管的是"无状态"应用（哪个副本处理请求都一样）：

| 能力 | 说明 |
|---|---|
| 声明式副本数 | `replicas: 2`，多了砍掉、少了补齐，永远向声明收敛 |
| 滚动更新 | 新版本逐个替换旧副本，`kubectl rollout status` 看进度 |
| 回滚 | `kubectl rollout undo` 一键回到上个版本 |
| 自愈 | 容器挂了/节点挂了 → 控制器自动重建新 Pod |
| 弹性伸缩 | 配合 HPA（Horizontal Pod Autoscaler）按 CPU/自定义指标自动加减副本 |

### 4.5 StatefulSet：有状态应用的控制器

数据库、向量库（qdrant）、消息队列这类"每个副本有身份"的应用用 StatefulSet：

- 每个副本有**稳定的名字**：`qdrant-0`、`qdrant-1`……重建后名字不变；
- 每个副本有**独立的存储**：`volumeClaimTemplates` 自动为每个副本创建一个 PVC（对应 compose 的 `volumes`）；
- **有序**启停：扩容时逐个启动、缩容时从末尾逐个删。

### 4.6 ConfigMap / Secret：配置与密钥

| 类型 | 存什么 | 特点 |
|---|---|---|
| **ConfigMap** | 非敏感配置（URL、开关、超时时间） | 明文，注入方式：环境变量 / 挂载成文件 |
| **Secret** | 敏感信息（API Key、HF_TOKEN、密码） | 值要求 base64 编码，K8s 内部加密存储（etcd 加密可选） |

> 最佳实践：镜像里**不要**烧配置和密钥。镜像只认 ConfigMap/Secret 注入的变量——同一个镜像，测试环境一套配置、生产一套配置，镜像不用重新构建。

### 4.7 一个 Pod 的一生（状态机）

| 状态 | 含义 | 常见卡住原因 |
|---|---|---|
| `Pending` | 已创建但还没跑起来（在排队或等资源） | 资源不足、GPU 调度器没装、镜像拉取中 |
| `Running` | 容器已启动 | —— |
| `CrashLoopBackOff` | 启动就崩，反复重启 | 代码异常、探针太苛刻 |
| `OOMKilled` | 内存超限被杀 | 内存 limit 太小 |
| `Succeeded` / `Failed` | 一次性任务（Job）跑完/失败 | —— |

### 4.8 扩展资源与 Device Plugin：GPU/NPU 怎么变成 K8s 资源

K8s 本身**不认识** GPU/NPU，`nvidia.com/gpu`、`huawei.com/Ascend910` 这种资源名都是**扩展资源（Extended Resource）**——由节点上的 **Device Plugin** 上报并分配的。

> 链路（以昇腾为例）：**NPU 驱动 + CANN → Ascend Docker Runtime（运行时把设备注入容器）→ ascend-device-plugin（把 NPU 变成可调度资源）→ 容器里 `npu-smi info` 才能看到卡**。四环缺一，都会表现为"申请了 NPU 但用不了"。

| | NVIDIA GPU | 昇腾 NPU (Ascend) |
|---|---|---|
| 资源名 | `nvidia.com/gpu` | `huawei.com/Ascend910`（910A/910B 训练）、`huawei.com/Ascend310`（310 推理） |
| Device Plugin | `nvidia-device-plugin`（NVIDIA/k8s-device-plugin） | `ascend-device-plugin`（华为开源，Ascend/ascend-device-plugin） |
| 容器内查看 | `nvidia-smi` | `npu-smi info` |
| 设备可见性控制 | `CUDA_VISIBLE_DEVICES` | `ASCEND_VISIBLE_DEVICES` |
| 镜像内依赖 | CUDA Toolkit / cuDNN | CANN Toolkit（`pip install torch_npu` 配套） |
| 部署形态 | 都是 DaemonSet（每节点一个），部署在 `kube-system` | 同左 |

> 注意区分：**Device Plugin 负责"资源上报与分配"**（K8s 层面），**容器运行时负责"设备注入"**（NVIDIA 是 nvidia-container-toolkit，昇腾是 Ascend Docker Runtime）——两者都要配好，缺一个都拿不到卡。

---

## 5. K8s YAML 详解：字段逐个拆

### 5.1 三段式骨架

任何 K8s 资源都是这个骨架，先记死它：

```yaml
apiVersion: apps/v1        # ① 资源属于哪个 API 组、哪个版本
kind: Deployment           # ② 资源类型：Pod/Service/Deployment/...
metadata:                  # ③ 元数据：名字、标签、命名空间
  name: my-app
  labels:
    app: my-app            #   label 是"给资源贴的标签"，selector 靠它认亲
spec:                      # ④ 期望状态（声明）：你想要的最终样子
  # ...具体字段随 kind 不同
```

### 5.2 Deployment 字段逐行解释

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-engineering-app    # 资源名字（同命名空间内唯一）
  labels:                      # 给这个 Deployment 自己贴的标签
    app: llm-engineering
spec:
  replicas: 2                  # 期望副本数：控制器保证永远有 2 个 Pod 活着
  selector:                    # 控制器"认领"哪些 Pod（必须与模板标签一致！）
    matchLabels:
      app: llm-engineering
  template:                    # Pod 模板：描述"要创建的 Pod 长什么样"
    metadata:
      labels:                  # 给 Pod 贴标签 → 被上面的 selector 和 Service 选中
        app: llm-engineering
    spec:
      containers:
        - name: app            # 容器名（同 Pod 内唯一）
          image: your-registry/llm-engineering:latest
          ports:
            - containerPort: 8000   # 容器内监听端口（声明用，不自动开放外部访问）
          env:                      # 环境变量（也支持从 ConfigMap/Secret 引用）
            - name: MONGODB_URL
              value: "mongodb://llm_engineering:***@mongo:27017"
          resources:                # ⭐ 调度决策的依据，最重要的一节
            requests:               #   请求量：调度器保证"至少这么多"
              cpu: "2"              #   2 核
              memory: 4Gi           #   4 GiB
              nvidia.com/gpu: 1     #   1 张 GPU（需要装 device plugin 才有这个资源）；昇腾集群写 huawei.com/Ascend910: 1
            limits:                 #   上限：最多能用多少
              cpu: "2"              #   CPU 是可压缩资源，超了只是限速
              memory: 8Gi           #   ⭐ 内存不可压缩，超了直接 OOMKilled！
          readinessProbe:           # 就绪探针：通过才给流量（不通过 = 摘流量）
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10 # 启动后等 10 秒再探
          livenessProbe:            # 存活探针：不通过 = 杀掉重启
            httpGet:
              path: /health
              port: 8000
```

> **requests vs limits 的灵魂区别**：requests 是"调度承诺"（写进调度决策），limits 是"运行上限"。CPU 超限只会被限速（可压缩），**内存超限直接被杀进程（不可压缩）**——这是 OOMKilled 的头号来源。日常建议：requests 按真实用量估，limits 给 1.5~2 倍余量，别 requests=limits 卡死自己。

### 5.3 Service 字段解释

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-engineering          # Service 的 DNS 名：其他 Pod 访问 http://llm-engineering:8000
spec:
  selector:                      # 按标签选中后端 Pod（要跟 Pod 模板 labels 完全一致）
    app: llm-engineering
  ports:
    - port: 8000                 # Service 对外端口（别人访问这个端口）
      targetPort: 8000           # 转发到 Pod 的哪个容器端口
  type: ClusterIP                # 默认值；外部访问用 NodePort / LoadBalancer
```

### 5.4 kubectl 命令速查

```bash
kubectl get nodes                       # 看集群有几台机器
kubectl get pods -o wide                # 看 Pod 及所在节点/IP
kubectl get deploy,svc,cm,secret        # 一次看多种资源
kubectl describe pod <name>             # ⭐ 排障第一命令：看 Events
kubectl logs <pod>                      # 看日志；崩溃后看上次：kubectl logs <pod> --previous
kubectl apply -f xx.yaml                # 声明式提交（幂等，可反复执行）
kubectl delete -f xx.yaml               # 按文件删除
kubectl port-forward svc/<name> 8000:8000  # 本地临时访问集群内服务
```

---

## 6. Volcano：K8s 上的 AI 作业调度器

### 6.1 为什么 K8s 原生调度器不够用

K8s 原生调度器是"**逐个 Pod**"调度的：来一个 Pod 看一次有没有空位。这对微服务没问题，但对 AI 作业有三个硬伤：

1. **不感知"一批"**：64 卡作业 = 8 个 8 卡 Pod，调度器逐个放，可能放进去 7 个、第 8 个永远等不到——7 个占着资源空转，谁也跑不了（死锁）；
2. **没有队列/优先级概念**：团队 A 和团队 B 抢资源没有公平性和优先级一说；
3. **不支持公平性（fairness）**：一个占坑任务可能饿死所有其他任务。

Volcano 就是 K8s 生态里补这三块的**批处理调度器**，也是很多 AI 训练平台（如 kubeflow 生态）的默认调度器。

### 6.2 三个核心概念

| 概念 | 类比 | 说明 |
|---|---|---|
| **Queue（队列）** | 银行排号窗口 | 资源管理单元：每个队列有容量（capability）、权重（weight）、优先级。训练任务必须提交到某个队列 |
| **PodGroup（Pod 组）** | 一个"旅行团" | 把一批 Pod 捆成一个调度单元，是 Gang Scheduling 的载体，`minMember` 指定最少几人成团 |
| **VolcanoJob（Job）** | 一张"出团单" | Volcano 自己的作业 CRD：声明任务（tasks）、每个任务几个副本（replicas）、提交到哪个队列、失败策略（policies） |

> 注：Volcano 还有 `PriorityClass`（优先级抢占）和 `Share`（公平调度）等机制，入门先把 Queue / PodGroup / Job 三件套吃透。

### 6.3 Gang Scheduling（成团调度）：为什么训练必须"要么全齐、要么全等"

> 类比：8 人拼桌的餐厅。如果来 5 个人就先占 5 个座位，剩下 3 个永远坐不满，5 个人干等、座位还浪费——不如 8 人齐了再一起入座。**Gang Scheduling = "成团才调度，不成团全排队"。**

没有 Gang Scheduling 的死锁过程：

1. 64 卡作业的 8 个 Pod 陆续被调度；
2. 资源碎片化：7 个 Pod 占住 56 卡，最后 1 个等不到 8 张连续空卡；
3. 先到的 7 个占着资源空转（或初始化 NCCL 永远连不上），其他作业也用不了这 56 卡——**全局死锁**。

Volcano 的做法：把 8 个 Pod 组成一个 PodGroup（`minMember: 8`），**要么 8 个 Pod 的资源一次找齐一起调度，要么一个都不调度**，全部留在队列里等，把资源让给能跑的任务。

### 6.4 真实 VolcanoJob YAML（字段已注释）

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training
spec:
  weight: 1                  # 队列权重：多队列共享集群时按权重分资源
  capability:                # 该队列最多能用多少（可选，不写 = 不设上限）
    cpu: "200"
    memory: 512Gi
---
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-pretrain
spec:
  minAvailable: 8            # ⭐ Gang Scheduling：8 个 task 齐了才一起调度
  queue: training            # 提交到 training 队列
  tasks:
    - name: worker           # 任务名（日志里能看到）
      replicas: 8            # 这个 task 要 8 个副本 → 8 个 Pod
      template:
        spec:
          containers:
            - name: worker
              image: your-registry/train:latest
              resources:
                limits:
                  nvidia.com/gpu: 8   # 每个 worker 要 8 卡 → 整作业 64 卡
  policies:                  # 作业失败策略
    - event: PodFailed       # 有 Pod 失败时
      action: RestartJob     # 重启整个作业
```

> 对比 K8s 原生：这个"8 个 Pod 必须同时调度"的需求，原生 Deployment 表达不了，必须靠 Volcano 这类调度器。这也是第 3 节表格里"Volcano = K8s 原生调度器的插件增强"的含义。

### 6.5 昇腾 NPU 集群部署示例（Volcano + ascend-device-plugin）

Volcano 由**华为云开源**（2019 年捐赠给 CNCF），是昇腾 NPU 集群的事实标准调度器——华为云 ModelArts、CCE 的 AI 作业调度底层就是它。昇腾集群上 Volcano 除了 Gang Scheduling，还支持 **NPU 拓扑感知调度**（把 8 卡全互联的 HCCS 拓扑考虑进放置决策，减少跨节点通信）。

在昇腾集群上跑训练分两步：**先让 K8s"看见"NPU（device plugin），再提交作业（VolcanoJob）**。

**① 部署 ascend-device-plugin（每节点一个 DaemonSet）：**

```yaml
# ascend-device-plugin.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ascend-device-plugin
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: ascend-device-plugin
  template:
    metadata:
      labels:
        app: ascend-device-plugin
    spec:
      hostNetwork: true                     # 插件要跟 kubelet 通信，用宿主机网络
      containers:
        - name: ascend-device-plugin
          image: ascendai/ascend-device-plugin:latest
          securityContext:
            privileged: true                # 需要访问宿主机设备节点
          env:
            - name: DEVICE_TYPE             # 上报哪种 NPU：Ascend910 / Ascend310
              value: "Ascend910"
          volumeMounts:
            - name: device-plugin            # kubelet 的 device plugin socket 目录
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
```

```bash
kubectl apply -f ascend-device-plugin.yaml
kubectl get ds -n kube-system                    # ascend-device-plugin Running
kubectl get node <node> -o json | jq '.status.capacity'
# 看到 "huawei.com/Ascend910": "8" = 节点 8 张 NPU 已可调度
```

> 前置条件：节点已装 NPU 驱动 + CANN，且容器运行时配好 **Ascend Docker Runtime**（否则 NPU 不会注入容器，见 10.4 排障）。

**② 提交昇腾训练作业（VolcanoJob，64 卡 = 8 worker × 8 NPU）：**

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-pretrain-ascend
spec:
  minAvailable: 8                 # Gang Scheduling：8 个 worker 齐了才一起调度
  queue: training
  tasks:
    - name: worker
      replicas: 8
      template:
        spec:
          containers:
            - name: worker
              image: your-registry/train-ascend:latest   # 装好 CANN + torch_npu 的镜像
              resources:
                limits:
                  huawei.com/Ascend910: 8    # ⭐ 昇腾资源名：每个 worker 8 张 NPU
              env:
                - name: ASCEND_VISIBLE_DEVICES   # 显式指定可见设备（可选）
                  value: "0,1,2,3,4,5,6,7"
```

```bash
kubectl apply -f ascend-job.yaml
kubectl get pods -w                     # 8 个 worker 同时 Pending → 同时 Running
kubectl exec -it <pod> -- npu-smi info  # 容器内看到 8 张 NPU 即成功
```

> 与 GPU 集群的唯一区别就是**资源名**（`huawei.com/Ascend910` vs `nvidia.com/gpu`）和**镜像内的软件栈**（CANN + torch_npu vs CUDA）。调度逻辑、Gang Scheduling、队列全部一致——这就是第 4.8 节"扩展资源"抽象带来的好处。

---

## 7. Ray：分布式 Python 计算框架

### 7.1 核心抽象：四个概念吃透 Ray

Ray 的思路很直接：**你的代码还是 Python，Ray 负责把它分散到多台机器上跑。** 集群 = 1 个 Head 节点 + N 个 Worker 节点。

| 概念 | 一句话 | 类比 |
|---|---|---|
| **Task（任务）** | `@ray.remote` 装饰的无状态函数，异步提交，返回 ObjectRef | 把活分包给流水线工人，交回执 |
| **Actor（演员）** | `@ray.remote` 装饰的**类**，实例常驻内存、跨调用保留状态 | 一个有记忆的专属员工，比如计数器、模型副本 |
| **Object Store（对象存储）** | 分布式共享内存，Task/Actor 之间传大对象（numpy 张量零拷贝） | 工人们共用的货架，放半成品 |
| **Ray Serve（推理服务）** | 基于 Ray 的模型服务框架：HTTP 端点、自动扩缩、请求批处理 | 前台点单 + 后厨批量出餐 |

> Ray 的核心理念：**Task 管"无状态计算"，Actor 管"有状态服务"，Object Store 管"数据搬运"，Serve 管"对外服务"。**

### 7.2 十行代码感受 Ray

```python
import ray

ray.init()                      # 单机模式；连集群用 ray.init(address="auto")

@ray.remote
def square(x):                  # Task：无状态远程函数
    return x * x

@ray.remote
class Counter:                  # Actor：有状态远程对象
    def __init__(self):
        self.n = 0
    def inc(self):
        self.n += 1
        return self.n

# 提交 10 个 Task，立刻返回 10 个"回执"（ObjectRef），不阻塞
futures = [square.remote(i) for i in range(10)]
print(ray.get(futures))         # 阻塞取回结果 → [0, 1, 4, ..., 81]

c = Counter.remote()            # 创建 Actor（在某个 worker 上常驻）
print(ray.get(c.inc.remote()))  # → 1
print(ray.get(c.inc.remote()))  # → 2（状态跨调用保留）
```

### 7.3 Ray 与 K8s / Volcano 的分工

| 场景 | 谁管 | 管什么 |
|---|---|---|
| K8s（KubeRay） | 管 Ray 集群本身 | Head/Worker 节点的 Pod 生命周期、扩缩容 |
| Ray | 管 Python 任务 | Task/Actor 的调度、对象存储、容错重试 |
| Volcano | 管训练作业 | K8s 上跑 PyTorchJob 等批处理作业的排队/成团 |

**选型一句话**：代码里已经 `@ray.remote` 满天飞、要的是 Python 级分布式 → 用 Ray；要的是"在 K8s 上排队跑训练作业" → 用 Volcano；要的是稳定对外服务的推理 → 用 K8s 原生 Deployment + Service。

---

## 8. 代码：把 docker-compose.yml 翻译成 K8s

LLM-Engineers-Handbook 用 `docker-compose.yml` 在单机编排三个服务：`mongo`（元数据）、`qdrant`（向量库）、应用本体（`Dockerfile` 构建）。compose 的声明式风格（services / volumes / networks / restart: always）翻译成 K8s 就是 Deployment + Service + PVC。

### 8.1 应用服务：Deployment + Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-engineering-app
  labels:
    app: llm-engineering
spec:
  replicas: 2                      # compose 的 restart: always → 这里由控制器保证
  selector:
    matchLabels:
      app: llm-engineering
  template:
    metadata:
      labels:
        app: llm-engineering
    spec:
      containers:
        - name: app
          image: your-registry/llm-engineering:latest   # Dockerfile 构建产物
          ports:
            - containerPort: 8000
          env:
            - name: MONGODB_URL        # compose 里的环境变量注入同理
              value: "mongodb://llm_engineering:***@mongo:27017"
            - name: QDRANT_URL
              value: "http://qdrant:6333"
          resources:                  # 调度器做放置决策的依据
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              memory: 8Gi
          readinessProbe:             # 健康检查 → 不健康就不给流量/重启
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llm-engineering
spec:
  selector:
    app: llm-engineering
  ports:
    - port: 8000
      targetPort: 8000
```

### 8.2 有状态组件：StatefulSet + PVC（对应 compose 的 volumes）

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
          volumeMounts:
            - name: qdrant-data
              mountPath: /qdrant/storage   # 对应 volumes: qdrant_data:/qdrant/storage
  volumeClaimTemplates:                     # 每个副本自动生成独立 PVC
    - metadata:
        name: qdrant-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 20Gi
```

### 8.3 compose → K8s 对照表

| docker-compose | Kubernetes | 说明 |
|---|---|---|
| `services:` | Deployment / StatefulSet | 无状态用 Deployment，有状态用 StatefulSet |
| `restart: always` | 控制器 + 探针 | 控制器保证副本数，探针决定健康 |
| `volumes:` | PVC + volumeMounts | 持久化存储，Pod 重建数据不丢 |
| `networks:` + 服务名 | Service + DNS | `http://qdrant:6333` 照旧可用 |
| `environment:` | env / ConfigMap / Secret | 敏感信息放 Secret |
| `ports:` | Service（+ NodePort） | compose 暴露端口 → K8s 用 Service 暴露 |

---

## 9. 实战：用 K8s 部署 vLLM 推理服务

把第 5 章的 vLLM 服务搬上 K8s，完整走一遍"写 YAML → 部署 → 验证"。

### 9.1 完整 YAML（字段已在第 5 节解释，这里直接看组合）

```yaml
# vllm.yaml
apiVersion: v1
kind: PersistentVolumeClaim          # ① 模型缓存卷：HuggingFace 缓存放这里，Pod 重建不重新下载
metadata:
  name: hf-cache
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 50Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama8b
spec:
  replicas: 1                       # 推理服务起步 1 副本，后续按需扩
  selector:
    matchLabels:
      app: vllm-llama8b
  template:
    metadata:
      labels:
        app: vllm-llama8b
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.3
          command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - --model
            - meta-llama/Llama-3.1-8B-Instruct
            - --tensor-parallel-size
            - "1"
            - --gpu-memory-utilization
            - "0.9"                 # 允许 vLLM 用 90% 显存（含 KV cache）
          ports:
            - containerPort: 8000
          env:
            - name: HF_TOKEN         # 私有模型仓库的令牌，从 Secret 注入
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: token
          resources:
            requests:
              cpu: "8"
              memory: 32Gi
              nvidia.com/gpu: 1      # ⭐ 声明要 1 张 GPU（前提：节点装了 device plugin）
            limits:
              cpu: "8"
              memory: 40Gi
              nvidia.com/gpu: 1
          readinessProbe:            # vLLM 有 /health 端点，天然适合做就绪探针
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30  # 模型加载要时间，别一上来就探
            periodSeconds: 10
          volumeMounts:
            - name: hf-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: hf-cache
          persistentVolumeClaim:
            claimName: hf-cache
---
apiVersion: v1
kind: Service
metadata:
  name: vllm
spec:
  selector:
    app: vllm-llama8b
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### 9.2 部署与验证步骤

```bash
# ① 创建 Secret（HF_TOKEN 需要 base64）
echo -n "hf_xxxx" | base64   # 得到编码串，写进 Secret YAML 或：
kubectl create secret generic hf-secret --from-literal=token=hf_xxxx

# ② 提交全部资源
kubectl apply -f vllm.yaml

# ③ 看部署进度（-w = 持续观察直到 Running）
kubectl get pods -w
kubectl logs deploy/vllm-llama8b   # 看到 "Application startup complete" 即就绪

# ④ 本地打通到集群内服务，发一次推理请求
kubectl port-forward svc/vllm 8000:8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct",
       "messages":[{"role":"user","content":"用一句话介绍 Kubernetes"}]}'
```

### 9.3 弹性伸缩（HPA）

```bash
# 按 CPU 自动扩缩：CPU 平均超 70% 就加副本（1~4 个）
kubectl autoscale deployment vllm-llama8b --cpu-percent=70 --min=1 --max=4
kubectl get hpa -w
```

> ⚠️ 注意：vLLM 按 CPU 扩缩是简化做法——每副本要 1 张 GPU，扩缩真正该看的是 GPU 利用率和 KV cache 占用（需要 Prometheus 自定义指标，进阶后再上）。先手动 `kubectl scale deployment vllm-llama8b --replicas=2` 也行。

---

## 10. 排障案例

### 10.1 Pod 一直 Pending（调度失败）

**症状**：`kubectl get pods` 显示 Pod 卡在 `Pending` 几个小时。

**排查**：`kubectl describe pod <name>`，看最下面的 **Events**：

```
Events:
  Type     Reason            Age   From               Message
  Warning  FailedScheduling  12s   default-scheduler  0/8 nodes available: 3 Insufficient nvidia.com/gpu, 5 Insufficient memory.
```

**根因与解法**：

| 根因 | Events 特征 | 解法 |
|---|---|---|
| 资源不足 | `Insufficient nvidia.com/gpu` / `Insufficient memory` | 扩容节点，或减小 requests |
| GPU 调度器没装 | `0/8 nodes available` 但节点明明有 GPU，且看不到 `nvidia.com/gpu` 资源 | 装 NVIDIA device plugin（`kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/...`），确认 `kubectl get nodes` 显示 `nvidia.com/gpu: 8` |
| 节点被"污点"（taint） | `node(s) had untolerated taint` | 给 Pod 加 toleration，或去掉节点污点 |
| 镜像拉取慢 | `Pulling image ...` 卡住（ImagePullBackOff） | 换内网镜像仓库 / 提前预热镜像 |

> 口诀：**Pending 先 describe，Events 会告诉你一切。** 不要瞎猜。

### 10.2 容器被 OOMKilled

**症状**：Pod 反复重启，`kubectl get pods` 显示 `OOMKilled`，`Restart Count` 飙升。

**排查**：

```bash
kubectl describe pod <name>          # 看 Last State: Terminated, Reason: OOMKilled, Exit Code: 137
kubectl logs <pod> --previous        # 看被杀前的最后日志（常常有 CUDA out of memory 或分配失败的痕迹）
```

**根因与解法**：

| 根因 | 说明 | 解法 |
|---|---|---|
| 内存 limit 设太小 | 容器实际用量超了 `limits.memory`，被 cgroup 杀 | 调大 limit（给 1.5~2 倍余量） |
| 训练进程组连带被杀 | 节点开了 `memory.oom.group=1`，一个进程 OOM 整组陪葬（ml-engineering 里的经典坑） | 让管理员在节点池配置 `singleProcessOOMKill: true`，只杀肇事进程 |
| 推理时显存规划不当 | vLLM `--gpu-memory-utilization` 设太高 + 模型加载超内存 | 调低该参数，或减小 `--max-model-len` |

> **Exit Code 速查**：`137` = 被 SIGKILL（OOM 典型）；`143` = 被 SIGTERM（正常下线）；`1` = 应用自己报错退出；`126/127` = 命令不存在/不可执行。看到 137 先怀疑内存，不要只盯着应用日志。

### 10.3 容器里看不到 GPU

**症状**：Pod 明明申请了 `nvidia.com/gpu: 1` 也调度成功了，但容器里 `nvidia-smi` 报错 / PyTorch 报 `CUDA not available`。

**根因排查顺序**：

| 步骤 | 命令 | 说明 |
|---|---|---|
| ① 节点有没有 GPU 资源可分配 | `kubectl get node <node> -o yaml \| grep nvidia` | 节点 capacity 里没有 `nvidia.com/gpu` → device plugin 没装 |
| ② device plugin 是否在跑 | `kubectl get ds -n kube-system` | 没有 `nvidia-device-plugin` DaemonSet → 安装它 |
| ③ 运行时是否配了 GPU 支持 | 节点上 `nvidia-container-toolkit` 是否安装配置 | containerd/docker 需要它把 GPU 设备注入容器 |
| ④ 容器内实测 | `kubectl exec -it <pod> -- nvidia-smi` | 容器内看不到卡 = ①②③ 之一，看到卡但 torch 报错 = PyTorch 装成 CPU 版（见第 1 章） |

> 记住：**K8s 不会自动把 GPU 给容器。** 链路是 `节点驱动 → nvidia-container-toolkit（运行时注入）→ device plugin（把 GPU 变成可调度资源）→ 容器里 nvidia-smi 才能看到卡`。四环缺一，都会表现为"申请了 GPU 但用不了"。

### 10.4 昇腾 NPU 分配失败

昇腾集群的排障套路和 GPU 几乎一一对应，只是命令和资源名不同：

**症状 A：Pod 卡 Pending，Events 报 `Insufficient huawei.com/Ascend910`**

```bash
kubectl describe pod <name>          # Events: 0/4 nodes available: 4 Insufficient huawei.com/Ascend910
```

| 根因 | 排查 | 解法 |
|---|---|---|
| NPU 真的不够 | `kubectl get node <node> -o json \| jq '.status.allocatable'` 看剩余 | 扩容节点，或减小 `huawei.com/Ascend910` 请求量 |
| device plugin 没上报资源 | 节点 capacity 里**根本没有** `huawei.com/Ascend910` | 部署 ascend-device-plugin（见 6.5），确认 `kubectl get ds -n kube-system` 里它在 Running |
| device plugin 上报了错误型号 | 装了 910 的插件但节点是 310，或 `DEVICE_TYPE` 配错 | 改 DaemonSet 的 `DEVICE_TYPE` 环境变量，重建插件 |

**症状 B：Pod 调度成功，但容器里 `npu-smi info` 报错 / PyTorch 报 `NPU not available`**

| 步骤 | 命令 | 说明 |
|---|---|---|
| ① 节点有没有 NPU 资源 | `kubectl get node <node> -o yaml \| grep Ascend` | capacity 里没有 → device plugin 没装 |
| ② 运行时是否注入设备 | 节点上检查容器运行时是否配了 Ascend Runtime | containerd 配置里要加 `ascend` runtime（类似 GPU 的 nvidia-container-toolkit） |
| ③ 驱动/CANN 是否正常 | 节点上 `npu-smi info` | 宿主机都看不到卡 → 驱动或固件问题，与 K8s 无关 |
| ④ 容器内实测 | `kubectl exec -it <pod> -- npu-smi info` | 看到卡但 torch 报错 = `torch_npu` 没装对版本（要和 CANN 版本匹配，见第 3 章） |

> 昇腾的对应口诀：**驱动/CANN（宿主机）→ Ascend Docker Runtime（注入）→ ascend-device-plugin（上报资源）→ torch_npu（容器内）**，一环断链都表现为"申请了 NPU 但用不了"。容器内没有 `ASCEND_VISIBLE_DEVICES` 时插件默认把容器申请到的卡全部注入；多机任务还要确认节点间 RoCE/HCCS 组网连通（`hccn_tool -i 0 -t ls -p` 查链路）。

### 10.5 高频排查清单

1. **Pod 一直 Pending** → `kubectl describe pod <name>` 看 Events：多半是资源 requests 超出节点剩余，或 GPU/NPU 调度器（device plugin：`nvidia-device-plugin` / `ascend-device-plugin`）没装。
2. **Pod 反复 CrashLoopBackOff** → `kubectl logs <pod> --previous` 看上次退出原因；检查 liveness/readiness 探针是否太苛刻。
3. **训练作业中途被杀** → 先查 OOM（见 10.2），再查抢占（Volcano priorityClass 配置）。
4. **多机训练连不上** → 确认 head 节点地址/端口在所有 worker 可达（K8s 网络策略、SLURM 的 `MASTER_ADDR` 是否用了 `scontrol` 解析而非写死 IP）。
5. **本地仓库** `ml-engineering` 的 orchestration 目录是排查编排问题的第一站：`orchestration/kubernetes/README.md`（OOM 组杀问题）、`orchestration/slurm/`（SLURM 提交与队列管理）。

---

## 11. 动手实验：本地跑一个 K8s

没有真实集群也能学：macOS/Linux 上用 **minikube** 或 **kind** 三分钟起一个单机 K8s。

### 11.1 安装与启动

```bash
# 方案 A：minikube（带完整控制面，适合体验全功能）
brew install minikube kubectl        # macOS；Linux 用对应包管理器
minikube start --driver=docker --cpus=4 --memory=8g
kubectl get nodes                    # 看到 minikube 节点 = 成功

# 方案 B：kind（快、省资源，适合 CI 和纯实验）
brew install kind
kind create cluster --name demo
kubectl get nodes
```

> minikube 自带一堆 addons，值得玩：`minikube addons list`、`minikube dashboard`（打开 Web 控制台）。

### 11.2 实验 1：部署 + 自愈（15 分钟）

```bash
kubectl create deployment web --image=nginx --replicas=3
kubectl get pods -o wide                 # 3 个副本分布在节点上

# 见证自愈：删掉一个 Pod，控制器立刻重建
kubectl delete pod web-xxxxx             # 换成实际 Pod 名
kubectl get pods -w                      # 秒级出现新的 web-xxxxx

# 见证滚动更新 + 回滚
kubectl set image deployment/web web=nginx:1.25
kubectl rollout status deployment/web
kubectl rollout undo deployment/web      # 一键回滚
```

### 11.3 实验 2：ConfigMap + 探针（10 分钟）

```bash
# 配置与代码分离：改配置不用重建镜像
kubectl create cm app-config --from-literal=LOG_LEVEL=debug --from-literal=MODE=test
kubectl get cm app-config -o yaml        # 看 ConfigMap 长什么样

# 在 Deployment 里引用它（envFrom 整包注入）
kubectl set env deployment/web --from=configmap/app-config
kubectl exec deploy/web -- env | grep -E 'LOG_LEVEL|MODE'
```

### 11.4 实验 3：模拟 GPU 调度（理解"GPU 是资源"这件事）

本地 Mac 没有 GPU，但可以**给节点注册假的扩展资源**，观察调度器如何对待 `nvidia.com/gpu`：

```bash
# 把节点伪装成有 4 张 GPU（扩展资源名 nvidia.com/gpu）
kubectl proxy &   # 起本地代理
curl -X PATCH -H "Content-Type: application/json-patch+json" \
  --data '[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"4"}]' \
  http://localhost:8001/api/v1/nodes/minikube/status

kubectl describe node minikube | grep -A4 Capacity   # 看到 nvidia.com/gpu: 4

# 部署一个要 4 张卡的 Pod → 能调度；改成 8 张 → Pending，describe 看 Insufficient nvidia.com/gpu
```

> 真实 GPU 集群只是把"假的扩展资源"换成 device plugin 上报的真实 GPU 数量，调度逻辑一模一样。理解了这个，10.1 的 Pending 排障就通了。

### 11.5 实验 4：本地玩 Volcano / Ray

```bash
# Volcano：一条命令装上（实际是装一批 CRD + 调度器）
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
kubectl get pods -n volcano-system       # volcano-scheduler 起来了
# 然后就可以 apply 第 6.4 节的 Queue + Job 试试 Gang Scheduling

# Ray：不装集群，先体验 API
pip install ray
python -c "import ray; ray.init(); ..."  # 或直接跑第 7.2 节的脚本
```

---

## 12. 自测题

**Q1** ⭐：一个 64 卡训练任务，如果没有 Gang Scheduling，为什么可能永远跑不完？K8s 原生调度器 vs Volcano 在处理这个问题上有什么区别？

<details>
<summary>参考答案</summary>

资源碎片化 + 死锁：任务分到 60 卡先启动，剩下 4 卡永远等不到，60 卡还被占死。K8s 原生调度器逐个调度 Pod，不感知"一批要齐"；Volcano 提供 Gang Scheduling（minMember/minAvailable），整批成员齐了才一起调度，否则全部排队等待。

</details>

**Q2** ⭐：docker-compose 的 `volumes: qdrant_data:/qdrant/storage` 和 `restart: always`，在 K8s 里分别对应什么机制？

<details>
<summary>参考答案</summary>

卷 → StatefulSet 的 `volumeClaimTemplates` + PVC（持久化存储，Pod 重建后数据还在）；`restart: always` → Deployment/StatefulSet 控制器 + readinessProbe（容器挂了控制器重建 Pod，探针失败则摘流量/重启）。

</details>

**Q3** ⭐⭐：推理服务（长驻、弹性扩缩）和预训练作业（批处理、多机多卡），按 ml-engineering 的经验分别更适合 K8s 还是 SLURM？为什么？

<details>
<summary>参考答案</summary>

推理服务适合 K8s：长驻、多副本、按流量弹性伸缩、需要服务发现和自愈，这是 K8s 的强项。预训练作业适合 SLURM：基于 Unix、多机协调（ssh/环境变量）自然、交互式调试方便；K8s 对训练来说重造轮子且缺 Unix 特性（连 Unix 用户名都没有），需要大团队支撑。混合集群常见形态：K8s 跑推理 + SLURM 跑训练。

</details>

**Q4** ⭐⭐：Pod 被 OOMKilled，`kubectl describe pod` 里该看哪几处？Exit Code 137 意味着什么？为什么 CPU 超限不会杀进程、内存超限会？

<details>
<summary>参考答案</summary>

看 `Last State`（Reason: OOMKilled, Exit Code: 137）和 `Restart Count`。137 = 进程被 SIGKILL，是 cgroup 内存超限的典型信号（应用自己报错退出一般是 1）。CPU 是可压缩资源（compressible）：超限只是被限速（throttling），进程还活着；内存是不可压缩资源：超限时内核必须立刻腾内存，只能杀进程。所以 `limits.memory` 要给余量，requests 按真实用量估。

</details>

**Q5** ⭐⭐：一个要 GPU 的 Pod 卡在 Pending 一整天，写出你的完整排查步骤（至少 4 步）。

<details>
<summary>参考答案</summary>

1. `kubectl describe pod <name>` 看 Events——先确认到底是不是资源问题（`Insufficient nvidia.com/gpu` 还是别的）；
2. `kubectl get nodes` + `kubectl describe node <node>` 看节点 capacity 里有没有 `nvidia.com/gpu`——没有 = device plugin 没装，装了才能上报 GPU 资源；
3. 确认 `kubectl get ds -n kube-system` 里 `nvidia-device-plugin` 在 Running；
4. 检查节点污点：Events 里有 `untolerated taint` 就给 Pod 加 toleration；
5. 最后检查镜像是否拉得动（ImagePullBackOff），以及 requests 是否超过节点剩余资源。

</details>

**Q6** ⭐⭐⭐：手写一个 vLLM 推理服务的 Deployment 要点清单——至少包含：GPU 资源声明、模型缓存持久化、探针、启动命令。（不需要完整 YAML，列要点即可）

<details>
<summary>参考答案</summary>

1. `resources.requests/limits.nvidia.com/gpu: 1`（前提：节点有 device plugin）；
2. 模型缓存挂 PVC 到 `/root/.cache/huggingface`，Pod 重建不重新下载（也可用 model 目录挂载）；
3. `readinessProbe` 打 `/health`，`initialDelaySeconds` 给足 30s+（模型加载慢）；
4. 启动命令用 `python -m vllm.entrypoints.openai.api_server --model <name> --tensor-parallel-size 1`；
5. 私有模型记得 `HF_TOKEN` 从 Secret 注入；
6. `--gpu-memory-utilization 0.9` 控制显存规划，防止 OOM。

</details>

**Q7** ⭐⭐：在昇腾 NPU 集群上用 K8s 跑 64 卡训练作业，需要装哪些组件（按链路从宿主机到容器）？资源名应该写什么？和 GPU 集群相比，唯一区别是什么？

<details>
<summary>参考答案</summary>

链路：宿主机 NPU 驱动 + CANN → Ascend Docker Runtime（容器运行时注入设备）→ ascend-device-plugin（DaemonSet，把 NPU 上报为可调度资源）→ 容器镜像内装 CANN + torch_npu。资源名写 `huawei.com/Ascend910`（推理用 `huawei.com/Ascend310`）。和 GPU 集群的唯一区别就是资源名和镜像内软件栈（CANN + torch_npu vs CUDA），调度、Gang Scheduling、队列这些编排机制完全一致——这就是"扩展资源"抽象的意义。

</details>

---

## 13. 类比总结：编排 = 餐厅经理协调 10 个厨师

把集群想象成一家餐厅的后厨：

- **容器 = 标准半成品**：每道菜（任务）连同配料、菜谱、工具打包成一份标准箱，哪个厨师（机器）来都能做，味道一致。
- **调度器 = 餐厅经理**：
  - 客人点"64 人宴席"（64 卡任务），经理确认 8 个厨师凑齐才开火——这就是 **Gang Scheduling**（Volcano）；
  - 有人点了招牌菜（推理服务），经理保证炉灶永远留两个，客人随到随做，忙时加人手（**弹性伸缩**，K8s HPA）；
  - 一个厨师切到手（节点故障），经理立刻换人顶上，菜照常上（**自愈**）。
- **ConfigMap/Secret = 菜单与备料单**：菜谱（配置）单独放，换菜单不用换厨房（镜像）。
- **没有经理的后果**：10 个厨师各自抢灶台、半成品堆满台面、客人投诉——就像 100 台服务器没人调度时的混乱。

**一句话记住本章**：容器解决"环境一致性"，调度器解决"资源分配"，编排层把 100 台机器变成一台"按需取用"的巨型计算机。

---

## 参考链接

- [Kubernetes 官方文档 — 概念](https://kubernetes.io/zh-cn/docs/concepts/)（Pod / Service / Deployment / StatefulSet 逐个看）
- [Volcano 官方文档](https://volcano.sh/en/docs/)（Gang Scheduling / Queue / Job）
- [ascend-device-plugin（华为开源）](https://github.com/Ascend/ascend-device-plugin)（昇腾 NPU 的 K8s device plugin）
- [Ascend 官方文档 — 容器化部署](https://www.hiascend.com/document)（Ascend Docker Runtime / CANN 版本配套）
- [Ray 官方文档](https://docs.ray.io/)（Core / Serve）
- [KubeRay](https://github.com/ray-project/kuberay)（Ray on Kubernetes）
- [ml-engineering — orchestration 目录](https://github.com/stas00/ml-engineering/tree/master/orchestration)（K8s / SLURM 排障手册）
- [vLLM 官方文档 — 部署](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)

---

## 导航

- 上一章：[第5章 推理层](../05-推理层/README.md)
- 下一章：[07-应用层](../07-应用层/README.md)
- 返回目录：[AI-Infra 教程](../README.md)
