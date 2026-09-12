---
title: 第07章 应用层
tags: ['ai-infra', '教程', '应用', 'dify', 'langchain']
章节: 07
状态: 完成
---

# 第7章：应用层

> 不会写代码也想用 AI？这层就是为你准备的。

## 7.1 问题：技术门槛太高

你有一个好想法，想用 AI 实现，但：
- 不会写 Python
- 不懂什么是 API
- 只想拖拖拽拽就能用

**应用层的目标**：让非技术人员也能用 AI。

## 7.2 方案对比

| 方案 | 一句话 | 适合谁 | 上手时间 |
|------|--------|--------|---------|
| 低代码平台 | 拖拽式搭建 AI 应用 | 产品经理/运营 | 30 分钟 |
| API 封装 | 把模型包装成 HTTP 接口 | 前端/后端开发者 | 2 小时 |
| Agent 框架 | 让 AI 自己调用工具 | 进阶开发者 | 1 天 |

## 7.3 项目详解

### Dify：拖拽式 AI 应用开发平台 ⭐60k

**一句话**：不用写代码，拖拖拽拽就能搭建 AI 应用。

**核心功能**：
- 可视化编排：拖组件、连线、配置参数
- 多模型支持：OpenAI、Claude、本地模型（Ollama）
- 知识库：上传文档，自动切片、向量化
- 工作流：条件分支、循环、人工审核

**快速上手**：
```bash
# 克隆
git clone https://github.com/langgenius/dify.git
cd dify/docker

# 启动
docker-compose up -d
```

打开 `http://localhost`，注册账号，创建应用：
1. 选"聊天助手"模板
2. 拖一个"LLM"节点，选模型（GPT-4 或本地模型）
3. 拖一个"知识库"节点，上传 PDF
4. 发布，获得 API 链接

**适用场景**：
- 企业内部知识库问答
- 客服机器人
- 内容生成助手

### FastGPT：知识库问答专家 ⭐25k

**一句话**：中文场景开箱即用的知识库问答系统。

**核心功能**：
- 文档上传：PDF、Word、网页、Markdown
- 自动切片：按段落/语义切分
- 向量检索：Milvus/PGVector
- 可视化编排：类似 Dify

**快速上手**：
```bash
# 一键部署
curl -O https://raw.githubusercontent.com/labring/FastGPT/main/deploy/docker-compose.yml
docker-compose up -d
```

**优势**：
- 中文优化好
- 社区活跃，中文文档全
- 与微信生态集成方便

### LangChain：代码编排框架 ⭐100k

**一句话**：用 Python 代码编排 AI 应用，灵活度最高。

**核心概念**：

| 概念 | 一句话 | 例子 |
|------|--------|------|
| Chain | 固定流程 | 输入 → 翻译 → 摘要 → 输出 |
| Agent | 自主决策 | AI 自己决定调用搜索还是计算器 |
| Memory | 记忆 | 记住之前的对话内容 |
| Tool | 工具 | 搜索引擎、计算器、API |

**代码示例**：
```python
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType

# 1. 简单 Chain
template = "请用简单的话解释：{concept}"
prompt = PromptTemplate(template=template, input_variables=["concept"])
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("Transformer")

# 2. Agent（自主决策）
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("2023 年美国总统是谁？他的年龄乘以 2 是多少？")
```

**适用场景**：
- 需要复杂逻辑的应用
- 需要调用外部工具（搜索、数据库、API）
- 需要记忆上下文的多轮对话

### ModelScope：国内模型社区

**一句话**：阿里出的 HuggingFace 替代品。

**优势**：
- 国内访问快
- 中文模型多
- 与阿里云集成

## 7.4 选型指南

| 场景 | 推荐工具 | 理由 |
|------|---------|------|
| 企业内部知识库 | FastGPT / Dify | 中文好，开箱即用 |
| 复杂 AI 应用 | LangChain | 灵活，可定制 |
| 快速原型验证 | Dify | 30 分钟出 Demo |
| 国内部署 | ModelScope + 本地模型 | 网络稳定 |

## 7.5 排障参考

参考 ml-engineering `debug/` 通用方法论：

1. **先看日志**：Dify/FastGPT 都有 Web 界面日志
2. **再简化问题**：把复杂流程拆成单步测试
3. **最后分段排查**：用二分法定位问题步骤

常见问题：
- Dify 启动失败 → 检查 Docker 内存分配（建议 4GB+）
- FastGPT 问答不准确 → 检查知识库切片大小（建议 500-1000 字）
- LangChain 调用超时 → 检查网络、API Key 是否过期

## 7.6 动手实验

> 环境要求：Docker（实验1、2）、Python 3.9+（实验3）、一个可用的 LLM API Key（OpenAI 或任意兼容接口）。

### 实验1：用 Dify 搭建一个简单聊天机器人

**目标**：本地启动 Dify，创建一个"聊天助手"应用并发布，最终能在 Web 界面里跟机器人对话。

**步骤**：

1. 克隆并启动 Dify（约需 5-10 分钟拉取镜像）：

```bash
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env
docker-compose up -d
```

2. 等所有容器变为 healthy：

```bash
docker-compose ps
# 期望看到 api / worker / web / db / redis / weaviate 等均为 Up
```

3. 打开 `http://localhost`，注册管理员账号（首次打开会引导创建）。

4. 在 Dify 里接入模型：**设置 → 模型供应商**，填入你的 `OPENAI_API_KEY`（或添加 Ollama 等本地模型）。

5. 创建应用：**创建空白应用 → 聊天助手**，在"提示词编排"里写一句人设提示词，如：

```
你是一个耐心的 AI 助教，用简单的语言回答初学者的问题。
```

6. 在右下角预览窗口直接对话，测试"你好，什么是 Transformer？"。

7. **发布**：右上角点击 **发布** → 生成"访问 API"，可拿到 `API Key` 和 `APP URL`，分享链接即可让别人用。

**预期输出**：

- 浏览器打开 `http://localhost` 显示 Dify 登录/注册页
- 预览窗口能流式输出模型的回答
- 发布后拿到形如 `app-xxxxx` 的 API Key 和可访问的 Web App 链接

**常见错误**：

| 错误现象 | 原因 | 解决 |
|---------|------|------|
| `docker-compose up` 后 Web 页面打不开 | 镜像未拉完或容器未就绪 | 等 1-2 分钟再刷新；用 `docker-compose logs web` 看日志 |
| 对话时报"模型未配置" | 没填模型供应商的 API Key | 设置 → 模型供应商 → 添加 OpenAI/Ollama |
| 上传文件/对话卡死 | Docker 内存不足 | 给 Docker Desktop 分配 4GB+ 内存 |
| 回答是英文或很生硬 | 系统提示词太简单 | 在提示词里明确"用中文回答" |

---

### 实验2：用 FastGPT 上传 PDF 并问答

**目标**：本地启动 FastGPT，上传一份 PDF 建立知识库，然后基于该 PDF 提问并得到带引用的回答。

**步骤**：

1. 下载部署文件并启动（需先准备 MongoDB 和 OneAPI 的配置文件）：

```bash
mkdir fastgpt && cd fastgpt
curl -O https://raw.githubusercontent.com/labring/FastGPT/main/projects/app/data/config.json
curl -O https://raw.githubusercontent.com/labring/FastGPT/main/deploy/docker-compose.yml
# 按官方文档在 config.json 中配置 LLM 模型（如 OneAPI 指向 OpenAI）
docker-compose up -d
```

2. 打开 `http://localhost:3000`，用初始账号 `root`（密码在部署文档中，默认 `123456`）登录。

3. 创建知识库：**知识库 → 新建**，命名如"产品手册"，选择"文本"类型。

4. 上传 PDF：进入知识库 → **导入 → 文件导入**，选择本地 PDF，等待文件训练完成（页面上会显示切片数量）。

5. 新建应用：**应用 → 新建 → 简单编排**，在"AI 对话"节点把"知识库"选为刚建的产品手册，保存。

6. 在预览窗口提问，如"产品的保修期是多久？"，检查回答是否包含知识库中的内容。

**预期输出**：

- 知识库列表中显示上传的 PDF 及自动切片的数量
- 提问"保修期"这类问题时，回答内容与 PDF 原文一致，并附上来源引用

**常见错误**：

| 错误现象 | 原因 | 解决 |
|---------|------|------|
| 登录后提示无法连接 Mongo | 首次启动 Mongo 初始化较慢 | 重启容器：`docker-compose restart mongo` |
| 上传 PDF 后一直"训练中" | 向量模型/Embedding 未配置 | 在 OneAPI 中配置 embedding 模型并填入 key |
| 问答答非所问 | 知识库没关联到应用 | 在应用的 AI 对话节点中确认选择了该知识库 |
| 回答不准确 | 切片太大或太小 | 调整切片大小为 500-1000 字，重新训练 |

---

### 实验3：用 LangChain 写一个 Chain 调用 LLM

**目标**：用 Python 写一个 LangChain Chain，把"概念解释 + 翻译"串成固定流程，调用真实 LLM 并拿到结果。

**步骤**：

1. 创建项目并安装依赖：

```bash
mkdir langchain-demo && cd langchain-demo
python3 -m venv .venv && source .venv/bin/activate
pip install langchain langchain-openai
```

2. 创建 `chain_demo.py`（使用新版 `langchain-core` 的 `ChatPromptTemplate` + `pipe`，旧版 `LLMChain` 已弃用）：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="sk-xxx",          # 换成你的 Key，或通过环境变量 OPENAI_API_KEY 传入
    base_url="https://api.openai.com/v1",  # 国内可换成中转/本地服务地址
)

explain_prompt = ChatPromptTemplate.from_template(
    "请用一句话向初学者解释：{concept}"
)
translate_prompt = ChatPromptTemplate.from_template(
    "请把下面的中文翻译成英文：\n{explanation}"
)

# 用 pipe 串联两个环节：概念解释 → 翻译
chain = explain_prompt | llm | translate_prompt | llm

result = chain.invoke({"concept": "Transformer"})
print(result.content)
```

3. 运行：

```bash
python chain_demo.py
```

**预期输出**：终端打印两段内容——先用中文解释 Transformer，再输出对应的英文翻译。例如：

```
Transformer 是一种让机器同时处理整句话、捕捉词语之间关系的深度学习模型。
A Transformer is a deep learning model that processes an entire sentence at once and captures relationships between words.
```

**常见错误**：

| 错误现象 | 原因 | 解决 |
|---------|------|------|
| `ImportError: cannot import name 'OpenAI' from 'langchain'` | 老 API 已迁移到独立包 | 改用 `from langchain_openai import ChatOpenAI` |
| `OpenAIError: AuthenticationError` | API Key 错误或过期 | 检查 Key；确认 `base_url` 与 Key 所属平台一致 |
| 请求超时/连接失败 | 网络不通（国内直连 OpenAI 常见） | 配代理，或把 `base_url` 指向可用的中转/本地 Ollama 地址 |
| 输出只有中文没有翻译 | 链没串联成功 | 检查 `pipe` 的顺序，确认第二个 prompt 接收了 `explanation` 变量 |

---

## 7.7 自测题

1. Dify 和 LangChain 的区别是什么？（提示：低代码 vs 代码）

<details>
<summary>参考答案</summary>

Dify 是低代码平台，拖拽式搭建，适合非技术人员；LangChain 是代码框架，需要写 Python，灵活度更高，适合开发者。
</details>

2. FastGPT 的核心功能是什么？（提示：知识库+问答）

<details>
<summary>参考答案</summary>

上传文档 → 自动切片 → 向量化存储 → 基于检索的问答。适合企业知识库、客服场景。
</details>

3. LangChain 的 Chain 和 Agent 有什么区别？（提示：固定流程 vs 自主决策）

<details>
<summary>参考答案</summary>

Chain 是固定流程：输入→步骤1→步骤2→输出，顺序写死。Agent 是自主决策：AI 根据当前情况决定调用什么工具、执行什么步骤，更灵活但更不可控。
</details>

---

**上一章**：[第6章：编排层](../06-编排层/README.md) | **下一章**：[第8章：实战路径](../08-实战路径/README.md)
