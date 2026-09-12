---
title: 第15章 MLOps
tags: ['ai-infra', '教程', 'mlops', 'mlflow']
章节: 15
状态: 完成
---

# 第15章：MLOps

> MLOps = 餐厅的中央厨房管理系统。菜谱（代码）、食材（数据）、出品（模型）都要统一管理。

## 15.1 问题：模型训练实验太多，怎么管理？

你训练了 100 个模型，但：
- 不记得哪个模型用了什么数据、什么参数
- 找不到上周效果最好的模型
- 新同事不知道怎么复现你的实验

**核心矛盾**：实验管理混乱 = 重复劳动 + 无法复现。

## 15.2 方案：MLOps 四大支柱

| 支柱 | 一句话 | 工具 |
|------|--------|------|
| **实验跟踪** | 记录每次实验的参数和结果 | MLflow、W&B |
| **模型注册** | 统一管理模型版本 | MLflow Model Registry |
| **版本控制** | 代码+数据+模型版本对应 | Git + DVC |
| **CI/CD** | 自动化训练、测试、部署 | GitHub Actions、GitLab CI |

## 15.3 项目

| 项目 | ⭐ | 定位 | 一句话 |
|------|-----|------|--------|
| **MLflow** | 27.7k | 实验跟踪+模型注册 | 最全面 |
| **W&B** | 9k | 实验跟踪 | 界面最好看 |
| **DVC** | 14k | 数据版本控制 | Git for Data |
| **GitHub Actions** | - | CI/CD | 自动化流水线 |

## 15.4 代码示例

### MLflow 跟踪实验

```python
import mlflow
import mlflow.pytorch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 开始实验
mlflow.start_run()

# 记录参数
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 5)

# 训练模型
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# 记录指标
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# 保存模型
mlflow.sklearn.log_model(model, "model")

# 结束实验
mlflow.end_run()

print(f"实验完成，准确率：{accuracy:.2f}")
```

### MLflow 模型注册

```python
# 注册模型
mlflow.register_model(
    "runs:/<run_id>/model",
    "iris-classifier"
)

# 加载模型
model = mlflow.pyfunc.load_model("models:/iris-classifier/1")
predictions = model.predict(X_test)
```

### GitHub Actions 自动化训练

```yaml
# .github/workflows/train.yml
name: Train Model
on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Train model
      run: python train.py
    - name: Log to MLflow
      run: mlflow log_artifact model.pkl
```

## 15.5 排障：MLOps 常见问题

| 问题 | 现象 | 根因 | 解法 |
|------|------|------|------|
| 实验结果不可复现 | 同样的代码结果不一样 | 随机种子没固定 | 设 `torch.manual_seed(42)` |
| 模型版本混乱 | 不知道哪个模型是最新的 | 没有统一注册 | 用 MLflow Model Registry |
| 数据版本对不上 | 代码是新的，数据是旧的 | 数据没有版本控制 | 用 DVC 管理数据 |
| CI/CD 失败 | 训练脚本在 CI 里报错 | 环境不一致 | 用 Docker 统一环境 |

## 15.6 动手实验：用 MLflow 跟踪一个训练实验

**目标**：体验 MLflow 的实验跟踪功能。

**步骤**：

```bash
# 1. 安装
pip install mlflow

# 2. 启动 MLflow UI
mlflow ui

# 3. 运行训练脚本（上面的代码）
python train.py

# 4. 打开浏览器查看
open http://localhost:5000
```

**预期输出**：
- MLflow UI 显示实验列表
- 每次实验记录参数、指标、模型文件

**常见错误**：
- MLflow UI 打不开 → 检查端口是否被占用
- 实验没记录 → 检查 `mlflow.start_run()` 和 `mlflow.end_run()` 是否配对

## 15.7 自测题

1. MLflow 的四个核心功能是什么？（提示：跟踪、项目、模型、注册）

<details>
<summary>参考答案</summary>

①Tracking（实验跟踪）：记录参数、指标、文件；②Projects（项目）：打包代码，可复现；③Models（模型）：统一管理模型格式；④Registry（注册）：模型版本管理和生命周期。
</details>

2. 为什么需要 DVC？（提示：Git 不适合管理大数据）

<details>
<summary>参考答案</summary>

Git 适合管理代码（文本文件，体积小），不适合管理大数据（GB/TB 级）。DVC 把数据存到远程存储（S3/MinIO），Git 只存元数据（.dvc 文件），实现"代码+数据"的版本对应。
</details>

3. 实验不可复现的常见原因有哪些？（提示：随机种子、环境、数据版本）

<details>
<summary>参考答案</summary>

①随机种子没固定；②环境不一致（Python 版本、依赖版本）；③数据版本不一致；④代码版本不一致。解法：固定随机种子 + 用 Docker 统一环境 + DVC 管理数据 + Git 管理代码。
</details>

---

**上一章**：[第14章：模型部署进阶](../14-模型部署进阶/README.md) | **下一章**：[第16章：边缘部署](../16-边缘部署/README.md)
