<p align="center">
  <h1 align="center">OpenRS: Open Rubric System</h1>
  <p align="center">
    <em>以自适应细粒度评分标准替代传统 Reward Model 的 LLM 评测框架</em>
  </p>
</p>

<p align="center">
  <a href="https://github.com/WyxBUPT-22/OpenRS"><img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue" alt="English"></a>
</p>

---

## 简介

**OpenRS**（Open Rubric System）是一个 LLM-as-a-Judge 评测框架，通过自适应的、**针对不同问题类型的细粒度评分标准（Rubric）**，替代传统 Reward Model 来进行偏好评测。支持多维度打分，输出可解释的评测结论。

框架支持三种评测范式：

| 范式 | 适用场景 | 说明 |
| :---: | :--- | :--- |
| **Pairwise（对比评测）** | 对话、代码、安全等 | A/B 双向比较，按 Rubric 多维度加权打分 |
| **Verifiable（事实核查）** | 数学、事实性 | 先对照标准答案核查，若分不出再回退到 Pairwise |
| **Precise IF（指令遵循）** | 指令遵循 | 检查硬性约束是否满足，平局时回退到 Pairwise |

<p align="center">
  <img src="assests/framework.png" width="800"/>
  <br/>
  <em>图 1：OpenRS 评测流程 — 从 Pairwise 候选回答出发，经事实核查与自适应 Rubric 生成，最终多维度打分。</em>
</p>

## 核心特性

- 🎯 **Open Rubric**：50+ 问题类型专用评分标准，按权重分级（硬伤 / 核心 / 重要 / 亮点）
- ⚖️ **双向消偏**：交换 A/B 位置，消除位置偏差
- 🔍 **硬伤一票否决**：关键错误直接判定，不被其他优势掩盖
- 📊 **4 个基准测试**：[JudgeBench](https://arxiv.org/abs/2410.12784), [PPE](https://arxiv.org/abs/2410.14872), [RewardBench V2](https://arxiv.org/abs/2506.01937), [RMBench](https://arxiv.org/abs/2410.16184)


## 主要结果

我们在四个基准测试上评测了五个 Judge 模型：

<p align="center">
  <img src="assests/main_results.png" width="800"/>
  <br/>
  <em>表 1：不同 Judge 模型在四个基准测试上的准确率（%）。</em>
</p>

## 安装

```bash
git clone https://github.com/WyxBUPT-22/OpenRS.git
cd OpenRS
pip install -r requirements.txt
```

**依赖**：`openai`、`tenacity`、`json5`、`json-repair`、`tqdm`

## 快速开始

### 1. 配置 API

OpenRS 兼容所有 OpenAI 兼容的推理后端（vLLM、SGLang、Ollama 等）：

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL_NAME="your-model-name"
```

### 2. 运行评测

<details>
<summary><b>JudgeBench / PPE</b></summary>

```bash
python judgebench_and_ppe.py \
    --input data/judgebench/gpt.jsonl \
    --output-dir results/judgebench \
    --annotation judgebench_gpt \
    --workers 50
```

</details>

<details>
<summary><b>RewardBench V2</b></summary>

```bash
python rewardbench_v2.py \
    --input data/rewardbench_v2/rewardbench_v2.jsonl \
    --output-dir results/rewardbench_v2 \
    --annotation rbv2 \
    --workers 10
```

</details>

<details>
<summary><b>RMBench</b></summary>

```bash
python rmbench.py \
    --input data/rmbench/rmbench.jsonl \
    --output results/rmbench_results.jsonl \
    --workers 10
```

</details>

### 常用参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :---: |
| `--input` | 输入数据路径 | *必填* |
| `--output-dir` | 输出目录 | `./results` |
| `--workers` | 并发线程数 | 10–50 |
| `--temperature` | 生成温度 | 0.0 |
| `--limit` | 最大处理条数（0=不限制） | 0 |
| `--no-resume` | 禁用断点续传 | False |
| `--stats-only` | 仅统计（不运行评测） | False |

## 评测流程

```
输入数据 → 评测路由 → 模型调用 → 分数解析 → 结果聚合 → 统计报告
```

### 打分机制

每个维度按重要性加权：

| 类别 | 权重 | 说明 |
| :---: | :---: | :--- |
| **硬伤** | 一票否决 | 关键错误直接判定，忽略其他维度 |
| **核心** | ×5 | 关键质量维度 |
| **重要** | ×2 | 重要但非关键的因素 |
| **亮点** | ×1 | 加分项 |

### 各数据集处理逻辑

- **JudgeBench / PPE**：每条数据执行完整的 `evaluate_pair` — 先事实核查，再双向 Pairwise
- **RewardBench V2**：1-vs-N 比较，按 subset 路由评测策略（Chat、Math、Safety、Precise IF、Focus）；Tie 不计入准确率
- **RMBench**：9 种配对（chosen/rejected 各 3 变体）× 2 顺序 = 18 次评测；按 Easy / Normal / Hard 分层统计

## 项目结构

```
OpenRS/
├── tools.py                    # 基础工具（API 调用、JSON 解析、文件读写）
├── evaluator.py                # 核心评测接口（evaluate_pair）
├── evaluator_precise_if.py     # Precise IF（指令遵循）评测器
├── robust_utils.py             # 鲁棒性工具（Unicode / JSON 容错）
│
├── judgebench_and_ppe.py       # JudgeBench / PPE 评测脚本
├── rewardbench_v2.py           # RewardBench V2 评测脚本
├── rmbench.py                  # RMBench 评测脚本
│
├── prompts/
│   ├── pairwise_prompts/       # 50+ 分类专用 Pairwise 评分标准（.md）
│   ├── pointwise_prompts/      # Precise IF Prompt
│   └── verifiable_prompts/     # 事实核查 Prompt
│
├── data/                       # 评测数据集
│   ├── judgebench/
│   ├── ppe/
│   ├── rewardbench_v2/
│   └── rmbench/
│
├── requirements.txt
└── LICENSE                     # Apache License 2.0
```

## 输出格式

评测完成后，结果按 verdict 分流保存：

```
results/
├── all_results_{annotation}.jsonl            # 全部结果
├── verifiable_good_cases_{annotation}.jsonl   # 事实核查：chosen 胜
├── verifiable_bad_cases_{annotation}.jsonl    # 事实核查：rejected 胜
├── pairwise_good_cases_{annotation}.jsonl     # Pairwise：chosen 胜
├── pairwise_bad_cases_{annotation}.jsonl      # Pairwise：rejected 胜
├── pairwise_same_cases_{annotation}.jsonl     # Pairwise：平局
├── error_cases_{annotation}.jsonl             # 评测失败
└── summary_{annotation}.json                  # 汇总统计
```

## 引用

如果本项目对您有帮助，请引用：

```bibtex
@misc{openrs2025,
  title   = {Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric},
  year    = {2025},
  url     = {https://github.com/WyxBUPT-22/OpenRS}
}
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
