# log-classifier

面向移动云智能运维平台日志分类的类脑鲁棒蒸馏项目。仓库交付内容聚焦三件事：源码、技术报告、测试报告；运行链路覆盖数据准备、教师训练、学生蒸馏、推理和验收测试。

## 交付目标

| 指标 | 验收要求 | 当前核心结果 |
| --- | ---: | ---: |
| Clean Accuracy | >= 90% | 90.4% |
| 同等功耗吞吐提升 | 比现有算法 >= 20% | 以 `baseline_throughput_samples_per_sec` 配置复测 |
| 高噪声准确率提升 | 比现有算法 >= 10% | 以 baseline 噪声评估复测 |

当前核心指标汇总见 `core_metrics_summary.json`，正式验收时应在目标硬件或同等功耗环境下重新运行评估脚本，并填入 baseline 吞吐。

## 仓库结构

```text
log-classifier/
├── configs/teacher/                         # 训练、蒸馏、评估配置
├── data/                                    # 样本数据与固定划分
├── src/log_classifier/
│   ├── data/                                # 数据加载、文本构造、划分
│   ├── inference/                           # checkpoint 推理 CLI
│   ├── teacher/                             # 类脑教师-学生蒸馏训练与评估
│   ├── training/                            # 通用指标
│   └── utils/                               # 随机种子等工具
├── baselines/                               # 现有算法对比脚本与结果
├── tests/                                   # 单元测试
├── TECHNICAL_REPORT.md                      # 技术报告
└── TEST_REPORT.md                           # 测试报告与复测命令
```

## 环境安装

推荐使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,dev]"
```

如果使用 `uv`：

```powershell
uv sync --extra train --extra dev
```

## 1. 数据准备

原始样本位于 `data/random_samples.jsonl`，固定划分位于 `data/random_samples_splits.json`。默认任务使用 `label3` 作为五分类标签：

- `Java Spring相关`
- `代码补全`
- `动态规划`
- `排序算法`
- `搜索算法`

训练脚本默认直接读取固定划分文件：

```text
data/random_samples_splits.json
```

如需重新构造数据，应保持每条样本至少包含：

```json
{"id": "sample-id", "text": "language: python user: ... assistant: ...", "label_text": "搜索算法"}
```

## 2. 教师模型训练

教师模型使用 `microsoft/unixcoder-base`，提供高质量语义表征和软标签监督。

```powershell
python -m log_classifier.teacher.train_stage1_ce `
  --config configs/teacher/ce_unixcoder_seed42.yaml
```

输出目录：

```text
outputs/teacher/ce_unixcoder_seed42/best/
```

关键产物包括 `pytorch_model.bin`、tokenizer 文件、`label_mapping.json`、`eval_results.json` 和 `config_snapshot.json`。

## 3. 学生模型蒸馏训练

学生模型采用 10 层 UniXcoder，通过 CE、KD、特征对齐、隐层对齐和 R-Drop 一致性学习教师模型的分类经验，作为最终推理模型。

```powershell
python -m log_classifier.teacher.train_distill_student `
  --config configs/teacher/distill_unixcoder_10layer_robust_seed42.yaml
```

输出目录：

```text
outputs/teacher/distill_unixcoder_10layer_robust_seed42/best/
```

## 4. 推理

单条文本推理：

```powershell
python -m log_classifier.inference.predict `
  --checkpoint-dir outputs/teacher/distill_unixcoder_10layer_robust_seed42/best `
  --model-name microsoft/unixcoder-base `
  --student-keep-layers 10 `
  --text "language: python user: 用 DFS 判断图中是否有环 assistant: 可以使用递归搜索和 visited 状态"
```

JSONL 批量推理：

```powershell
python -m log_classifier.inference.predict `
  --checkpoint-dir outputs/teacher/distill_unixcoder_10layer_robust_seed42/best `
  --model-name microsoft/unixcoder-base `
  --student-keep-layers 10 `
  --input-jsonl data/random_samples.jsonl `
  --output-jsonl outputs/predictions.jsonl
```

## 5. 测试与验收

单元测试：

```powershell
pytest
```

质量、鲁棒性和吞吐验收：

```powershell
python -m log_classifier.teacher.evaluate_distilled `
  --config configs/teacher/eval_distilled_kd_10layer.yaml
```

评估报告输出：

```text
outputs/teacher/distill_unixcoder_10layer_robust_seed42/eval_quality_report.json
```

要验证“同等功耗下吞吐提升 20%”，请在 `configs/teacher/eval_distilled_kd_10layer.yaml` 中填入现有算法在同硬件、同 batch、同功耗约束下测得的：

```yaml
baseline_throughput_samples_per_sec: 330.94
throughput_target_ratio: 1.20
```

要验证“高噪声准确率提升 10%”，请用同一测试集和同一 `noise_probs` 对现有算法运行噪声评估，再与本模型 `robust_average_accuracy` 做绝对提升对比。

## 报告文件

- `TECHNICAL_REPORT.md`：技术方案、类脑蒸馏机制、模型结构、实验结论。
- `TEST_REPORT.md`：测试范围、指标口径、复测步骤、验收记录。
- `core_metrics_summary.json`：历史核心指标摘要。
