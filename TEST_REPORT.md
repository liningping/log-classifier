# 测试报告

| 项目 | 内容 |
| --- | --- |
| 项目名称 | 移动云智能运维平台日志分类 |
| 测试对象 | 10 层鲁棒蒸馏 UniXcoder Student |
| 测试数据 | `data/random_samples_splits.json` 中 test split |
| 分类任务 | 五分类日志分类 |
| 测试日期 | 2026-05-29 |

## 1. 测试范围

本报告覆盖以下测试：

| 类型 | 目的 | 入口 |
| --- | --- | --- |
| 单元测试 | 验证数据读取、增强函数、损失函数等基础逻辑 | `pytest` |
| Clean 测试 | 验证无噪声测试集分类准确率是否达到 90% | `evaluate_distilled.py` |
| 高噪声测试 | 验证 UNK token 缺失噪声下的平均准确率 | `evaluate_distilled.py` |
| 吞吐测试 | 验证同等环境下 samples/sec 是否达到现有算法 1.2 倍 | `evaluate_distilled.py` |
| 推理冒烟测试 | 验证 checkpoint 可以完成单条或批量预测 | `inference.predict` |

## 2. 指标口径

| 指标 | 计算方式 |
| --- | --- |
| Clean Accuracy | 原始 test split 上 `正确预测数 / 样本总数` |
| Robust Accuracy | `noise_probs` 各缺失率下 accuracy 的平均值 |
| Throughput | warmup 后多轮前向推理的平均 samples/sec |
| 吞吐提升 | `本模型 throughput / baseline throughput` |
| 高噪声提升 | `本模型 robust accuracy - baseline robust accuracy` |

噪声方式为 input-id 级别固定比例 UNK 替换：先 tokenize，再按比例将非 special token 替换为 `unk_token_id`，不重新解码或重新 tokenize。

## 3. 当前核心结果

历史核心结果来自 `core_metrics_summary.json`：

| 模型 | Clean Accuracy | Robust Accuracy | Throughput |
| --- | ---: | ---: | ---: |
| 10 层鲁棒蒸馏 UniXcoder Student | 0.9040 | 0.7620 | 614.37 samples/s |
| BERT baseline | 0.8720 | 0.6342 | 330.94 samples/s |
| RoBERTa baseline | 0.8680 | 0.5538 | 326.27 samples/s |
| ERNIE baseline | 0.8660 | 0.6740 | 326.69 samples/s |

基于上述历史结果，相对 BERT baseline：

| 验收项 | 计算 | 结果 |
| --- | --- | ---: |
| Clean Accuracy >= 90% | 0.9040 >= 0.9000 | 通过 |
| 吞吐提升 >= 20% | 614.37 / 330.94 = 1.856 | 通过 |
| 高噪声准确率提升 >= 10% | 0.7620 - 0.6342 = 0.1278 | 通过 |

正式验收建议在目标硬件和同等功耗限制下复测，以复测输出 JSON 为准。

## 4. 复测步骤

安装环境：

```powershell
pip install -e ".[train,dev]"
```

运行单元测试：

```powershell
pytest
```

训练教师模型：

```powershell
python -m log_classifier.teacher.train_stage1_ce `
  --config configs/teacher/ce_unixcoder_seed42.yaml
```

训练 10 层学生模型：

```powershell
python -m log_classifier.teacher.train_distill_student `
  --config configs/teacher/distill_unixcoder_10layer_robust_seed42.yaml
```

评估 clean、robust 和 throughput：

```powershell
python -m log_classifier.teacher.evaluate_distilled `
  --config configs/teacher/eval_distilled_kd_10layer.yaml
```

推理冒烟测试：

```powershell
python -m log_classifier.inference.predict `
  --checkpoint-dir outputs/teacher/distill_unixcoder_10layer_robust_seed42/best `
  --model-name microsoft/unixcoder-base `
  --student-keep-layers 10 `
  --text "language: python user: 使用 DFS 搜索图路径 assistant: 维护 visited 集合避免重复访问"
```

## 5. 验收配置

在 `configs/teacher/eval_distilled_kd_10layer.yaml` 中配置：

```yaml
clean_accuracy_target: 0.90
robust_accuracy_target: 0.7342
baseline_throughput_samples_per_sec: 330.94
throughput_target_ratio: 1.20
```

其中 `robust_accuracy_target` 应设置为现有算法高噪声准确率加 0.10；`baseline_throughput_samples_per_sec` 应使用同等功耗下现有算法实测吞吐。

## 6. 输出文件

评估脚本会生成：

```text
outputs/teacher/distill_unixcoder_10layer_robust_seed42/eval_quality_report.json
```

重点检查其中的：

- `summary.clean_accuracy_pass`
- `summary.robust_accuracy_pass`
- `summary.throughput_pass`
- `summary.throughput_lift`
- `summary.robust_average_accuracy`

三项 pass 均为 `true` 时，可判定模型满足本项目测试验收要求。
