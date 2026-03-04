# 三类识别最佳结果汇总（2026-03-04）

## 1. README 与教程比对结论
- 两份文档都明确了三类任务：字母识别、数字识别、健康场景识别。
- 健康任务“当前最优结果目录”在两份文档中一致：`d:\huxi\morse_train\health_bestsubset_pipeline_verify_20260304`。
- 数字任务在两份文档中给出的“当前状态指标”一致（`0.9199 / 0.9099` 等），但与当前实际最佳运行目录 `digit_task_20260301_new69_cuda` 中 `best_run.json` 的指标存在差异（当前目录里更高）。
- 字母任务“最终快照”指标在 README 中给出更完整（`artifacts_delivery_final/metrics.json`）。

## 2. 三类任务最佳结果（已复制并分类）

### A. 字母识别（A-Z）
- 来源目录：`d:\huxi\morse_train\artifacts_delivery_final`
- 复制位置：`01_letters_best\artifacts_delivery_final`
- 关键指标（metrics.json）：
  - `test_accuracy = 0.9527027027`
  - `test_macro_f1 = 0.9490108182`
  - `best_epoch = 90`
  - `best_val_macro_f1 = 0.9798697029`
- 预测汇总（词约束）：`01_letters_best\predictions_delivery_final_lex\decoding_summary.csv`

### B. 数字识别（0-9）
- 来源目录：`d:\huxi\morse_train\digit_task_20260301_new69_cuda`
- 复制位置：`02_digits_best\digit_task_20260301_new69_cuda`
- 采用最佳模型目录：`stage2\random_segments__ch1`
- 关键指标（best_run.json / metrics.json）：
  - `test_accuracy = 0.9897959184`
  - `test_macro_f1 = 0.9885714286`
  - `best_epoch = 97`
- 目标序列预测（final_prediction_summary.json）：
  - `raw_sequence = 0123456789`
  - `final_sequence = 0123456789`

### C. 健康场景识别（60s 分组）
- 来源目录：`d:\huxi\morse_train\health_bestsubset_pipeline_verify_20260304`
- 复制位置：`03_health_best\health_bestsubset_pipeline_verify_20260304`
- 最优方案：`seed 52 + 62` logits 集成
- 关键指标（pipeline_summary.json）：
  - fine(label): `accuracy = 0.9230769231`, `macro_f1 = 0.8712918660`
  - coarse_label: `accuracy = 0.8461538462`, `macro_f1 = 0.8770833333`
- 训练曲线与混淆矩阵：`03_health_best\health_training_figures_20260304`

## 3. 目录结构
- `01_letters_best/`：字母任务最佳训练产物 + 词约束预测汇总
- `02_digits_best/`：数字任务最佳运行产物（含模型、评估、预测）
- `03_health_best/`：健康任务最佳集成结果 + 训练/验证曲线 + 混淆矩阵

## 4. 说明
- 本次操作为“复制归档”，未修改原始训练结果目录。
