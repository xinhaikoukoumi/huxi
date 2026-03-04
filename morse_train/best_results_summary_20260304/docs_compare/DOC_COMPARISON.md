# README vs PROJECT_TUTORIAL 对比摘要（2026-03-04）

## 一致项
- 三类任务定义一致：字母识别、数字识别、健康场景识别。
- 健康任务当前最优目录一致：`d:\huxi\morse_train\health_bestsubset_pipeline_verify_20260304`。
- 健康任务最优指标一致：
  - fine: macro_f1 约 0.8713, accuracy 约 0.9231
  - coarse: macro_f1 约 0.8771, accuracy 约 0.8462
- 数字任务“旧口径状态说明”一致：`0.9199 / 0.9099` 及目标序列说明。

## 差异项
- README 含“当前交付结果（最终快照）”的字母任务指标段落更完整。
- PROJECT_TUTORIAL 更侧重执行步骤和参数解释。
- 数字任务文档指标未反映 `digit_task_20260301_new69_cuda` 中更高的新结果（best_run.json）。

## 建议
- 若要保持文档与产物严格同步，建议在两份文档都更新数字任务“当前最优指标”到最新运行目录口径。
