# 摩斯呼吸识别项目总览与使用教程

## 1. 适用范围
本项目支持三类独立任务：
- 字母识别任务（`A-Z`）
- 数字识别任务（`0-9`）
- 健康场景任务（非深度学习扫描 + 深度学习分类）

统一输入为呼吸信号 `zip` 文件。字母/数字任务默认分段为 `30s`，健康任务默认分段为 `60s`。

## 2. 目录结构
```text
morse_train/
├─ src/
│  ├─ config.py
│  ├─ io_zip.py
│  ├─ preprocess.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  ├─ predict.py
│  ├─ decoder.py
│  ├─ evaluate.py
│  ├─ visualize.py
│  ├─ health_analysis.py
│  └─ utils.py
├─ tests/
├─ run_train.py
├─ run_predict.py
├─ run_compare_channels.py
├─ run_digit_pipeline.py
├─ run_health_scan.py
├─ run_health_deep.py
├─ run_health_bestsubset_pipeline.py
├─ run_visualize.py
├─ run_report.py
├─ run_freeze.py
├─ README.md
└─ PROJECT_TUTORIAL.md
```

## 3. 任务隔离规则（必须遵守）
为避免字母与数字互相污染，必须隔离：
- 训练时显式设置 `--label_mode letters` 或 `--label_mode digits`。
- 推理时 `morse_char_model.pt` 与 `label_map.json` 必须来自同一任务目录。
- 字母与数字不要共用同一 `out_dir`。
- 若目录复用，必须加 `--rebuild_cache`。

推荐命名：
- 字母实验目录：`artifacts_letters_*`
- 数字实验目录：`artifacts_digits_*` 或 `digit_task_*`

## 4. 数据格式与标签规则
通用数据格式：
- 输入为 `*.zip`，优先读取 zip 内首个 `csv`，否则首个 `xlsx`。
- 表格前两行为双表头，从第 3 行开始为数据。
- 必需列：`Time (s)`、`Ch1 ΔR/R0 (%)`、`Ch2 ΔR/R0 (%)`。

标签提取规则：
- 字母任务：`AB[，,]([A-Z])[，,]`
- 数字任务：`AB[，,]([0-9])[，,]`

常用目录（你的当前环境）：
- 字母训练集：`d:\huxi\摩斯编码\AD+BC`
- 数字训练集：`d:\huxi\数字编码\AD+BC数字编码`
- 数字目标文件：`d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip`

## 5. 环境安装
```powershell
cd d:\huxi\morse_train
pip install -r requirements.txt
python -m pytest -q
```

## 6. 字母任务完整教程

### 6.1 训练双通道基线（推荐起点）
```powershell
python run_train.py ^
  --train_dir d:\huxi\摩斯编码\AD+BC ^
  --out_dir d:\huxi\morse_train\artifacts_letters_dual ^
  --label_mode letters ^
  --channel_mode dual ^
  --split_mode random_segments ^
  --seed 42 ^
  --max_epochs 120 ^
  --rebuild_cache
```

### 6.2 训练单通道模型（ch1/ch2）
```powershell
python run_train.py ^
  --train_dir d:\huxi\摩斯编码\AD+BC ^
  --out_dir d:\huxi\morse_train\artifacts_letters_ch1 ^
  --label_mode letters ^
  --channel_mode ch1 ^
  --seed 42 ^
  --max_epochs 120 ^
  --rebuild_cache

python run_train.py ^
  --train_dir d:\huxi\摩斯编码\AD+BC ^
  --out_dir d:\huxi\morse_train\artifacts_letters_ch2 ^
  --label_mode letters ^
  --channel_mode ch2 ^
  --seed 42 ^
  --max_epochs 120 ^
  --rebuild_cache
```

### 6.3 一键对比 dual/ch1/ch2
```powershell
python run_compare_channels.py ^
  --train_dir d:\huxi\摩斯编码\AD+BC ^
  --out_root d:\huxi\morse_train\channel_compare_letters ^
  --reference_config d:\huxi\morse_train\artifacts_letters_dual\train_config.json
```

对比输出：
- `comparison_metrics.csv`
- `comparison_summary.json`

关键字段：
- `test_accuracy`
- `test_macro_f1`
- `delta_accuracy_vs_dual`
- `delta_macro_f1_vs_dual`

### 6.4 字母预测（目录批量）
```powershell
python run_predict.py ^
  --model_path d:\huxi\morse_train\artifacts_letters_dual\morse_char_model.pt ^
  --label_map d:\huxi\morse_train\artifacts_letters_dual\label_map.json ^
  --input_dir d:\huxi\摩斯编码 ^
  --output_dir d:\huxi\morse_train\pred_letters_lex ^
  --lexicon_file d:\huxi\morse_train\lexicon_commands.txt
```

### 6.5 字母预测（单文件）
`--input_dir` 与 `--input_file` 二选一：
```powershell
python run_predict.py ^
  --model_path d:\huxi\morse_train\artifacts_letters_dual\morse_char_model.pt ^
  --label_map d:\huxi\morse_train\artifacts_letters_dual\label_map.json ^
  --input_file d:\huxi\摩斯编码\某文件.zip ^
  --output_dir d:\huxi\morse_train\pred_letters_single ^
  --lexicon_file d:\huxi\morse_train\lexicon_commands.txt
```

### 6.6 固定30s纯模型输出（关闭偏移搜索和词典）
用于做严格段级评估：
```powershell
python run_predict.py ^
  --model_path d:\huxi\morse_train\artifacts_letters_dual\morse_char_model.pt ^
  --label_map d:\huxi\morse_train\artifacts_letters_dual\label_map.json ^
  --input_file d:\huxi\摩斯编码\某文件.zip ^
  --output_dir d:\huxi\morse_train\pred_letters_fixed30 ^
  --disable_offset_search ^
  --disable_adaptive_boundaries ^
  --disable_lexicon_decoder ^
  --disable_auto_lexicon
```

### 6.7 可视化与报告
```powershell
python run_visualize.py ^
  --artifacts_dir d:\huxi\morse_train\artifacts_letters_dual ^
  --prediction_dir d:\huxi\morse_train\pred_letters_lex ^
  --input_dir d:\huxi\摩斯编码

python run_report.py ^
  --artifacts_dir d:\huxi\morse_train\artifacts_letters_dual ^
  --prediction_dir d:\huxi\morse_train\pred_letters_lex ^
  --input_dir d:\huxi\摩斯编码
```

### 6.8 字母任务输出文件解读
训练目录：
- `morse_char_model.pt`
- `label_map.json`
- `train_config.json`
- `train_history.csv`
- `metrics.json`
- `confusion_matrix.csv`

预测目录：
- `*_segments.csv`（逐段预测）
- `decoding_summary.csv`（汇总）

## 7. 数字任务完整教程

### 7.1 数字单次训练（手动）
```powershell
python run_train.py ^
  --train_dir d:\huxi\数字编码\AD+BC数字编码 ^
  --out_dir d:\huxi\morse_train\artifacts_digits_ch2 ^
  --label_mode digits ^
  --channel_mode ch2 ^
  --split_mode group_by_file ^
  --seed 42 ^
  --max_epochs 120 ^
  --rebuild_cache
```

### 7.2 数字两阶段自动搜索（推荐）
```powershell
python run_digit_pipeline.py ^
  --train_dir d:\huxi\数字编码\AD+BC数字编码 ^
  --target_file d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip
```

当前数据建议：
- 训练集已更新为 `52` 个数字标注文件（新增 `0/5/7/9` 共 `7` 个文件）。
- 运行后建议同时查看：
  - `stage2_results.csv`（全局泛化指标）
  - 固定30s逐段结果（目标序列口径）

主要产物：
- `stage1_results.csv`
- `stage2_results.csv`
- `best_run.json`
- `prediction_final\*_segments.csv`
- `prediction_final\decoding_summary.csv`

### 7.3 数字固定30s逐段预测
```powershell
python run_predict.py ^
  --model_path d:\huxi\morse_train\artifacts_digits_ch2\morse_char_model.pt ^
  --label_map d:\huxi\morse_train\artifacts_digits_ch2\label_map.json ^
  --input_file d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip ^
  --output_dir d:\huxi\morse_train\pred_digits_fixed30 ^
  --disable_offset_search ^
  --disable_adaptive_boundaries ^
  --disable_lexicon_decoder ^
  --disable_auto_lexicon
```

## 8. 训练参数进阶说明
`run_train.py` 关键参数：
- 任务与通道：`--label_mode`、`--channel_mode`
- 划分方式：`--split_mode`
- 优化器参数：`--learning_rate`、`--weight_decay`
- 损失函数：`--loss_type {ce,focal}`、`--focal_gamma`
- 正则策略：`--label_smoothing`、`--mixup_alpha`、`--mixup_prob`
- 预处理：`--clip_low_pct`、`--clip_high_pct`、`--median_window`、`--smooth_window`
- 增强：
- `--aug_shift_max`
- `--aug_noise_std`
- `--aug_scale_min`、`--aug_scale_max`
- `--aug_drift_max`
- `--aug_time_mask_prob`
- `--aug_time_mask_max_width`

## 9. 当前数字任务状态记录
- 截至 `2026-02-26`，数字训练集为 `52` 文件（新增标签集中在 `0/5/7/9`）。
- 新数据下“同配置基线”（`off=0,5,10,15,20,25`）：
  - `test_accuracy=0.9063`
  - `test_macro_f1=0.9018`
- 本轮“全局指标最优单模型”（`digit_new52_cand_focal_mixup`）：
  - `test_accuracy=0.9199`
  - `test_macro_f1=0.9099`
- 本轮“目标序列口径最优单模型”（`digit_new52_cand_dense_offsets`）：
  - 目标序列固定30s：`0133456749`（`8/10`）
  - 同模型全局指标：`test_accuracy=0.9148`、`test_macro_f1=0.9033`
- 若做推理集成（3模型概率平均，固定30s）可达：
  - 目标序列：`0123456749`（`9/10`）
  - 说明：这是推理策略收益，不是单模型 test 指标提升。

### 9.1 为什么“基准提升”但短序列仍有错误
- 新增数据主要补到 `0/5/7/9`，对 `2/8` 等难分标签帮助有限。
- `test_accuracy/macro_f1` 是全测试集平均口径；单条 `10` 段序列是高方差口径。
- 序列级 exact match 对单点错误非常敏感（错 1 段即降为 `9/10`）。
- 边界对齐（offset / adaptive boundary）会显著影响个别段预测。

### 9.2 是否可以只用一条短序列展示模型能力
- 可以展示“案例效果”（case study），但不应作为主结论。
- 建议把短序列放在“可视化示例”部分，并与全局指标并列展示。
- 对外结论应以文件级隔离测试集统计结果为主。

### 9.3 论文常见评估口径（推荐落地）
- 段级主指标：`Accuracy`、`Macro-F1`、`per-class recall`、`confusion matrix`。
- 序列级指标：多条序列上的 `Exact Match`、字符错误率（CER，或等价指标）。
- 稳定性：多随机种子或交叉验证，报告 `mean ± std`。
- 消融：通道模式、offset 搜索、边界自适应、增强策略分别报告增益。

续跑命令：
```powershell
python run_digit_pipeline.py ^
  --train_dir d:\huxi\数字编码\AD+BC数字编码 ^
  --target_file d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip
```

## 10. 常见问题
- `Cached channel count mismatch`
- 说明缓存通道数与当前 `channel_mode` 不一致。
- 处理：换新 `--out_dir` 或加 `--rebuild_cache`。

- `No labeled zip files matched AB,<DIGIT>, pattern`
- 说明 `--label_mode digits` 下训练目录命名不符合数字标签规则。

- `input_dir/input_file` 参数错误
- 两者必须二选一，不能同时为空，也不能同时提供。

- 训练后出现 CUDA DLL 警告但文件已产出
- 若 `metrics.json`、模型与预测结果都已生成，通常不影响本轮结果。

## 11. 归档与封盘
```powershell
python run_freeze.py --root_dir d:\huxi --project_dir d:\huxi\morse_train --dry_run
python run_freeze.py --root_dir d:\huxi --project_dir d:\huxi\morse_train --apply
```

## 12. 健康数据任务（非深度学习 + 深度学习）

### 12.1 使用场景与限制
- 适用于 `d:\huxi\健康数据` 这类按场景命名的呼吸信号 `zip` 文件。
- 当前口径是“可分性评估”，主方法为无监督 + 规则判别，不依赖深度学习。
- 当存在“每类仅1文件”时，严格文件级监督评估不可用；报告会明确写明该限制。

### 12.2 一键运行
```powershell
python run_health_scan.py ^
  --input_dir d:\huxi\健康数据 ^
  --feature_set freq_time_hybrid ^
  --window_sec_for_stats 10 ^
  --eval_mode coarse ^
  --cv_mode group_by_day
```

可选参数：
- `--output_dir`：指定输出目录（默认 `health_scan_时间戳`）
- `--recursive`：递归扫描子目录
- `--segment_window_sec`：大于 0 时附加分段统计特征（默认 `0`，整文件）
- `--feature_set`：`freq_only` 或 `freq_time_hybrid`（推荐）
- `--window_sec_for_stats`：窗口稳健统计长度，默认 `10s`
- `--eval_mode`：`coarse/fine/both`，默认 `coarse`
- `--cv_mode`：`none/group_by_day`，默认 `group_by_day`
- `--seed`：随机种子（默认 `42`）
- `--no_save_plots`：不生成图表

### 12.3 输出解读
输出目录：`d:\huxi\morse_train\health_scan_<timestamp>`

核心文件：
- `file_features.csv`：每个 zip 一行的统计/频域/跨通道特征。
- `pairwise_distance.csv`：样本间欧氏距离矩阵（标准化特征空间）。
- `unsupervised_7class_metrics.json`：7 场景无监督结果（PCA/KMeans/Agglomerative + silhouette/DBI/ARI/NMI）。
- `unsupervised_4class_metrics.json`：4 大类（`normal/nasal_block/exercise/cough`）无监督结果。
- `rule_predictions.csv`：规则判别输出（含 `pred_label`、`trigger_rule`、`key_feature_values`）。
- `rule_thresholds.json`：规则阈值（由数据分位数自动定标）。
- `classical_model_metrics.csv`：LR/SVM/RF 的 `accuracy/macro_f1/macro_recall` 与每类召回。
- `grouped_cv_summary.json`：分组验证策略、折数、混淆矩阵等摘要。
- `feature_importance.csv`：随机森林/逻辑回归的特征重要度。
- `error_cases.csv`：经典模型验证中的错例清单。
- `health_summary.json`：机器可读总结。
- `health_brief.md`：图表版简报文字结论。

图表文件（`figures/`）：
- `pca_7class.png`
- `pca_4class.png`
- `distance_heatmap.png`
- `feature_boxplots.png`

### 12.4 标签与粗类映射（固定）
- `口呼吸`、`鼻子呼吸` -> `normal`
- `左鼻塞`、`右鼻塞` -> `nasal_block`
- `小跑`、`跑步` -> `exercise`
- `咳嗽` -> `cough`

### 12.5 深度学习训练（GPU）
推荐场景：`0302+0303` 数据子集，使用 `stratified` 文件级划分。

推荐命令（当前最优单模型参数）：
```powershell
python run_health_deep.py ^
  --input_dir d:\huxi\health_data_0302_0303_20260304_205759 ^
  --output_dir d:\huxi\morse_train\health_tune_0302_0303_d2_nobal_nomix_lr1e3 ^
  --target both ^
  --cv_mode stratified ^
  --model_variant enhanced_reslstm ^
  --max_epochs 60 ^
  --patience 12 ^
  --batch_size 32 ^
  --learning_rate 0.001 ^
  --weight_decay 0.0001 ^
  --label_smoothing 0 ^
  --mixup_alpha 0 ^
  --mixup_prob 0 ^
  --tta_shifts 0,-4,4 ^
  --merge_fine_labels ^
  --no_balanced_sampler ^
  --seed 42
```

说明：
- 数据分段按 `60s` 组信号处理。
- 合并标签规则（脚本内置）：
  - `左鼻塞/右鼻塞 -> 鼻塞`
  - `小跑/跑步 -> 锻炼`

### 12.6 一键多 seed 集成（推荐）
`run_health_bestsubset_pipeline.py` 支持：
- 训练多个 seed
- 按指定 seed 子集做文件级 OOF logits 平均集成
- 自动输出单 seed 与集成对比报告

一键命令：
```powershell
python run_health_bestsubset_pipeline.py ^
  --mode train_and_ensemble ^
  --input_dir d:\huxi\health_data_0302_0303_20260304_205759 ^
  --output_dir d:\huxi\morse_train\health_bestsubset_pipeline_latest ^
  --seed_train 42,52,62,72,82 ^
  --ensemble_seeds 52,62 ^
  --target both ^
  --cv_mode stratified ^
  --model_variant enhanced_reslstm ^
  --learning_rate 0.001 ^
  --weight_decay 0.0001 ^
  --batch_size 32 ^
  --label_smoothing 0 ^
  --mixup_alpha 0 ^
  --mixup_prob 0 ^
  --tta_shifts 0,-4,4 ^
  --no_use_balanced_sampler
```

### 12.7 今日工作记录（2026-03-04）
- 修复并完善健康标签解析与一致化。
- 增加健康深度学习脚本 `run_health_deep.py`，支持：
  - `cv_mode`（`group_by_day/stratified`）
  - 标签合并
  - mixup / label smoothing / TTA / balanced sampler 组合调参
- 增加一键 pipeline `run_health_bestsubset_pipeline.py`。
- 新增文件级 OOF logits 导出，支持跨 seed 真正 logits 集成。

### 12.8 当前最优结果路径与指标
最优结果文件夹（当前推荐）：
- `d:\huxi\morse_train\health_bestsubset_pipeline_verify_20260304`

核心文件：
- `pipeline_summary.json`
- `single_seed_run_summary.csv`
- `ensemble_label_file_oof.csv`
- `ensemble_coarse_label_file_oof.csv`

当前最优指标（seed 子集 `52+62`）：
- `fine`: `macro_f1=0.8713`，`accuracy=0.9231`
- `coarse`: `macro_f1=0.8771`，`accuracy=0.8462`
