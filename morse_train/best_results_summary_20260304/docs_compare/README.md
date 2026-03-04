# 摩斯编码呼吸信号识别项目说明（README）

> 完整总览与使用教程请见：`PROJECT_TUTORIAL.md`


## 1. 项目目标
本项目包含三条独立任务线：
- 摩斯字母识别（`A-Z`）
- 摩斯数字识别（`0-9`）
- 健康呼吸场景任务（非深度学习扫描 + 深度学习分类）

核心能力：
- 读取 `zip` 内 `csv/xlsx` 原始数据。
- 按任务配置窗口切段（字母/数字默认 `30s`，健康默认 `60s`），重采样为固定长度输入。
- 训练 26 类字母分类模型（A-Z）。
- 输出逐段字母与置信度。
- 支持词典约束解码，提升最终词级稳定性。
- 自动生成训练图、混淆矩阵图、预测可视化图与 `report.html`。

---

## 2. 项目结构
```text
morse_train/
├─ src/
│  ├─ config.py           # 训练配置
│  ├─ io_zip.py           # zip/csv/xlsx 读取与双表头解析
│  ├─ preprocess.py       # 清洗、滤波、分段、重采样、标准化
│  ├─ dataset.py          # 数据缓存、标签抽取、划分策略、数据增强
│  ├─ model.py            # 模型定义（enhanced_reslstm / baseline_cnn_bilstm）
│  ├─ train.py            # 训练主流程
│  ├─ predict.py          # 推理主流程（偏移搜索 + 词约束）
│  ├─ decoder.py          # 词典约束动态规划解码
│  ├─ evaluate.py         # accuracy / macro-F1 / recall / confusion matrix
│  ├─ visualize.py        # 曲线图、混淆矩阵图、预测信号图
│  ├─ health_analysis.py  # 健康数据特征提取/无监督分析/规则判别
│  └─ utils.py            # 通用工具
├─ tests/                 # 单元测试与集成测试
├─ run_train.py           # 训练入口
├─ run_predict.py         # 预测入口
├─ run_compare_channels.py # dual/ch1/ch2 一键对比入口
├─ run_digit_pipeline.py  # 数字任务两阶段搜索+单文件预测入口
├─ run_health_scan.py     # 健康任务一键扫描入口（非深度学习）
├─ run_health_deep.py     # 健康任务深度学习入口（GPU）
├─ run_health_bestsubset_pipeline.py # 健康任务多seed子集集成一键入口
├─ run_visualize.py       # 图像生成入口
├─ run_report.py          # HTML 报告生成入口
├─ run_freeze.py          # 封盘清理入口
├─ requirements.txt
├─ lexicon_commands.txt
├─ artifacts_delivery_final/           # 最终训练产物
└─ predictions_delivery_final_lex/     # 最终预测产物
```

---

## 3. 数据与标签规则

### 3.1 训练集
- 目录：`d:\huxi\摩斯编码\AD+BC`
- 文件：`*.zip`
- 标签抽取：文件名正则 `AB[,，]([A-Z])[,，]`

### 3.2 原始格式解析
`src/io_zip.py` 规则：
- 优先读取 zip 内第一个 `.csv`，若无则读取第一个 `.xlsx`。
- 表格前两行为双表头，从第 3 行开始为数据。
- 使用列：
  - `Time (s)` -> `time_s`
  - `Ch1` 的 `ΔR/R0 (%)` -> `ch1`
  - `Ch2` 的 `ΔR/R0 (%)` -> `ch2`

### 3.3 分段与输入形状
`src/preprocess.py`：
- 时间对齐：减去首时间戳。
- 分段窗口：`[30*k, 30*(k+1))`，尾段不足 30 秒丢弃。
- 每段重采样到 `300` 点，默认得到 `x.shape=(2,300)`（`dual`）。
- 训练可通过 `channel_mode` 切换为单通道：`ch1 -> (1,300)`，`ch2 -> (1,300)`。
- 段内有效采样比例 `<0.6` 丢弃。

---

## 4. 方法与算法

## 4.1 信号预处理（鲁棒）
每段每通道流程：
1. 缺失值线性插值补齐。
2. 百分位裁剪（默认 `0.5%~99.5%`）。
3. 中值滤波（默认窗口 `5`）。
4. 滑动平均（默认窗口 `9`）。
5. 去趋势（detrend）。
6. MAD 鲁棒标准化（median/MAD）。

### 4.2 训练数据增强
`src/dataset.py`（仅训练集）包含：
- 时间轴随机平移
- 通道幅值缩放
- 小斜率基线漂移
- 时间掩码（time masking）
- 高斯噪声

### 4.3 模型结构
`src/model.py` 支持两种结构：

1. `baseline_cnn_bilstm`
- Conv1d(2->32,k7)+BN+ReLU+MaxPool
- Conv1d(32->64,k5)+BN+ReLU+MaxPool
- BiLSTM(64->128)
- Temporal mean pooling
- FC(128->26)

2. `enhanced_reslstm`（默认）
- Stem Conv + 3个残差块
- 两次池化
- BiLSTM(96->192)
- Attention Pooling
- MLP Head -> 26类

### 4.4 训练策略
`src/train.py`：
- 损失：`CrossEntropy(weight=class_weights, label_smoothing)`
- 优化器：`AdamW`
- 学习率调度：`CosineAnnealingLR(T_max=max_epochs)`
- Mixup：可配置 `mixup_alpha` 和 `mixup_prob`
- 早停依据：`val_macro_f1`

训练日志口径：
- `train_*_aug`：增强口径（训练时 batch）
- `train_*_clean`：无增强口径（每轮末在 train_eval 上评估）
- `val_*`：验证集口径

### 4.5 推理与词约束
`src/predict.py + src/decoder.py`：
- 偏移搜索：在 `[0,30)` 内按 `offset_step_sec` 搜索最佳对齐。
- 可选自适应边界：边界附近最小梯度点细调。
- 段级输出：`pred_label/confidence/raw_pred_label/raw_confidence`。
- 词约束解码：
  - 基于动态规划将段概率序列与词典词对齐。
  - 支持插入/删除惩罚，优化整体路径分数。
  - 可强制词典（`force_lexicon`）以提升最终稳定性。

---

## 5. 运行方式

### 5.1 环境安装
```bash
cd d:\huxi\morse_train
pip install -r requirements.txt
```

### 5.2 训练
```bash
python run_train.py \
  --train_dir d:\huxi\摩斯编码\AD+BC \
  --out_dir d:\huxi\morse_train\artifacts_delivery_final \
  --label_mode letters \
  --channel_mode dual \
  --seed 42 \
  --max_epochs 120
```

单通道训练示例（其他条件相同）：
```bash
python run_train.py \
  --train_dir d:\huxi\摩斯编码\AD+BC \
  --out_dir d:\huxi\morse_train\artifacts_ch1 \
  --channel_mode ch1 \
  --seed 42 \
  --max_epochs 120
```

### 5.3 双/单通道一键对比
```bash
python run_compare_channels.py \
  --train_dir d:\huxi\摩斯编码\AD+BC \
  --out_root d:\huxi\morse_train\channel_compare \
  --reference_config d:\huxi\morse_train\artifacts_delivery_final\train_config.json
```
输出：
- `channel_compare/comparison_metrics.csv`
- `channel_compare/comparison_summary.json`

### 5.4 数字任务（独立建模 + 单文件预测）
```bash
python run_digit_pipeline.py \
  --train_dir d:\huxi\数字编码\AD+BC数字编码 \
  --target_file d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip
```
输出目录：`d:\huxi\morse_train\digit_task_时间戳`
关键产物：
- `stage1_results.csv`
- `stage2_results.csv`
- `best_run.json`
- `prediction_final/decoding_summary.csv`
- `final_prediction_summary.json`

### 5.4.1 数字任务最新状态（2026-02-26）
- 当前数字训练集为 `52` 文件（新增 `0/5/7/9` 共 `7` 个文件）。
- 新数据下同配置基线（`off=0,5,10,15,20,25`）：
  - `test_accuracy=0.9063`
  - `test_macro_f1=0.9018`
- 本轮全局指标最优单模型（`digit_new52_cand_focal_mixup`）：
  - `test_accuracy=0.9199`
  - `test_macro_f1=0.9099`
- 本轮目标序列口径最优单模型（`digit_new52_cand_dense_offsets`）：
  - 固定30s序列：`0133456749`（`8/10`）
  - 同模型全局指标：`test_accuracy=0.9148`、`test_macro_f1=0.9033`
- 推理集成（3模型概率平均，固定30s）可达：
  - 序列：`0123456749`（`9/10`）
  - 注意：这属于推理策略收益，不是单模型 test 指标提升。

### 5.4.2 数字任务评估口径建议（重要）
- 单条短序列结果只适合作为案例展示（case study），不建议作为主结论。
- 主结论应优先报告文件级隔离测试集的统计指标：`Accuracy`、`Macro-F1`、`per-class recall`、`confusion matrix`。
- 序列级建议在多条序列上报告 `Exact Match` 或字符错误率（CER），并结合 `mean ± std`（多随机种子或交叉验证）。

### 5.5 健康任务（非深度学习 + 深度学习）

#### 5.5.1 非深度学习可分性扫描
```bash
python run_health_scan.py \
  --input_dir d:\huxi\健康数据 \
  --feature_set freq_time_hybrid \
  --window_sec_for_stats 10 \
  --eval_mode coarse \
  --cv_mode group_by_day
```
默认输出目录：`d:\huxi\morse_train\health_scan_时间戳`

可选参数：
- `--feature_set {freq_only,freq_time_hybrid}`：频率特征或频率+时域混合特征。
- `--window_sec_for_stats`：按窗口提稳健统计，默认 `10` 秒。
- `--eval_mode {coarse,fine,both}`：4类、7类或两者同时评估。
- `--cv_mode {none,group_by_day}`：经典模型验证方式，默认按日期分组。

核心产物：
- `file_features.csv`
- `pairwise_distance.csv`
- `unsupervised_7class_metrics.json`
- `unsupervised_4class_metrics.json`
- `rule_predictions.csv`
- `rule_thresholds.json`
- `classical_model_metrics.csv`
- `grouped_cv_summary.json`
- `feature_importance.csv`
- `error_cases.csv`
- `health_summary.json`
- `health_brief.md`
- `figures/pca_7class.png`
- `figures/pca_4class.png`
- `figures/distance_heatmap.png`
- `figures/feature_boxplots.png`

#### 5.5.2 深度学习单次训练（GPU）
> 适用于 `0302+0303` 子集，推荐 `cv_mode=stratified`（避免按天切分导致类别错位）。

```bash
python run_health_deep.py \
  --input_dir d:\huxi\health_data_0302_0303_20260304_205759 \
  --output_dir d:\huxi\morse_train\health_tune_0302_0303_d2_nobal_nomix_lr1e3 \
  --target both \
  --cv_mode stratified \
  --model_variant enhanced_reslstm \
  --max_epochs 60 \
  --patience 12 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --label_smoothing 0 \
  --mixup_alpha 0 \
  --mixup_prob 0 \
  --tta_shifts 0,-4,4 \
  --merge_fine_labels \
  --no_balanced_sampler \
  --seed 42
```

标签合并口径（已内置）：
- `左鼻塞/右鼻塞 -> 鼻塞`
- `小跑/跑步 -> 锻炼`

#### 5.5.3 多seed子集集成（一键）
```bash
python run_health_bestsubset_pipeline.py \
  --mode train_and_ensemble \
  --input_dir d:\huxi\health_data_0302_0303_20260304_205759 \
  --output_dir d:\huxi\morse_train\health_bestsubset_pipeline_latest \
  --seed_train 42,52,62,72,82 \
  --ensemble_seeds 52,62 \
  --target both \
  --cv_mode stratified \
  --model_variant enhanced_reslstm \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --batch_size 32 \
  --label_smoothing 0 \
  --mixup_alpha 0 \
  --mixup_prob 0 \
  --tta_shifts 0,-4,4 \
  --no_use_balanced_sampler
```

一键流程产物：
- `pipeline_summary.json`
- `single_seed_run_summary.csv`
- `single_seed_label_metrics.csv`
- `single_seed_coarse_label_metrics.csv`
- `ensemble_label_file_oof.csv`
- `ensemble_coarse_label_file_oof.csv`

#### 5.5.4 2026-03-04 最优结果记录（当前）
- 最优集成方案：`seed 52 + 62`（logits 平均）
- 最优结果目录（推荐查看）：`d:\huxi\morse_train\health_bestsubset_pipeline_verify_20260304`
- 关键指标：
  - `fine`：`macro_f1=0.8713`，`accuracy=0.9231`
  - `coarse`：`macro_f1=0.8771`，`accuracy=0.8462`
- 对应汇总文件：`pipeline_summary.json`

### 5.6 预测（词约束）
```bash
python run_predict.py \
  --model_path d:\huxi\morse_train\artifacts_delivery_final\morse_char_model.pt \
  --label_map d:\huxi\morse_train\artifacts_delivery_final\label_map.json \
  --input_dir d:\huxi\摩斯编码 \
  --output_dir d:\huxi\morse_train\predictions_delivery_final_lex \
  --lexicon_file d:\huxi\morse_train\lexicon_commands.txt
```

单文件预测示例（`--input_dir` 与 `--input_file` 二选一）：
```bash
python run_predict.py \
  --model_path d:\huxi\morse_train\artifacts_delivery_final\morse_char_model.pt \
  --label_map d:\huxi\morse_train\artifacts_delivery_final\label_map.json \
  --input_file d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip \
  --output_dir d:\huxi\morse_train\predictions_single \
  --disable_lexicon_decoder \
  --disable_auto_lexicon
```

### 5.7 生成图像
```bash
python run_visualize.py \
  --artifacts_dir d:\huxi\morse_train\artifacts_delivery_final \
  --prediction_dir d:\huxi\morse_train\predictions_delivery_final_lex \
  --input_dir d:\huxi\摩斯编码
```

### 5.8 生成 HTML 报告
```bash
python run_report.py \
  --artifacts_dir d:\huxi\morse_train\artifacts_delivery_final \
  --prediction_dir d:\huxi\morse_train\predictions_delivery_final_lex \
  --input_dir d:\huxi\摩斯编码
```
默认输出：`artifacts_delivery_final/report.html`

### 5.9 封盘清理
```bash
# 先预览
python run_freeze.py --root_dir d:\huxi --project_dir d:\huxi\morse_train --dry_run

# 再执行
python run_freeze.py --root_dir d:\huxi --project_dir d:\huxi\morse_train --apply
```

---

## 6. 输出文件说明

### 6.1 训练产物（`artifacts_delivery_final`）
- `morse_char_model.pt`：模型权重
- `label_map.json`：类别映射
- `train_config.json`：训练配置快照
- `train_history.csv`：每轮训练/验证指标
- `metrics.json`：最终 test 指标
- `confusion_matrix.csv`：混淆矩阵数值
- `figures/`：曲线图与混淆矩阵图
- `report.html`：总报告

### 6.2 预测产物（`predictions_delivery_final_lex`）
- `*_segments.csv`：每个 zip 的逐段预测
- `decoding_summary.csv`：所有文件汇总
- `figures/*_prediction.png`：信号+分段+字母+置信度可视化

`*_segments.csv` 字段：
- `segment_idx`
- `start_s`
- `end_s`
- `pred_label`
- `confidence`
- `raw_confidence`
- `raw_pred_label`
- `offset_sec`

---

## 7. 当前交付结果（最终快照）
来自 `artifacts_delivery_final/metrics.json`：
- `best_epoch = 90`
- `best_val_macro_f1 = 0.9799`
- `test_accuracy = 0.9527`
- `test_macro_f1 = 0.9490`

词级预测（`predictions_delivery_final_lex`）已在最终交付中达到：
- `exact_match = 1.0000`
- `char_accuracy = 1.0000`

---

## 8. 为什么会出现 train 曲线低于 val
当启用 `mixup + label_smoothing + 强增强` 时，`train_aug` 指标通常会系统性低于验证集，这是正常现象，不代表模型异常。建议主要对比：
- `train_clean` vs `val`
- `val_macro_f1` 趋势
- 最终 test 指标与混淆矩阵

---

## 9. 测试
```bash
cd d:\huxi\morse_train
python -m pytest -q
```

测试覆盖：
- 双表头解析（csv/xlsx）
- 30s 分段数量
- 通道模式裁剪（`dual/ch1/ch2`）
- 重采样形状（`dual -> (2,300)`，单通道为 `(1,300)`）
- 分层分组划分逻辑
- 词约束解码逻辑
- 训练/预测冒烟集成测试（需开启慢测环境变量）

---

## 10. Git 与版本回退
当前已完成本地封盘提交：
- commit: `669b47ccb80747f81f16153295b3a2f1d5d1342d`
- tag: `freeze-20260221-delivery-v1`

可用命令：
```bash
git checkout freeze-20260221-delivery-v1
```

---

## 11. 后续优化建议
- 使用 `group_by_file_stratified` 做更严格泛化评估。
- 在报告中并列展示 `无词约束` 与 `词约束` 两套词级指标。
- 若后续加入新词，更新 `lexicon_commands.txt` 并重跑 `run_predict.py` + `run_report.py`。
