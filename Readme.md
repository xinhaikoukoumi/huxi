# huxi 项目运行说明

本目录当前包含 3 类任务代码与数据：

1. 字母识别（A-Z）
2. 数字识别（0-9）
3. 健康场景识别（60s 分组）

统一代码目录：`d:\huxi\morse_train`

## 1. 环境准备

``powershell +cd d:\huxi\morse_train +pip install -r requirements.txt +``

## 2. 三类任务如何运行

### 2.1 字母识别（A-Z）

训练：
``powershell +cd d:\huxi\morse_train +python run_train.py --train_dir d:\huxi\摩斯编码\AD+BC --out_dir d:\huxi\morse_train\artifacts_letters_dual --label_mode letters --channel_mode dual --seed 42 --max_epochs 120 --rebuild_cache +``

预测：
``powershell +cd d:\huxi\morse_train +python run_predict.py --model_path d:\huxi\morse_train\artifacts_letters_dual\morse_char_model.pt --label_map d:\huxi\morse_train\artifacts_letters_dual\label_map.json --input_dir d:\huxi\摩斯编码 --output_dir d:\huxi\morse_train\pred_letters_lex --lexicon_file d:\huxi\morse_train\lexicon_commands.txt +``

### 2.2 数字识别（0-9）

推荐一键流程（训练+搜索+目标文件预测）：
``powershell +cd d:\huxi\morse_train +python run_digit_pipeline.py --train_dir d:\huxi\数字编码\AD+BC数字编码 --target_file "d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip" +``

### 2.3 健康场景识别（60s 分组）

推荐深度学习训练（GPU）：
``powershell +cd d:\huxi\morse_train +python run_health_deep.py --input_dir d:\huxi\health_data_0302_0303_20260304_205759 --output_dir d:\huxi\morse_train\health_run_latest --target both --cv_mode stratified --model_variant enhanced_reslstm --max_epochs 60 --patience 12 --batch_size 32 --learning_rate 0.001 --weight_decay 0.0001 --label_smoothing 0 --mixup_alpha 0 --mixup_prob 0 --tta_shifts 0,-4,4 --merge_fine_labels --no_balanced_sampler --seed 42 +``

如果要做多 seed 集成（当前更推荐）：
``powershell +cd d:\huxi\morse_train +python run_health_bestsubset_pipeline.py --mode train_and_ensemble --input_dir d:\huxi\health_data_0302_0303_20260304_205759 --output_dir d:\huxi\morse_train\health_bestsubset_pipeline_latest --seed_train 42,52,62,72,82 --ensemble_seeds 52,62 --target both --cv_mode stratified --model_variant enhanced_reslstm --learning_rate 0.001 --weight_decay 0.0001 --batch_size 32 --label_smoothing 0 --mixup_alpha 0 --mixup_prob 0 --tta_shifts 0,-4,4 --no_use_balanced_sampler +``

## 3. 当前最终结果

`d:\huxi\morse_train\best_results_summary_20260304`

其中按类别分类如下：

- `d:\huxi\morse_train\best_results_summary_20260304\01_letters_best`
  +- `d:\huxi\morse_train\best_results_summary_20260304\02_digits_best`
  +- `d:\huxi\morse_train\best_results_summary_20260304\03_health_best`

总说明文件：

- `d:\huxi\morse_train\best_results_summary_20260304\BEST_RESULTS_SUMMARY.md`
   `d:\huxi\morse_train\best_results_summary_20260304\COPY_MANIFEST.csv`
