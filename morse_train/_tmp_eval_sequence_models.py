import json
import re
from pathlib import Path

import pandas as pd

from src.predict import predict_directory

OUT_ROOT = Path(r"d:/huxi/morse_train/digit_task_20260301_new69_cuda")
STAGE2_CSV = OUT_ROOT / "stage2_results.csv"
TEST_DIR = OUT_ROOT / "test_sequences"
PRED_ROOT = OUT_ROOT / "predictions_stage2_compare"
PRED_ROOT.mkdir(parents=True, exist_ok=True)

pat_seq = re.compile(r"AB[，,]([0-9]{10})-")
pat_legacy = re.compile(r"AB[，,]0-9[，,]")

def gt_from_name(name: str):
    m = pat_seq.search(name)
    if m:
        return m.group(1)
    if pat_legacy.search(name):
        return "0123456789"
    return None

def char_acc(gt: str, pred: str):
    m = min(len(gt), len(pred))
    matches = sum(1 for i in range(m) if gt[i] == pred[i])
    denom = max(len(gt), len(pred), 1)
    return matches / denom

stage2_df = pd.read_csv(STAGE2_CSV)
per_file_rows = []
model_rows = []

for row in stage2_df.itertuples(index=False):
    split_mode = str(row.split_mode)
    channel_mode = str(row.channel_mode)
    artifacts_dir = Path(str(row.artifacts_dir))
    model_path = artifacts_dir / "morse_char_model.pt"
    label_map_path = artifacts_dir / "label_map.json"
    tag = f"{split_mode}__{channel_mode}"
    pred_dir = PRED_ROOT / tag

    predict_directory(
        model_path=model_path,
        label_map_path=label_map_path,
        input_dir=TEST_DIR,
        output_dir=pred_dir,
        use_lexicon_decoder=False,
        auto_lexicon=False,
        force_lexicon=False,
    )

    summary = pd.read_csv(pred_dir / "decoding_summary.csv")
    model_file_rows = []
    for s in summary.itertuples(index=False):
        zip_name = str(s.zip_file)
        gt = gt_from_name(zip_name)
        pred = str(s.final_sequence)
        if gt is None:
            continue
        acc = char_acc(gt, pred)
        exact = 1 if gt == pred else 0
        r = {
            "model_tag": tag,
            "split_mode": split_mode,
            "channel_mode": channel_mode,
            "zip_file": zip_name,
            "gt_sequence": gt,
            "pred_sequence": pred,
            "segments": int(s.segments),
            "offset_sec": float(s.offset_sec),
            "mean_confidence": float(s.mean_confidence),
            "char_accuracy": float(acc),
            "exact_match": int(exact),
            "prediction_dir": str(pred_dir),
            "artifacts_dir": str(artifacts_dir),
        }
        model_file_rows.append(r)
        per_file_rows.append(r)

    if not model_file_rows:
        continue

    mdf = pd.DataFrame(model_file_rows)
    model_rows.append(
        {
            "model_tag": tag,
            "split_mode": split_mode,
            "channel_mode": channel_mode,
            "n_sequences": int(len(mdf)),
            "exact_match_rate": float(mdf["exact_match"].mean()),
            "mean_char_accuracy": float(mdf["char_accuracy"].mean()),
            "mean_confidence": float(mdf["mean_confidence"].mean()),
            "artifacts_dir": str(artifacts_dir),
            "prediction_dir": str(pred_dir),
            "stage2_test_accuracy": float(row.test_accuracy),
            "stage2_test_macro_f1": float(row.test_macro_f1),
            "stage2_best_val_macro_f1": float(row.best_val_macro_f1),
        }
    )

per_file_df = pd.DataFrame(per_file_rows)
per_model_df = pd.DataFrame(model_rows)

per_file_path = OUT_ROOT / "sequence_eval_per_file.csv"
per_model_path = OUT_ROOT / "sequence_eval_per_model.csv"
per_file_df.to_csv(per_file_path, index=False, encoding="utf-8")
per_model_df = per_model_df.sort_values(
    by=["exact_match_rate", "mean_char_accuracy", "mean_confidence"],
    ascending=[False, False, False],
)
per_model_df.to_csv(per_model_path, index=False, encoding="utf-8")

best = per_model_df.iloc[0].to_dict()
best_json = {
    "selection_metric": ["exact_match_rate", "mean_char_accuracy", "mean_confidence"],
    "best_model": best,
    "per_model_csv": str(per_model_path),
    "per_file_csv": str(per_file_path),
}
with (OUT_ROOT / "best_sequence_model.json").open("w", encoding="utf-8") as f:
    json.dump(best_json, f, ensure_ascii=False, indent=2)

print("saved", per_model_path)
print("saved", per_file_path)
print("best_model_tag", best["model_tag"])
print("best_exact_match_rate", best["exact_match_rate"])
print("best_mean_char_accuracy", best["mean_char_accuracy"])
