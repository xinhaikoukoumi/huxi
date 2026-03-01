import json
import re
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(r"d:/huxi/morse_train/digit_task_20260301_new69_cuda")
STAGE2_CSV = OUT_ROOT / "stage2_results.csv"
PRED_ROOT = OUT_ROOT / "predictions_stage2_compare"

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
stage2_map = {}
for r in stage2_df.itertuples(index=False):
    tag = f"{r.split_mode}__{r.channel_mode}"
    stage2_map[tag] = r

per_file_rows = []
model_rows = []

for pred_dir in sorted(PRED_ROOT.iterdir()):
    if not pred_dir.is_dir():
        continue
    summary_path = pred_dir / "decoding_summary.csv"
    if not summary_path.exists():
        continue
    tag = pred_dir.name
    if tag not in stage2_map:
        continue
    sr = stage2_map[tag]

    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    rows = []
    for rec in summary.to_dict(orient="records"):
        zip_name = str(rec.get("zip_file", ""))
        gt = gt_from_name(zip_name)
        if gt is None:
            continue
        pred = str(rec.get("final_sequence", ""))
        acc = char_acc(gt, pred)
        exact = 1 if gt == pred else 0
        row = {
            "model_tag": tag,
            "split_mode": str(sr.split_mode),
            "channel_mode": str(sr.channel_mode),
            "zip_file": zip_name,
            "gt_sequence": gt,
            "pred_sequence": pred,
            "segments": int(float(rec.get("segments", "0") or 0.0)),
            "offset_sec": float(rec.get("offset_sec", "0") or 0.0),
            "mean_confidence": float(rec.get("mean_confidence", "0") or 0.0),
            "char_accuracy": float(acc),
            "exact_match": int(exact),
            "prediction_dir": str(pred_dir),
            "artifacts_dir": str(sr.artifacts_dir),
        }
        rows.append(row)
        per_file_rows.append(row)

    if not rows:
        continue
    mdf = pd.DataFrame(rows)
    model_rows.append(
        {
            "model_tag": tag,
            "split_mode": str(sr.split_mode),
            "channel_mode": str(sr.channel_mode),
            "n_sequences": int(len(mdf)),
            "exact_match_rate": float(mdf["exact_match"].mean()),
            "mean_char_accuracy": float(mdf["char_accuracy"].mean()),
            "mean_confidence": float(mdf["mean_confidence"].mean()),
            "artifacts_dir": str(sr.artifacts_dir),
            "prediction_dir": str(pred_dir),
            "stage2_test_accuracy": float(sr.test_accuracy),
            "stage2_test_macro_f1": float(sr.test_macro_f1),
            "stage2_best_val_macro_f1": float(sr.best_val_macro_f1),
        }
    )

per_file_df = pd.DataFrame(per_file_rows)
per_model_df = pd.DataFrame(model_rows).sort_values(
    by=["exact_match_rate", "mean_char_accuracy", "mean_confidence"],
    ascending=[False, False, False],
)

per_file_path = OUT_ROOT / "sequence_eval_per_file.csv"
per_model_path = OUT_ROOT / "sequence_eval_per_model.csv"
per_file_df.to_csv(per_file_path, index=False, encoding="utf-8")
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

print("best_model_tag", best["model_tag"])
print("best_exact_match_rate", best["exact_match_rate"])
print("best_mean_char_accuracy", best["mean_char_accuracy"])
