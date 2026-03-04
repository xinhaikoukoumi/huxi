import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from run_health_deep import parse_seed_list, run_health_deep
from src.utils import ensure_dir, save_json


DEFAULT_INPUT_DIR = r"d:\huxi\health_data_0302_0303_20260304_205759"
DEFAULT_OUT_BASE = r"d:\huxi\morse_train"


def _parse_args():
    p = argparse.ArgumentParser(description="One-click train + best-subset logits ensemble for 0302/0303 health data.")
    p.add_argument("--mode", type=str, default="train_and_ensemble", choices=["train_and_ensemble", "ensemble_only"])
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", type=str, default=None, help="Default: health_bestsubset_pipeline_<timestamp>")
    p.add_argument("--runs_root", type=str, default=None, help="Where per-seed run dirs live. Default: <output_dir>/runs")
    p.add_argument("--seed_train", type=str, default="42,52,62,72,82")
    p.add_argument("--ensemble_seeds", type=str, default="52,62")
    p.add_argument("--seed_dir_template", type=str, default="seed_{seed}", help="Dir name template under runs_root.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip training for seeds with existing deep_summary.json.")
    p.add_argument("--no_resume", dest="resume", action="store_false")

    # Training hyperparams (current best for 0302+0303)
    p.add_argument("--target", type=str, default="both", choices=["coarse", "fine", "both"])
    p.add_argument("--window_sec", type=float, default=60.0)
    p.add_argument("--target_points", type=int, default=300)
    p.add_argument("--min_valid_ratio", type=float, default=0.6)
    p.add_argument("--feature_clip_low_pct", type=float, default=0.5)
    p.add_argument("--feature_clip_high_pct", type=float, default=99.5)
    p.add_argument("--median_window", type=int, default=5)
    p.add_argument("--smooth_window", type=int, default=9)
    p.add_argument("--cv_mode", type=str, default="stratified", choices=["group_by_day", "stratified"])
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--mixup_prob", type=float, default=0.0)
    p.add_argument("--model_variant", type=str, default="enhanced_reslstm", choices=["enhanced_reslstm", "baseline_cnn_bilstm"])
    p.add_argument("--tta_shifts", type=str, default="0,-4,4")
    p.add_argument("--merge_fine_labels", action="store_true", default=True)
    p.add_argument("--no_merge_fine_labels", dest="merge_fine_labels", action="store_false")
    p.add_argument("--use_balanced_sampler", action="store_true", default=False)
    p.add_argument("--no_use_balanced_sampler", dest="use_balanced_sampler", action="store_false")
    p.add_argument("--val_ratio", type=float, default=0.2)
    return p.parse_args()


def _resolve_output_dir(out_dir: str = None) -> Path:
    if out_dir:
        return ensure_dir(Path(out_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(DEFAULT_OUT_BASE) / f"health_bestsubset_pipeline_{ts}")


def _resolve_runs_root(output_dir: Path, runs_root: str = None) -> Path:
    if runs_root:
        return ensure_dir(Path(runs_root))
    return ensure_dir(output_dir / "runs")


def _seed_run_dir(runs_root: Path, seed_dir_template: str, seed: int) -> Path:
    return ensure_dir(runs_root / str(seed_dir_template).format(seed=int(seed)))


def _read_file_oof(run_dir: Path, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    csv_path = run_dir / f"deep_cv_{target_col}_file_oof.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing OOF csv: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Empty OOF csv: {csv_path}")

    logit_cols = sorted([c for c in df.columns if c.startswith("logit_")], key=lambda x: int(x.split("_")[1]))
    if not logit_cols:
        raise RuntimeError(f"No logit columns in: {csv_path}")

    # Defensive aggregation in case of accidental duplicate rows.
    key_cols = [c for c in ["file_id", "zip_file", "true_id", "true_label"] if c in df.columns]
    num_cols = [c for c in logit_cols if c in df.columns]
    agg_num = df[["file_id"] + num_cols].groupby("file_id", as_index=False).mean()
    if "true_id" in df.columns:
        true_df = df[["file_id", "true_id"]].drop_duplicates("file_id")
        agg_num = agg_num.merge(true_df, on="file_id", how="left")
    if "zip_file" in df.columns:
        zf = df[["file_id", "zip_file"]].drop_duplicates("file_id")
        agg_num = agg_num.merge(zf, on="file_id", how="left")
    if "true_label" in df.columns:
        tl = df[["file_id", "true_label"]].drop_duplicates("file_id")
        agg_num = agg_num.merge(tl, on="file_id", how="left")

    out_cols = ["file_id", "zip_file", "true_id", "true_label"] + logit_cols
    out_cols = [c for c in out_cols if c in agg_num.columns]
    return agg_num[out_cols].sort_values("file_id").reset_index(drop=True), logit_cols


def _metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(logits, axis=1).astype(np.int64)
    n_classes = int(logits.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_files": int(y_true.size),
        "n_classes": int(n_classes),
    }


def _ensemble_for_target(
    runs_root: Path,
    seed_dir_template: str,
    seeds_for_eval: List[int],
    ensemble_seeds: List[int],
    target_col: str,
    out_dir: Path,
) -> Dict[str, object]:
    seed_frames: Dict[int, pd.DataFrame] = {}
    seed_logits: Dict[int, np.ndarray] = {}
    logit_cols_ref: List[str] = []
    y_ref = None
    file_id_ref = None

    for s in seeds_for_eval:
        run_dir = _seed_run_dir(runs_root, seed_dir_template, s)
        df, logit_cols = _read_file_oof(run_dir, target_col=target_col)
        if not logit_cols_ref:
            logit_cols_ref = logit_cols
        elif logit_cols_ref != logit_cols:
            raise RuntimeError(f"Logit columns mismatch for seed {s}, target {target_col}")

        logits = df[logit_cols_ref].to_numpy(dtype=np.float64)
        y_true = df["true_id"].to_numpy(dtype=np.int64)
        fid = df["file_id"].to_numpy(dtype=np.int64)
        if file_id_ref is None:
            file_id_ref = fid
            y_ref = y_true
        else:
            if not np.array_equal(file_id_ref, fid):
                raise RuntimeError(f"file_id mismatch for seed {s}, target {target_col}")
            if not np.array_equal(y_ref, y_true):
                raise RuntimeError(f"true_id mismatch for seed {s}, target {target_col}")

        seed_frames[s] = df
        seed_logits[s] = logits

    for s in ensemble_seeds:
        if s not in seed_logits:
            raise RuntimeError(f"ensemble seed {s} not available in evaluated seeds for target={target_col}")

    single_rows = []
    for s in seeds_for_eval:
        m = _metrics_from_logits(y_ref, seed_logits[s])
        single_rows.append({"seed": int(s), **m})
    single_df = pd.DataFrame(single_rows).sort_values("seed").reset_index(drop=True)
    single_csv = out_dir / f"single_seed_{target_col}_metrics.csv"
    single_df.to_csv(single_csv, index=False, encoding="utf-8-sig")

    stack = np.stack([seed_logits[s] for s in ensemble_seeds], axis=0)
    logits_mean = np.mean(stack, axis=0)
    ens_metrics = _metrics_from_logits(y_ref, logits_mean)
    y_pred = np.argmax(logits_mean, axis=1).astype(np.int64)

    base_df = seed_frames[ensemble_seeds[0]][["file_id"]].copy()
    if "zip_file" in seed_frames[ensemble_seeds[0]].columns:
        base_df["zip_file"] = seed_frames[ensemble_seeds[0]]["zip_file"].astype(str)
    base_df["true_id"] = y_ref
    if "true_label" in seed_frames[ensemble_seeds[0]].columns:
        base_df["true_label"] = seed_frames[ensemble_seeds[0]]["true_label"].astype(str)
    base_df["pred_id_ensemble"] = y_pred
    for i, c in enumerate(logit_cols_ref):
        base_df[c] = logits_mean[:, i]
    ens_csv = out_dir / f"ensemble_{target_col}_file_oof.csv"
    base_df.to_csv(ens_csv, index=False, encoding="utf-8-sig")

    return {
        "target_col": target_col,
        "ensemble_seeds": [int(x) for x in ensemble_seeds],
        "ensemble_metrics": ens_metrics,
        "single_seed_metrics_csv": str(single_csv),
        "ensemble_file_oof_csv": str(ens_csv),
        "single_seed_mean_macro_f1": float(single_df["macro_f1"].mean()),
        "single_seed_std_macro_f1": float(single_df["macro_f1"].std(ddof=0)),
        "single_seed_mean_accuracy": float(single_df["accuracy"].mean()),
        "single_seed_std_accuracy": float(single_df["accuracy"].std(ddof=0)),
        "delta_macro_f1_vs_single_mean": float(ens_metrics["macro_f1"] - float(single_df["macro_f1"].mean())),
        "delta_accuracy_vs_single_mean": float(ens_metrics["accuracy"] - float(single_df["accuracy"].mean())),
        "single_seed_best_macro_f1": float(single_df["macro_f1"].max()),
        "single_seed_best_accuracy": float(single_df["accuracy"].max()),
    }


def _run_training(
    args,
    runs_root: Path,
    seeds_train: List[int],
) -> pd.DataFrame:
    rows = []
    for seed in seeds_train:
        run_dir = _seed_run_dir(runs_root, args.seed_dir_template, seed)
        summary_path = run_dir / "deep_summary.json"
        if bool(args.resume) and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = run_health_deep(
                input_dir=Path(args.input_dir),
                output_dir=run_dir,
                window_sec=float(args.window_sec),
                target_points=int(args.target_points),
                min_valid_ratio=float(args.min_valid_ratio),
                clip_low_pct=float(args.feature_clip_low_pct),
                clip_high_pct=float(args.feature_clip_high_pct),
                median_window=int(args.median_window),
                smooth_window=int(args.smooth_window),
                target=str(args.target),
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                label_smoothing=float(args.label_smoothing),
                mixup_alpha=float(args.mixup_alpha),
                mixup_prob=float(args.mixup_prob),
                model_variant=str(args.model_variant),
                tta_shifts=[int(x) for x in str(args.tta_shifts).split(",") if str(x).strip()],
                cv_mode=str(args.cv_mode),
                merge_fine_labels=bool(args.merge_fine_labels),
                use_balanced_sampler=bool(args.use_balanced_sampler),
                val_ratio=float(args.val_ratio),
                seed=int(seed),
                baseline_scan_dir=None,
            )

        row = {
            "seed": int(seed),
            "run_dir": str(run_dir),
            "n_files": int(summary.get("n_files", 0)),
            "n_segments": int(summary.get("n_segments", 0)),
        }
        if "coarse_label" in summary.get("targets", {}) and summary["targets"]["coarse_label"].get("status", "ok") != "failed":
            row["coarse_f1"] = float(summary["targets"]["coarse_label"]["file_macro_f1_mean"])
            row["coarse_acc"] = float(summary["targets"]["coarse_label"]["file_accuracy_mean"])
        else:
            row["coarse_f1"] = np.nan
            row["coarse_acc"] = np.nan
        if "label" in summary.get("targets", {}) and summary["targets"]["label"].get("status", "ok") != "failed":
            row["fine_f1"] = float(summary["targets"]["label"]["file_macro_f1_mean"])
            row["fine_acc"] = float(summary["targets"]["label"]["file_accuracy_mean"])
        else:
            row["fine_f1"] = np.nan
            row["fine_acc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def main():
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    runs_root = _resolve_runs_root(output_dir, args.runs_root)
    seeds_train = parse_seed_list(args.seed_train)
    ensemble_seeds = parse_seed_list(args.ensemble_seeds)

    if args.mode == "train_and_ensemble":
        train_df = _run_training(args=args, runs_root=runs_root, seeds_train=seeds_train)
    else:
        train_df = pd.DataFrame({"seed": seeds_train, "run_dir": [str(_seed_run_dir(runs_root, args.seed_dir_template, s)) for s in seeds_train]})

    train_csv = output_dir / "single_seed_run_summary.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")

    summary = {
        "task": "health_bestsubset_pipeline",
        "mode": str(args.mode),
        "created_at": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "runs_root": str(runs_root),
        "seeds_train": [int(x) for x in seeds_train],
        "ensemble_seeds": [int(x) for x in ensemble_seeds],
        "single_seed_run_summary_csv": str(train_csv),
        "train_params": {
            "target": str(args.target),
            "cv_mode": str(args.cv_mode),
            "model_variant": str(args.model_variant),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "mixup_alpha": float(args.mixup_alpha),
            "mixup_prob": float(args.mixup_prob),
            "tta_shifts": str(args.tta_shifts),
            "merge_fine_labels": bool(args.merge_fine_labels),
            "use_balanced_sampler": bool(args.use_balanced_sampler),
        },
        "targets": {},
    }

    if str(args.target) in ("coarse", "both"):
        summary["targets"]["coarse_label"] = _ensemble_for_target(
            runs_root=runs_root,
            seed_dir_template=args.seed_dir_template,
            seeds_for_eval=seeds_train,
            ensemble_seeds=ensemble_seeds,
            target_col="coarse_label",
            out_dir=output_dir,
        )
    if str(args.target) in ("fine", "both"):
        summary["targets"]["label"] = _ensemble_for_target(
            runs_root=runs_root,
            seed_dir_template=args.seed_dir_template,
            seeds_for_eval=seeds_train,
            ensemble_seeds=ensemble_seeds,
            target_col="label",
            out_dir=output_dir,
        )

    save_json(output_dir / "pipeline_summary.json", summary)

    print("output_dir:", str(output_dir))
    print("runs_root:", str(runs_root))
    if "coarse_label" in summary["targets"]:
        print("coarse_ensemble_macro_f1:", f"{summary['targets']['coarse_label']['ensemble_metrics']['macro_f1']:.6f}")
        print("coarse_ensemble_accuracy:", f"{summary['targets']['coarse_label']['ensemble_metrics']['accuracy']:.6f}")
    if "label" in summary["targets"]:
        print("fine_ensemble_macro_f1:", f"{summary['targets']['label']['ensemble_metrics']['macro_f1']:.6f}")
        print("fine_ensemble_accuracy:", f"{summary['targets']['label']['ensemble_metrics']['accuracy']:.6f}")


if __name__ == "__main__":
    main()

