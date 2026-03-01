import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.predict import predict_directory
from src.train import train_pipeline
from src.utils import ensure_dir, save_json


DEFAULT_TRAIN_DIR = r"d:\huxi\数字编码\AD+BC数字编码"
DEFAULT_TARGET_FILE = r"d:\huxi\数字编码\AB，0-9，一整组-0206135836(2).zip"
DEFAULT_OUT_ROOT_BASE = r"d:\huxi\morse_train"

SPLIT_MODES = ["random_segments", "group_by_file", "group_by_file_stratified"]
CHANNEL_MODES = ["dual", "ch1", "ch2"]
HYPER_GRID = [
    {
        "learning_rate": 6e-4,
        "weight_decay": 1.5e-4,
        "loss_type": "ce",
        "focal_gamma": 2.0,
        "label_smoothing": 0.05,
        "mixup_alpha": 0.4,
        "mixup_prob": 0.6,
        "aug_shift_max": 18,
        "aug_noise_std": 0.03,
        "aug_scale_min": 0.80,
        "aug_scale_max": 1.20,
        "aug_drift_max": 0.10,
        "aug_time_mask_prob": 0.45,
        "aug_time_mask_max_width": 30,
    },
    {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "loss_type": "focal",
        "focal_gamma": 1.5,
        "label_smoothing": 0.00,
        "mixup_alpha": 0.2,
        "mixup_prob": 0.5,
        "aug_shift_max": 20,
        "aug_noise_std": 0.03,
        "aug_scale_min": 0.80,
        "aug_scale_max": 1.20,
        "aug_drift_max": 0.12,
        "aug_time_mask_prob": 0.45,
        "aug_time_mask_max_width": 36,
    },
    {
        "learning_rate": 4e-4,
        "weight_decay": 2e-4,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "label_smoothing": 0.00,
        "mixup_alpha": 0.2,
        "mixup_prob": 0.4,
        "aug_shift_max": 16,
        "aug_noise_std": 0.025,
        "aug_scale_min": 0.82,
        "aug_scale_max": 1.18,
        "aug_drift_max": 0.10,
        "aug_time_mask_prob": 0.40,
        "aug_time_mask_max_width": 30,
    },
    {
        "learning_rate": 6e-4,
        "weight_decay": 1e-4,
        "loss_type": "ce",
        "focal_gamma": 2.0,
        "label_smoothing": 0.00,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "aug_shift_max": 14,
        "aug_noise_std": 0.02,
        "aug_scale_min": 0.85,
        "aug_scale_max": 1.15,
        "aug_drift_max": 0.08,
        "aug_time_mask_prob": 0.35,
        "aug_time_mask_max_width": 24,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Digit task pipeline: search, full train, and single-file prediction.")
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR, help="Digit training zip directory.")
    parser.add_argument("--target_file", type=str, default=DEFAULT_TARGET_FILE, help="Single zip file for final prediction.")
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root directory. Defaults to d:\\huxi\\morse_train\\digit_task_<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--model_variant", type=str, default="enhanced_reslstm", help="Model variant.")

    parser.add_argument("--stage1_max_epochs", type=int, default=35, help="Stage1 max epochs.")
    parser.add_argument("--stage1_patience", type=int, default=8, help="Stage1 early stopping patience.")
    parser.add_argument("--stage2_max_epochs", type=int, default=120, help="Stage2 max epochs.")
    parser.add_argument("--stage2_patience", type=int, default=999, help="Stage2 early stopping patience.")
    parser.add_argument(
        "--stage1_limit",
        type=int,
        default=None,
        help="Optional limit for number of stage1 runs (for smoke/debug only).",
    )
    parser.add_argument("--skip_stage2", action="store_true", help="Skip stage2 full training.")
    parser.add_argument("--skip_predict", action="store_true", help="Skip final single-file prediction.")
    return parser.parse_args()


def _make_out_root(args) -> Path:
    if args.out_root:
        return ensure_dir(Path(args.out_root))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(DEFAULT_OUT_ROOT_BASE) / f"digit_task_{ts}")


def _stage1_runs(train_dir: Path, out_root: Path, args) -> pd.DataFrame:
    stage1_root = ensure_dir(out_root / "stage1")
    rows = []
    run_idx = 0
    for split_mode in SPLIT_MODES:
        for channel_mode in CHANNEL_MODES:
            for grid_idx, hp in enumerate(HYPER_GRID):
                run_idx += 1
                if args.stage1_limit is not None and run_idx > int(args.stage1_limit):
                    break
                out_dir = stage1_root / f"{split_mode}__{channel_mode}__g{grid_idx}"
                result = train_pipeline(
                    train_dir=train_dir,
                    out_dir=out_dir,
                    seed=args.seed,
                    label_mode="digits",
                    channel_mode=channel_mode,
                    max_epochs=args.stage1_max_epochs,
                    batch_size=args.batch_size,
                    early_stopping_patience=args.stage1_patience,
                    split_mode=split_mode,
                    learning_rate=hp["learning_rate"],
                    weight_decay=hp["weight_decay"],
                    loss_type=hp.get("loss_type", "ce"),
                    focal_gamma=hp.get("focal_gamma", 2.0),
                    label_smoothing=hp["label_smoothing"],
                    mixup_alpha=hp["mixup_alpha"],
                    mixup_prob=hp["mixup_prob"],
                    aug_shift_max=hp.get("aug_shift_max", 12),
                    aug_noise_std=hp.get("aug_noise_std", 0.02),
                    aug_scale_min=hp.get("aug_scale_min", 0.85),
                    aug_scale_max=hp.get("aug_scale_max", 1.15),
                    aug_drift_max=hp.get("aug_drift_max", 0.08),
                    aug_time_mask_prob=hp.get("aug_time_mask_prob", 0.35),
                    aug_time_mask_max_width=hp.get("aug_time_mask_max_width", 24),
                    model_variant=args.model_variant,
                    rebuild_cache=True,
                )
                row = {
                    "stage": "stage1",
                    "split_mode": split_mode,
                    "channel_mode": channel_mode,
                    "grid_idx": int(grid_idx),
                    "best_epoch": int(result["best_epoch"]),
                    "best_val_macro_f1": float(result["best_val_macro_f1"]),
                    "test_accuracy": float(result["test_accuracy"]),
                    "test_macro_f1": float(result["test_macro_f1"]),
                    "artifacts_dir": str(out_dir),
                    "learning_rate": float(hp["learning_rate"]),
                    "weight_decay": float(hp["weight_decay"]),
                    "loss_type": str(hp.get("loss_type", "ce")),
                    "focal_gamma": float(hp.get("focal_gamma", 2.0)),
                    "label_smoothing": float(hp["label_smoothing"]),
                    "mixup_alpha": float(hp["mixup_alpha"]),
                    "mixup_prob": float(hp["mixup_prob"]),
                    "aug_shift_max": int(hp.get("aug_shift_max", 12)),
                    "aug_noise_std": float(hp.get("aug_noise_std", 0.02)),
                    "aug_scale_min": float(hp.get("aug_scale_min", 0.85)),
                    "aug_scale_max": float(hp.get("aug_scale_max", 1.15)),
                    "aug_drift_max": float(hp.get("aug_drift_max", 0.08)),
                    "aug_time_mask_prob": float(hp.get("aug_time_mask_prob", 0.35)),
                    "aug_time_mask_max_width": int(hp.get("aug_time_mask_max_width", 24)),
                }
                rows.append(row)
            if args.stage1_limit is not None and run_idx > int(args.stage1_limit):
                break
        if args.stage1_limit is not None and run_idx > int(args.stage1_limit):
            break

    stage1_df = pd.DataFrame(rows)
    stage1_path = out_root / "stage1_results.csv"
    stage1_df.to_csv(stage1_path, index=False, encoding="utf-8")
    return stage1_df


def _pick_best_by_pair(stage1_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = stage1_df.sort_values(
        by=["split_mode", "channel_mode", "test_accuracy", "test_macro_f1", "best_val_macro_f1"],
        ascending=[True, True, False, False, False],
    )
    best_df = (
        sorted_df.groupby(["split_mode", "channel_mode"], as_index=False)
        .first()
        .reset_index(drop=True)
    )
    return best_df


def _stage2_runs(train_dir: Path, out_root: Path, best_df: pd.DataFrame, args) -> pd.DataFrame:
    stage2_root = ensure_dir(out_root / "stage2")
    rows = []
    for row in best_df.itertuples(index=False):
        out_dir = stage2_root / f"{row.split_mode}__{row.channel_mode}"
        result = train_pipeline(
            train_dir=train_dir,
            out_dir=out_dir,
            seed=args.seed,
            label_mode="digits",
            channel_mode=row.channel_mode,
            max_epochs=args.stage2_max_epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.stage2_patience,
            split_mode=row.split_mode,
            learning_rate=float(row.learning_rate),
            weight_decay=float(row.weight_decay),
            loss_type=str(row.loss_type),
            focal_gamma=float(row.focal_gamma),
            label_smoothing=float(row.label_smoothing),
            mixup_alpha=float(row.mixup_alpha),
            mixup_prob=float(row.mixup_prob),
            aug_shift_max=int(row.aug_shift_max),
            aug_noise_std=float(row.aug_noise_std),
            aug_scale_min=float(row.aug_scale_min),
            aug_scale_max=float(row.aug_scale_max),
            aug_drift_max=float(row.aug_drift_max),
            aug_time_mask_prob=float(row.aug_time_mask_prob),
            aug_time_mask_max_width=int(row.aug_time_mask_max_width),
            model_variant=args.model_variant,
            rebuild_cache=True,
        )
        rows.append(
            {
                "stage": "stage2",
                "split_mode": row.split_mode,
                "channel_mode": row.channel_mode,
                "best_epoch": int(result["best_epoch"]),
                "best_val_macro_f1": float(result["best_val_macro_f1"]),
                "test_accuracy": float(result["test_accuracy"]),
                "test_macro_f1": float(result["test_macro_f1"]),
                "artifacts_dir": str(out_dir),
                "learning_rate": float(row.learning_rate),
                "weight_decay": float(row.weight_decay),
                "loss_type": str(row.loss_type),
                "focal_gamma": float(row.focal_gamma),
                "label_smoothing": float(row.label_smoothing),
                "mixup_alpha": float(row.mixup_alpha),
                "mixup_prob": float(row.mixup_prob),
                "aug_shift_max": int(row.aug_shift_max),
                "aug_noise_std": float(row.aug_noise_std),
                "aug_scale_min": float(row.aug_scale_min),
                "aug_scale_max": float(row.aug_scale_max),
                "aug_drift_max": float(row.aug_drift_max),
                "aug_time_mask_prob": float(row.aug_time_mask_prob),
                "aug_time_mask_max_width": int(row.aug_time_mask_max_width),
            }
        )

    stage2_df = pd.DataFrame(rows)
    stage2_path = out_root / "stage2_results.csv"
    stage2_df.to_csv(stage2_path, index=False, encoding="utf-8")
    return stage2_df


def _pick_global_best(df: pd.DataFrame) -> dict:
    sorted_df = df.sort_values(
        by=["test_accuracy", "test_macro_f1", "best_val_macro_f1"],
        ascending=[False, False, False],
    )
    return sorted_df.iloc[0].to_dict()


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    target_file = Path(args.target_file)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not target_file.exists():
        raise FileNotFoundError(f"target_file not found: {target_file}")

    out_root = _make_out_root(args)

    stage1_df = _stage1_runs(train_dir=train_dir, out_root=out_root, args=args)
    if stage1_df.empty:
        raise RuntimeError("Stage1 produced no runs")
    best_stage1_by_pair = _pick_best_by_pair(stage1_df)
    best_stage1_path = out_root / "stage1_best_by_pair.csv"
    best_stage1_by_pair.to_csv(best_stage1_path, index=False, encoding="utf-8")

    if args.skip_stage2:
        best_row = _pick_global_best(stage1_df)
        source_stage = "stage1"
    else:
        stage2_df = _stage2_runs(train_dir=train_dir, out_root=out_root, best_df=best_stage1_by_pair, args=args)
        best_row = _pick_global_best(stage2_df)
        source_stage = "stage2"

    best_artifacts_dir = Path(best_row["artifacts_dir"])
    best_run = {
        "source_stage": source_stage,
        "best_run": best_row,
        "model_path": str(best_artifacts_dir / "morse_char_model.pt"),
        "label_map_path": str(best_artifacts_dir / "label_map.json"),
        "target_file": str(target_file),
        "output_root": str(out_root),
    }
    best_run_path = out_root / "best_run.json"
    save_json(best_run_path, best_run)

    if args.skip_predict:
        print("out_root:", out_root)
        print("best_run:", best_run_path)
        return

    pred_dir = ensure_dir(out_root / "prediction_final")
    prediction_rows = predict_directory(
        model_path=Path(best_run["model_path"]),
        label_map_path=Path(best_run["label_map_path"]),
        output_dir=pred_dir,
        input_file=target_file,
        use_lexicon_decoder=False,
        auto_lexicon=False,
        force_lexicon=False,
    )
    summary = {
        "best_run_path": str(best_run_path),
        "prediction_dir": str(pred_dir),
        "predicted_files": len(prediction_rows),
        "prediction_rows": prediction_rows,
    }
    final_summary_path = out_root / "final_prediction_summary.json"
    save_json(final_summary_path, summary)

    print("out_root:", out_root)
    print("stage1_results:", out_root / "stage1_results.csv")
    if args.skip_stage2:
        print("stage2_results: skipped")
    else:
        print("stage2_results:", out_root / "stage2_results.csv")
    print("best_run:", best_run_path)
    print("final_prediction_summary:", final_summary_path)


if __name__ == "__main__":
    main()
