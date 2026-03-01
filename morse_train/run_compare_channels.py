import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.train import train_pipeline
from src.utils import ensure_dir, load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dual/ch1/ch2 training under identical settings and compare metrics."
    )
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training zips directory")
    parser.add_argument("--out_root", type=str, required=True, help="Root output directory for channel comparison")
    parser.add_argument(
        "--reference_config",
        type=str,
        default="artifacts_delivery_final/train_config.json",
        help="Reference train_config.json used to keep settings identical.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override random seed")
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional override max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override batch size")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Optional override early stopping patience",
    )
    parser.add_argument(
        "--no_rebuild_cache",
        action="store_true",
        help="Do not force cache rebuild in each run (default is rebuild).",
    )
    return parser.parse_args()


def _resolve_optional(override, config_value):
    return override if override is not None else config_value


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    out_root = ensure_dir(Path(args.out_root))
    ref_cfg_path = Path(args.reference_config)
    if not ref_cfg_path.exists():
        raise FileNotFoundError(f"reference_config not found: {ref_cfg_path}")
    ref_cfg = load_json(ref_cfg_path)

    seed = int(_resolve_optional(args.seed, ref_cfg.get("seed", 42)))
    rebuild_cache = not args.no_rebuild_cache

    common_kwargs = {
        "max_epochs": _resolve_optional(args.max_epochs, ref_cfg.get("max_epochs")),
        "batch_size": _resolve_optional(args.batch_size, ref_cfg.get("batch_size")),
        "early_stopping_patience": _resolve_optional(
            args.early_stopping_patience,
            ref_cfg.get("early_stopping_patience"),
        ),
        "split_mode": ref_cfg.get("split_mode"),
        "learning_rate": ref_cfg.get("learning_rate"),
        "weight_decay": ref_cfg.get("weight_decay"),
        "label_smoothing": ref_cfg.get("label_smoothing"),
        "mixup_alpha": ref_cfg.get("mixup_alpha"),
        "mixup_prob": ref_cfg.get("mixup_prob"),
        "model_variant": ref_cfg.get("model_variant"),
        "clip_low_pct": ref_cfg.get("clip_low_pct"),
        "clip_high_pct": ref_cfg.get("clip_high_pct"),
        "median_window": ref_cfg.get("median_window"),
        "smooth_window": ref_cfg.get("smooth_window"),
    }

    rows = []
    for mode in ("dual", "ch1", "ch2"):
        run_out = out_root / f"artifacts_{mode}"
        result = train_pipeline(
            train_dir=train_dir,
            out_dir=run_out,
            seed=seed,
            channel_mode=mode,
            rebuild_cache=rebuild_cache,
            **common_kwargs,
        )
        rows.append(
            {
                "channel_mode": mode,
                "best_epoch": int(result["best_epoch"]),
                "best_val_macro_f1": float(result["best_val_macro_f1"]),
                "test_accuracy": float(result["test_accuracy"]),
                "test_macro_f1": float(result["test_macro_f1"]),
                "artifacts_dir": str(run_out),
            }
        )

    dual_row = next((r for r in rows if r["channel_mode"] == "dual"), None)
    dual_acc = float(dual_row["test_accuracy"]) if dual_row is not None else 0.0
    dual_f1 = float(dual_row["test_macro_f1"]) if dual_row is not None else 0.0
    for r in rows:
        r["delta_accuracy_vs_dual"] = float(r["test_accuracy"]) - dual_acc
        r["delta_macro_f1_vs_dual"] = float(r["test_macro_f1"]) - dual_f1

    metrics_path = out_root / "comparison_metrics.csv"
    summary_path = out_root / "comparison_summary.json"

    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8")
    save_json(
        summary_path,
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_dir": str(train_dir),
            "out_root": str(out_root),
            "reference_config": str(ref_cfg_path),
            "seed": seed,
            "rebuild_cache": rebuild_cache,
            "channel_modes": ["dual", "ch1", "ch2"],
            "results": rows,
        },
    )

    print("comparison_metrics:", metrics_path)
    print("comparison_summary:", summary_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
