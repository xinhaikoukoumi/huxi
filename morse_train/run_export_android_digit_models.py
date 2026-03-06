import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from src.model import MorseCharModel
from src.utils import ensure_dir


DEFAULT_TASK_ROOT = Path(r"d:\huxi\morse_train\digit_task_20260301_new69_cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export best digit models (one per split_mode) to Android-ready "
            "TorchScript Lite format (.ptl)."
        )
    )
    parser.add_argument(
        "--task_root",
        type=str,
        default=str(DEFAULT_TASK_ROOT),
        help="Digit task root containing stage2_results.csv and stage2 artifacts.",
    )
    parser.add_argument(
        "--stage2_results",
        type=str,
        default=None,
        help="Optional path to stage2_results.csv. Defaults to <task_root>/stage2_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <task_root>/android_export_best3",
    )
    return parser.parse_args()


def _load_state_dict_with_backward_compat(model: torch.nn.Module, state_dict: Dict) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if any(str(k).startswith("net.") for k in state_dict.keys()):
            stripped = {k[4:]: v for k, v in state_dict.items() if str(k).startswith("net.")}
            model.load_state_dict(stripped, strict=False)
        else:
            raise


def _choose_best_by_split(stage2_df: pd.DataFrame) -> pd.DataFrame:
    ranked = stage2_df.sort_values(
        by=["split_mode", "test_accuracy", "test_macro_f1", "best_val_macro_f1"],
        ascending=[True, False, False, False],
    )
    return ranked.groupby("split_mode", as_index=False).first().reset_index(drop=True)


def _export_one(row: pd.Series, out_root: Path) -> Dict:
    split_mode = str(row["split_mode"])
    channel_mode = str(row["channel_mode"])
    artifacts_dir = Path(str(row["artifacts_dir"]))

    ckpt_path = artifacts_dir / "morse_char_model.pt"
    label_map_path = artifacts_dir / "label_map.json"
    train_cfg_path = artifacts_dir / "train_config.json"
    metrics_path = artifacts_dir / "metrics.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_variant = str(cfg.get("model_variant", "enhanced_reslstm"))
    input_channels = int(checkpoint.get("input_channels", 2))
    target_points = int(cfg.get("target_points", 300))
    num_classes = int(checkpoint.get("num_classes", 10))

    model = MorseCharModel(
        num_classes=num_classes,
        input_channels=input_channels,
        variant=model_variant,
    )
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"checkpoint missing state_dict: {ckpt_path}")
    _load_state_dict_with_backward_compat(model, state_dict)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(1, input_channels, target_points)
        traced = torch.jit.trace(model, dummy)
        logits = traced(dummy)
    if logits.ndim != 2:
        raise ValueError(f"unexpected logits shape for {ckpt_path}: {tuple(logits.shape)}")

    optimized = optimize_for_mobile(traced)

    export_dir = ensure_dir(out_root / f"{split_mode}__{channel_mode}")
    ptl_path = export_dir / "morse_char_model_android.ptl"
    optimized._save_for_lite_interpreter(str(ptl_path))

    shutil.copy2(label_map_path, export_dir / "label_map.json")
    if train_cfg_path.exists():
        shutil.copy2(train_cfg_path, export_dir / "train_config.json")
    if metrics_path.exists():
        shutil.copy2(metrics_path, export_dir / "metrics.json")

    model_meta = {
        "split_mode": split_mode,
        "channel_mode": channel_mode,
        "artifacts_dir": str(artifacts_dir),
        "checkpoint_path": str(ckpt_path),
        "android_model_path": str(ptl_path),
        "label_map_path": str(export_dir / "label_map.json"),
        "model_variant": model_variant,
        "input_channels": input_channels,
        "target_points": target_points,
        "num_classes": num_classes,
        "output_shape_example": list(logits.shape),
        "test_accuracy": float(row["test_accuracy"]),
        "test_macro_f1": float(row["test_macro_f1"]),
        "best_val_macro_f1": float(row["best_val_macro_f1"]),
    }
    with (export_dir / "android_export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    return model_meta


def main() -> None:
    args = parse_args()
    task_root = Path(args.task_root)
    stage2_path = Path(args.stage2_results) if args.stage2_results else task_root / "stage2_results.csv"
    if not stage2_path.exists():
        raise FileNotFoundError(f"stage2_results.csv not found: {stage2_path}")

    out_root = Path(args.out_dir) if args.out_dir else task_root / "android_export_best3"
    out_root = ensure_dir(out_root)

    stage2_df = pd.read_csv(stage2_path)
    needed = {"split_mode", "channel_mode", "test_accuracy", "test_macro_f1", "best_val_macro_f1", "artifacts_dir"}
    if not needed.issubset(set(stage2_df.columns)):
        missing = sorted(list(needed - set(stage2_df.columns)))
        raise ValueError(f"stage2_results missing required columns: {missing}")

    best_df = _choose_best_by_split(stage2_df)
    exports: List[Dict] = []
    for _, row in best_df.iterrows():
        exports.append(_export_one(row=row, out_root=out_root))

    manifest = {
        "task_root": str(task_root),
        "stage2_results_path": str(stage2_path),
        "out_root": str(out_root),
        "num_models": len(exports),
        "models": exports,
    }
    manifest_path = out_root / "android_export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    pd.DataFrame(exports).to_csv(out_root / "android_export_manifest.csv", index=False, encoding="utf-8")

    print(f"Exported {len(exports)} models to: {out_root}")
    print(f"Manifest: {manifest_path}")
    for item in exports:
        print(
            f"- {item['split_mode']} / {item['channel_mode']} -> "
            f"{item['android_model_path']}"
        )


if __name__ == "__main__":
    main()

