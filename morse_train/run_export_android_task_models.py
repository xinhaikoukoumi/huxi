import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from src.model import MorseCharModel
from src.utils import ensure_dir, load_json


DEFAULT_LETTERS_MODEL = Path(r"d:\huxi\morse_train\artifacts_delivery_final\morse_char_model.pt")
DEFAULT_LETTERS_LABEL_MAP = Path(r"d:\huxi\morse_train\artifacts_delivery_final\label_map.json")
DEFAULT_LETTERS_CONFIG = Path(r"d:\huxi\morse_train\artifacts_delivery_final\train_config.json")

DEFAULT_DIGITS_MODEL = Path(
    r"d:\huxi\morse_train\digit_task_20260301_new69_cuda\stage2\random_segments__ch1\morse_char_model.pt"
)
DEFAULT_DIGITS_LABEL_MAP = Path(
    r"d:\huxi\morse_train\digit_task_20260301_new69_cuda\stage2\random_segments__ch1\label_map.json"
)
DEFAULT_DIGITS_CONFIG = Path(
    r"d:\huxi\morse_train\digit_task_20260301_new69_cuda\stage2\random_segments__ch1\train_config.json"
)

DEFAULT_OUT_DIR = Path(r"d:\huxi\morse_train\android_delivery_task_models")


@dataclass
class ModelSpec:
    model_id: str
    task: str
    checkpoint_path: Path
    label_map_path: Path
    train_config_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export 4 task-level models to Android TorchScript Lite (.ptl)")
    p.add_argument("--letters_model", type=str, default=str(DEFAULT_LETTERS_MODEL))
    p.add_argument("--letters_label_map", type=str, default=str(DEFAULT_LETTERS_LABEL_MAP))
    p.add_argument("--letters_config", type=str, default=str(DEFAULT_LETTERS_CONFIG))

    p.add_argument("--digits_model", type=str, default=str(DEFAULT_DIGITS_MODEL))
    p.add_argument("--digits_label_map", type=str, default=str(DEFAULT_DIGITS_LABEL_MAP))
    p.add_argument("--digits_config", type=str, default=str(DEFAULT_DIGITS_CONFIG))

    p.add_argument("--health_seed52_model", type=str, required=True)
    p.add_argument("--health_seed52_label_map", type=str, required=True)
    p.add_argument("--health_seed52_config", type=str, default="")

    p.add_argument("--health_seed62_model", type=str, required=True)
    p.add_argument("--health_seed62_label_map", type=str, required=True)
    p.add_argument("--health_seed62_config", type=str, default="")

    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def _load_state_dict_with_backward_compat(model: torch.nn.Module, state_dict: Dict) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if any(str(k).startswith("net.") for k in state_dict.keys()):
            stripped = {k[4:]: v for k, v in state_dict.items() if str(k).startswith("net.")}
            model.load_state_dict(stripped, strict=False)
        else:
            raise


def _export_one(spec: ModelSpec, out_root: Path) -> Dict:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {spec.checkpoint_path}")
    if not spec.label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {spec.label_map_path}")

    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"checkpoint missing state_dict: {spec.checkpoint_path}")

    label_map = load_json(spec.label_map_path)
    num_classes = int(checkpoint.get("num_classes", len(label_map)))
    cfg = dict(checkpoint.get("config", {}))
    model_variant = str(cfg.get("model_variant", "enhanced_reslstm"))
    input_channels = int(checkpoint.get("input_channels", 2))
    target_points = int(checkpoint.get("target_points", cfg.get("target_points", 300)))

    model = MorseCharModel(
        num_classes=num_classes,
        input_channels=input_channels,
        variant=model_variant,
    )
    _load_state_dict_with_backward_compat(model, checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(1, input_channels, target_points)
        traced = torch.jit.trace(model, dummy)
        logits = traced(dummy)
    if logits.ndim != 2:
        raise ValueError(f"unexpected logits shape for {spec.model_id}: {tuple(logits.shape)}")

    export_dir = ensure_dir(out_root / spec.model_id)
    ptl_path = export_dir / "morse_char_model_android.ptl"
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(str(ptl_path))

    shutil.copy2(spec.label_map_path, export_dir / "label_map.json")
    if spec.train_config_path and spec.train_config_path.exists() and spec.train_config_path.is_file():
        shutil.copy2(spec.train_config_path, export_dir / "train_config.json")

    meta = {
        "model_id": spec.model_id,
        "task": spec.task,
        "checkpoint_path": str(spec.checkpoint_path),
        "android_model_path": str(ptl_path),
        "label_map_path": str(export_dir / "label_map.json"),
        "train_config_path": str(export_dir / "train_config.json")
        if (export_dir / "train_config.json").exists()
        else "",
        "model_variant": model_variant,
        "input_channels": input_channels,
        "target_points": target_points,
        "num_classes": num_classes,
        "output_shape_example": list(logits.shape),
    }
    with (export_dir / "android_export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.out_dir))

    specs: List[ModelSpec] = [
        ModelSpec(
            model_id="letters",
            task="letters",
            checkpoint_path=Path(args.letters_model),
            label_map_path=Path(args.letters_label_map),
            train_config_path=Path(args.letters_config),
        ),
        ModelSpec(
            model_id="digits",
            task="digits",
            checkpoint_path=Path(args.digits_model),
            label_map_path=Path(args.digits_label_map),
            train_config_path=Path(args.digits_config),
        ),
        ModelSpec(
            model_id="health_seed52",
            task="health",
            checkpoint_path=Path(args.health_seed52_model),
            label_map_path=Path(args.health_seed52_label_map),
            train_config_path=Path(args.health_seed52_config) if str(args.health_seed52_config).strip() else Path(""),
        ),
        ModelSpec(
            model_id="health_seed62",
            task="health",
            checkpoint_path=Path(args.health_seed62_model),
            label_map_path=Path(args.health_seed62_label_map),
            train_config_path=Path(args.health_seed62_config) if str(args.health_seed62_config).strip() else Path(""),
        ),
    ]

    rows: List[Dict] = []
    for spec in specs:
        rows.append(_export_one(spec=spec, out_root=out_root))

    manifest = {
        "out_root": str(out_root),
        "num_models": len(rows),
        "models": rows,
    }
    manifest_json = out_root / "android_export_manifest.json"
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(out_root / "android_export_manifest.csv", index=False, encoding="utf-8")

    print(f"Exported {len(rows)} models to: {out_root}")
    print(f"Manifest: {manifest_json}")
    for row in rows:
        print(f"- {row['model_id']}: {row['android_model_path']}")


if __name__ == "__main__":
    main()
