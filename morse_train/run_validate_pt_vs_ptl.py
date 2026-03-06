import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from run_health_deep import build_segment_bundle
from src.dataset import load_cache
from src.model import MorseCharModel
from src.utils import ensure_dir, load_json


DEFAULT_HEALTH_INPUT_DIR = Path(r"d:\huxi\health_data_0302_0303_20260304_205759")
DEFAULT_TOP1_MATCH_THRESHOLD = 0.999
DEFAULT_ACC_DELTA_THRESHOLD = 0.002
DEFAULT_F1_DELTA_THRESHOLD = 0.002


@dataclass
class EvalData:
    X: np.ndarray
    y: np.ndarray
    file_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate pt vs ptl accuracy consistency.")
    p.add_argument("--manifest", type=str, required=True, help="android_export_manifest.json")
    p.add_argument("--health_input_dir", type=str, default=str(DEFAULT_HEALTH_INPUT_DIR))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    p.add_argument("--top1_match_threshold", type=float, default=DEFAULT_TOP1_MATCH_THRESHOLD)
    p.add_argument("--accuracy_delta_threshold", type=float, default=DEFAULT_ACC_DELTA_THRESHOLD)
    p.add_argument("--macro_f1_delta_threshold", type=float, default=DEFAULT_F1_DELTA_THRESHOLD)
    p.add_argument("--out_dir", type=str, default=None)
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


def _load_pt_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(checkpoint.get("config", {}))
    model_variant = str(cfg.get("model_variant", "enhanced_reslstm"))
    num_classes = int(checkpoint.get("num_classes", len(checkpoint.get("label_map", {}))))
    input_channels = int(checkpoint.get("input_channels", 2))
    model = MorseCharModel(
        num_classes=num_classes,
        input_channels=input_channels,
        variant=model_variant,
    )
    _load_state_dict_with_backward_compat(model, checkpoint["state_dict"])
    model.eval()
    return model


def _load_ptl_model(ptl_path: Path):
    try:
        model = torch.jit.load(str(ptl_path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        if hasattr(torch.jit, "mobile") and hasattr(torch.jit.mobile, "_load_for_lite_interpreter"):
            model = torch.jit.mobile._load_for_lite_interpreter(str(ptl_path))
            return model
        if hasattr(torch.jit, "_load_for_lite_interpreter"):
            model = torch.jit._load_for_lite_interpreter(str(ptl_path))
            return model
        raise


def _forward_model(model, x: torch.Tensor) -> torch.Tensor:
    out = None
    try:
        out = model(x)
    except Exception:
        out = model.forward(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(out)}")
    return out


@torch.no_grad()
def _predict_logits(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    logits_list: List[np.ndarray] = []
    for i in range(0, int(X.shape[0]), int(batch_size)):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        logits = _forward_model(model, xb).detach().cpu().numpy().astype(np.float32)
        logits_list.append(logits)
    if not logits_list:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(logits_list, axis=0)


def _compute_row(
    model_id: str,
    task: str,
    y: np.ndarray,
    pred_pt: np.ndarray,
    pred_ptl: np.ndarray,
    top1_match_threshold: float,
    accuracy_delta_threshold: float,
    macro_f1_delta_threshold: float,
) -> Dict[str, object]:
    pt_acc = float(accuracy_score(y, pred_pt))
    ptl_acc = float(accuracy_score(y, pred_ptl))
    pt_f1 = float(f1_score(y, pred_pt, average="macro", zero_division=0))
    ptl_f1 = float(f1_score(y, pred_ptl, average="macro", zero_division=0))
    top1_match = float(np.mean((pred_pt == pred_ptl).astype(np.float32)))
    acc_delta = float(abs(pt_acc - ptl_acc))
    f1_delta = float(abs(pt_f1 - ptl_f1))

    pass_top1 = top1_match >= float(top1_match_threshold)
    pass_acc_delta = acc_delta <= float(accuracy_delta_threshold)
    pass_f1_delta = f1_delta <= float(macro_f1_delta_threshold)

    return {
        "model_id": model_id,
        "task": task,
        "n_samples": int(y.size),
        "top1_match_rate": top1_match,
        "pt_accuracy": pt_acc,
        "ptl_accuracy": ptl_acc,
        "accuracy_delta_abs": acc_delta,
        "pt_macro_f1": pt_f1,
        "ptl_macro_f1": ptl_f1,
        "macro_f1_delta_abs": f1_delta,
        "pass_top1": bool(pass_top1),
        "pass_accuracy_delta": bool(pass_acc_delta),
        "pass_macro_f1_delta": bool(pass_f1_delta),
        "passed": bool(pass_top1 and pass_acc_delta and pass_f1_delta),
    }


def _load_letters_or_digits_eval_data(checkpoint_path: Path) -> EvalData:
    artifacts_dir = checkpoint_path.parent
    cache_path = artifacts_dir / "segments_cache.npz"
    label_map_path = artifacts_dir / "label_map.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"segments cache not found: {cache_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")
    bundle = load_cache(cache_path=cache_path, label_map_path=label_map_path)
    return EvalData(X=bundle.X, y=bundle.y.astype(np.int64), file_ids=np.asarray(bundle.source_files))


def _load_health_eval_data(
    health_input_dir: Path,
    model_label_map_path: Path,
    merge_fine_labels: bool = True,
) -> EvalData:
    label_map = load_json(model_label_map_path)
    bundle = build_segment_bundle(
        input_dir=health_input_dir,
        window_sec=60.0,
        target_points=300,
        min_valid_ratio=0.6,
        clip_low_pct=0.5,
        clip_high_pct=99.5,
        median_window=5,
        smooth_window=9,
        merge_fine_labels=bool(merge_fine_labels),
    )
    file_labels = bundle.file_df["label"].astype(str).to_numpy()
    file_label_ids = np.array([int(label_map[label]) for label in file_labels], dtype=np.int64)
    y_seg = file_label_ids[bundle.segment_file_ids]
    return EvalData(X=bundle.X.astype(np.float32), y=y_seg, file_ids=bundle.segment_file_ids.astype(np.int64))


def _aggregate_file_logits(logits: np.ndarray, file_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    file_ids = np.asarray(file_ids).astype(np.int64)
    uniq = np.unique(file_ids)
    file_logits = []
    for fid in uniq:
        idx = np.where(file_ids == fid)[0]
        file_logits.append(np.mean(logits[idx], axis=0))
    return uniq, np.stack(file_logits).astype(np.float32)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    models = list(manifest.get("models", []))
    if not models:
        raise ValueError("manifest has no models")

    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else ensure_dir(manifest_path.parent / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_root = ensure_dir(out_root)

    thresholds = {
        "top1_match_threshold": float(args.top1_match_threshold),
        "accuracy_delta_threshold": float(args.accuracy_delta_threshold),
        "macro_f1_delta_threshold": float(args.macro_f1_delta_threshold),
    }

    max_samples = int(args.max_samples)
    rows: List[Dict] = []
    per_model_detail: Dict[str, Dict] = {}
    health_store: Dict[str, Dict] = {}

    for m in models:
        model_id = str(m["model_id"])
        task = str(m["task"])
        checkpoint_path = Path(m["checkpoint_path"])
        ptl_path = Path(m["android_model_path"])
        label_map_path = Path(m["label_map_path"])

        if task in ("letters", "digits"):
            data = _load_letters_or_digits_eval_data(checkpoint_path=checkpoint_path)
        elif task == "health":
            data = _load_health_eval_data(
                health_input_dir=Path(args.health_input_dir),
                model_label_map_path=label_map_path,
                merge_fine_labels=True,
            )
        else:
            raise ValueError(f"Unsupported task in manifest: {task}")

        X = data.X
        y = data.y
        file_ids = data.file_ids
        if max_samples > 0 and int(X.shape[0]) > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
            file_ids = file_ids[:max_samples]

        pt_model = _load_pt_model(checkpoint_path=checkpoint_path)
        ptl_model = _load_ptl_model(ptl_path=ptl_path)
        logits_pt = _predict_logits(pt_model, X=X, batch_size=int(args.batch_size))
        logits_ptl = _predict_logits(ptl_model, X=X, batch_size=int(args.batch_size))
        pred_pt = np.argmax(logits_pt, axis=1).astype(np.int64)
        pred_ptl = np.argmax(logits_ptl, axis=1).astype(np.int64)

        row = _compute_row(
            model_id=model_id,
            task=task,
            y=y,
            pred_pt=pred_pt,
            pred_ptl=pred_ptl,
            top1_match_threshold=thresholds["top1_match_threshold"],
            accuracy_delta_threshold=thresholds["accuracy_delta_threshold"],
            macro_f1_delta_threshold=thresholds["macro_f1_delta_threshold"],
        )
        rows.append(row)
        per_model_detail[model_id] = {
            "model_id": model_id,
            "task": task,
            "checkpoint_path": str(checkpoint_path),
            "android_model_path": str(ptl_path),
            "n_samples": int(y.size),
        }

        if model_id in ("health_seed52", "health_seed62"):
            health_store[model_id] = {
                "y_seg": y,
                "file_ids": file_ids,
                "logits_pt": logits_pt,
                "logits_ptl": logits_ptl,
            }

    # Extra ensemble verification for health seed52 + seed62.
    if "health_seed52" in health_store and "health_seed62" in health_store:
        h52 = health_store["health_seed52"]
        h62 = health_store["health_seed62"]
        if not np.array_equal(h52["file_ids"], h62["file_ids"]):
            raise RuntimeError("health ensemble check failed: file_ids mismatch between seed52 and seed62")
        if not np.array_equal(h52["y_seg"], h62["y_seg"]):
            raise RuntimeError("health ensemble check failed: labels mismatch between seed52 and seed62")

        ensemble_seg_logits_pt = 0.5 * (h52["logits_pt"] + h62["logits_pt"])
        ensemble_seg_logits_ptl = 0.5 * (h52["logits_ptl"] + h62["logits_ptl"])
        file_ids = h52["file_ids"]
        y_seg = h52["y_seg"]

        uniq_file_ids, file_logits_pt = _aggregate_file_logits(ensemble_seg_logits_pt, file_ids=file_ids)
        _, file_logits_ptl = _aggregate_file_logits(ensemble_seg_logits_ptl, file_ids=file_ids)

        # File-level truth from first segment label of each file.
        y_file = np.array([int(y_seg[np.where(file_ids == fid)[0][0]]) for fid in uniq_file_ids], dtype=np.int64)
        pred_file_pt = np.argmax(file_logits_pt, axis=1).astype(np.int64)
        pred_file_ptl = np.argmax(file_logits_ptl, axis=1).astype(np.int64)

        rows.append(
            _compute_row(
                model_id="health_ensemble_52_62_file",
                task="health",
                y=y_file,
                pred_pt=pred_file_pt,
                pred_ptl=pred_file_ptl,
                top1_match_threshold=thresholds["top1_match_threshold"],
                accuracy_delta_threshold=thresholds["accuracy_delta_threshold"],
                macro_f1_delta_threshold=thresholds["macro_f1_delta_threshold"],
            )
        )

    report_df = pd.DataFrame(rows)
    report_csv = out_root / "pt_vs_ptl_report.csv"
    report_json = out_root / "pt_vs_ptl_report.json"
    report_df.to_csv(report_csv, index=False, encoding="utf-8")

    summary = {
        "manifest_path": str(manifest_path),
        "out_dir": str(out_root),
        "thresholds": thresholds,
        "all_passed": bool(report_df["passed"].all()) if not report_df.empty else False,
        "rows": rows,
        "models": per_model_detail,
    }
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {report_csv}")
    print(f"Saved JSON: {report_json}")
    print(f"all_passed={summary['all_passed']}")
    for _, r in report_df.iterrows():
        print(
            f"- {r['model_id']}: pass={bool(r['passed'])} "
            f"top1_match={float(r['top1_match_rate']):.6f} "
            f"acc_delta={float(r['accuracy_delta_abs']):.6f} "
            f"f1_delta={float(r['macro_f1_delta_abs']):.6f}"
        )


if __name__ == "__main__":
    main()
