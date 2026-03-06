import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from run_health_deep import merge_fine_label
from src.health_analysis import parse_health_label
from src.io_zip import _parse_dual_header, _read_csv_blob, load_signal_from_zip
from src.preprocess import canonicalize_signal, segment_signal
from src.utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate migrated PTL models on num/yundong/zimu folders.")
    p.add_argument("--root", type=str, default=r"d:\huxi")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def _load_frame_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        signal = load_signal_from_zip(path)
        return canonicalize_signal(signal.frame)
    if suffix == ".csv":
        blob = path.read_bytes()
        raw = _read_csv_blob(blob)
        return canonicalize_signal(_parse_dual_header(raw))
    if suffix == ".xlsx":
        raw = pd.read_excel(path, header=None)
        return canonicalize_signal(_parse_dual_header(raw))
    raise ValueError(f"Unsupported file suffix: {path}")


def _load_ptl(path: Path):
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        if hasattr(torch.jit, "mobile") and hasattr(torch.jit.mobile, "_load_for_lite_interpreter"):
            return torch.jit.mobile._load_for_lite_interpreter(str(path))
        if hasattr(torch.jit, "_load_for_lite_interpreter"):
            return torch.jit._load_for_lite_interpreter(str(path))
        raise


def _forward(model, x: torch.Tensor) -> torch.Tensor:
    try:
        out = model(x)
    except Exception:
        out = model.forward(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def _predict_probs(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, int(X.shape[0]), int(batch_size)):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        logits = _forward(model, xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        out.append(probs)
    if not out:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _choose_best_offset_candidate(
    frame: pd.DataFrame,
    model,
    id_to_label: Dict[int, str],
    channel_count: int,
    window_sec: float,
    batch_size: int,
) -> Tuple[str, float, int]:
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    if time_s.size == 0:
        return "", 0.0, 0
    duration = float(np.nanmax(time_s) - np.nanmin(time_s))
    nominal_segments = int(duration // float(window_sec))
    offsets = np.arange(0.0, float(window_sec), 1.0, dtype=np.float32)

    candidates = []
    for off in offsets:
        segs = segment_signal(
            frame,
            window_sec=float(window_sec),
            target_points=300,
            min_valid_ratio=0.6,
            offset_sec=float(off),
            clip_low_pct=0.5,
            clip_high_pct=99.5,
            median_window=5,
            smooth_window=9,
            detrend=True,
            adaptive_boundaries=True,
            boundary_search_radius=2.0,
        )
        if not segs:
            continue
        X = np.stack([seg.x[:channel_count, :] for seg in segs]).astype(np.float32)
        probs = _predict_probs(model, X=X, batch_size=batch_size)
        pred_ids = np.argmax(probs, axis=1).astype(np.int64)
        conf = float(np.mean(np.max(probs, axis=1)))
        seq = "".join([id_to_label[int(i)] for i in pred_ids])
        candidates.append((float(off), seq, conf, int(len(segs))))

    if not candidates:
        return "", 0.0, 0
    same_len = [c for c in candidates if c[3] == nominal_segments] if nominal_segments > 0 else []
    pool = same_len if same_len else candidates
    best = max(pool, key=lambda x: x[2])
    return str(best[1]), float(best[2]), int(best[3])


def _extract_token(name: str) -> str:
    m = re.search(r"AB[,\uFF0C]([^,\uFF0C]+)", name)
    if not m:
        return ""
    return str(m.group(1)).strip()


def _expand_digit_token(token: str) -> str:
    t = str(token).strip()
    m = re.fullmatch(r"(\d)\s*-\s*(\d)", t)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if a <= b:
            return "".join(str(x) for x in range(a, b + 1))
        return "".join(str(x) for x in range(a, b - 1, -1))
    digits = "".join(ch for ch in t if ch.isdigit())
    return digits


def _char_accuracy(y_true: str, y_pred: str) -> float:
    t = str(y_true)
    p = str(y_pred)
    den = max(len(t), len(p), 1)
    matches = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
    return float(matches / den)


def _collect_files(folder: Path) -> List[Path]:
    all_files = []
    for ext in ("*.zip", "*.csv", "*.xlsx"):
        all_files.extend(folder.rglob(ext))
    return sorted([p for p in all_files if p.is_file()])


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    num_dir = root / "num"
    zimu_dir = root / "zimu"
    yudong_dir = root / "yudong"
    if not yudong_dir.exists():
        yudong_dir = root / "yundong"

    model_root = root / "morse_train" / "model_output"
    assets_root = root / "morse_train" / "android_delivery_task_models_20260305"

    letters_model = _load_ptl(model_root / "letters_recognition_best.ptl")
    digits_model = _load_ptl(model_root / "digits_recognition_best.ptl")
    health_model = _load_ptl(model_root / "health_status_recognition_best.ptl")

    letters_map = load_json(assets_root / "letters" / "label_map.json")
    digits_map = load_json(assets_root / "digits" / "label_map.json")
    health_map = load_json(assets_root / "health_seed52" / "label_map.json")
    id_to_letters = {int(v): k for k, v in letters_map.items()}
    id_to_digits = {int(v): k for k, v in digits_map.items()}
    id_to_health = {int(v): k for k, v in health_map.items()}
    health_label_to_id = {str(k): int(v) for k, v in health_map.items()}

    out_dir = Path(args.out_dir) if args.out_dir else ensure_dir(
        root / "morse_train" / "model_output" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = ensure_dir(out_dir)

    # Digits sequence task
    digit_rows = []
    for fp in _collect_files(num_dir):
        frame = _load_frame_any(fp)
        pred_seq, conf, n_seg = _choose_best_offset_candidate(
            frame=frame,
            model=digits_model,
            id_to_label=id_to_digits,
            channel_count=1,
            window_sec=30.0,
            batch_size=int(args.batch_size),
        )
        expected = _expand_digit_token(_extract_token(fp.name))
        digit_rows.append(
            {
                "file": str(fp),
                "expected_sequence": expected,
                "pred_sequence": pred_seq,
                "exact_match": int(expected == pred_seq),
                "char_accuracy": _char_accuracy(expected, pred_seq),
                "mean_confidence": conf,
                "segments": n_seg,
            }
        )
    digit_df = pd.DataFrame(digit_rows)
    digit_df.to_csv(out_dir / "digits_eval.csv", index=False, encoding="utf-8")

    # Letters sequence task
    letter_rows = []
    for fp in _collect_files(zimu_dir):
        frame = _load_frame_any(fp)
        pred_seq, conf, n_seg = _choose_best_offset_candidate(
            frame=frame,
            model=letters_model,
            id_to_label=id_to_letters,
            channel_count=2,
            window_sec=30.0,
            batch_size=int(args.batch_size),
        )
        expected = "".join(ch for ch in _extract_token(fp.name).upper() if "A" <= ch <= "Z")
        letter_rows.append(
            {
                "file": str(fp),
                "expected_sequence": expected,
                "pred_sequence": pred_seq,
                "exact_match": int(expected == pred_seq),
                "char_accuracy": _char_accuracy(expected, pred_seq),
                "mean_confidence": conf,
                "segments": n_seg,
            }
        )
    letter_df = pd.DataFrame(letter_rows)
    letter_df.to_csv(out_dir / "letters_eval.csv", index=False, encoding="utf-8")

    # Health classification task (file-level)
    health_rows = []
    for fp in _collect_files(yudong_dir):
        frame = _load_frame_any(fp)
        segs = segment_signal(
            frame,
            window_sec=60.0,
            target_points=300,
            min_valid_ratio=0.6,
            offset_sec=0.0,
            clip_low_pct=0.5,
            clip_high_pct=99.5,
            median_window=5,
            smooth_window=9,
            detrend=True,
            adaptive_boundaries=False,
        )
        if not segs:
            health_rows.append(
                {
                    "file": str(fp),
                    "expected_label": "",
                    "pred_label": "",
                    "is_correct": 0,
                    "mean_confidence": 0.0,
                    "segments": 0,
                }
            )
            continue
        X = np.stack([seg.x[:2, :] for seg in segs]).astype(np.float32)
        probs = _predict_probs(health_model, X=X, batch_size=int(args.batch_size))
        logits_mean = np.mean(probs, axis=0)
        pred_id = int(np.argmax(logits_mean))
        pred_label = id_to_health[pred_id]
        expected_raw = parse_health_label(fp.name)
        expected_merged = merge_fine_label(expected_raw, merge_enabled=True)
        expected_id = health_label_to_id.get(expected_merged, -1)
        health_rows.append(
            {
                "file": str(fp),
                "expected_label": expected_merged,
                "pred_label": pred_label,
                "is_correct": int(expected_id == pred_id),
                "mean_confidence": float(np.max(logits_mean)),
                "segments": int(X.shape[0]),
            }
        )
    health_df = pd.DataFrame(health_rows)
    health_df.to_csv(out_dir / "health_eval.csv", index=False, encoding="utf-8")

    # Summary
    digits_exact = float(digit_df["exact_match"].mean()) if not digit_df.empty else float("nan")
    digits_char = float(digit_df["char_accuracy"].mean()) if not digit_df.empty else float("nan")
    letters_exact = float(letter_df["exact_match"].mean()) if not letter_df.empty else float("nan")
    letters_char = float(letter_df["char_accuracy"].mean()) if not letter_df.empty else float("nan")
    health_acc = float(health_df["is_correct"].mean()) if not health_df.empty else float("nan")

    summary = {
        "num_dir": str(num_dir),
        "zimu_dir": str(zimu_dir),
        "yudong_dir_used": str(yudong_dir),
        "digits_file_count": int(digit_df.shape[0]),
        "digits_exact_match_rate": digits_exact,
        "digits_char_accuracy_mean": digits_char,
        "letters_file_count": int(letter_df.shape[0]),
        "letters_exact_match_rate": letters_exact,
        "letters_char_accuracy_mean": letters_char,
        "health_file_count": int(health_df.shape[0]),
        "health_file_accuracy": health_acc,
        "out_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("digits_exact_match_rate:", digits_exact)
    print("digits_char_accuracy_mean:", digits_char)
    print("letters_exact_match_rate:", letters_exact)
    print("letters_char_accuracy_mean:", letters_char)
    print("health_file_accuracy:", health_acc)
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
