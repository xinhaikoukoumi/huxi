import argparse
import copy
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.health_analysis import (
    map_coarse_label,
    parse_group_day_from_filename,
    parse_health_label,
    run_grouped_cv,
)
from src.io_zip import load_signal_from_zip
from src.model import MorseCharModel
from src.preprocess import canonicalize_signal, segment_signal
from src.utils import ensure_dir, log, save_json, set_seed


DEFAULT_INPUT_DIR = r"d:\huxi\健康数据"
DEFAULT_OUT_BASE = r"d:\huxi\morse_train"
MERGED_FINE_LABEL_MAP = {
    "左鼻塞": "鼻塞",
    "右鼻塞": "鼻塞",
    "小跑": "锻炼",
    "跑步": "锻炼",
}


def merge_fine_label(label: str, merge_enabled: bool = True) -> str:
    if not merge_enabled:
        return str(label)
    key = str(label).strip()
    return MERGED_FINE_LABEL_MAP.get(key, key)


def parse_tta_shifts(shifts_text: str) -> List[int]:
    raw = str(shifts_text or "0").strip()
    if not raw:
        return [0]
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        out = [0]
    if 0 not in out:
        out = [0] + out
    return sorted(set(out))


def parse_seed_list(seed_text: str) -> List[int]:
    raw = str(seed_text or "").strip()
    if not raw:
        return [42]
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        out = [42]
    return sorted(set(out))


class HealthSegDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        file_ids: np.ndarray,
        indices: np.ndarray,
        augment: bool = False,
        shift_max: int = 8,
        noise_std: float = 0.01,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        time_mask_prob: float = 0.2,
        time_mask_max_width: int = 24,
    ):
        self.X = X
        self.y = y
        self.file_ids = file_ids
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)
        self.shift_max = int(max(0, shift_max))
        self.noise_std = float(max(0.0, noise_std))
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if self.scale_max < self.scale_min:
            self.scale_min, self.scale_max = self.scale_max, self.scale_min
        self.time_mask_prob = float(min(max(time_mask_prob, 0.0), 1.0))
        self.time_mask_max_width = int(max(4, time_mask_max_width))

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @staticmethod
    def _time_mask(x: np.ndarray, max_width: int) -> np.ndarray:
        t = int(x.shape[1])
        if t <= 8:
            return x
        width = int(np.random.randint(4, min(max_width, t - 1) + 1))
        start = int(np.random.randint(0, t - width))
        x[:, start : start + width] = 0.0
        return x

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X[idx].copy()
        if self.augment:
            if self.shift_max > 0:
                shift = int(np.random.randint(-self.shift_max, self.shift_max + 1))
                if shift != 0:
                    x = np.roll(x, shift=shift, axis=1)
            scales = np.random.uniform(self.scale_min, self.scale_max, size=(x.shape[0], 1)).astype(np.float32)
            x = x * scales
            if np.random.rand() < self.time_mask_prob:
                x = self._time_mask(x, max_width=self.time_mask_max_width)
            if self.noise_std > 0:
                noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
                x = x + noise
        return (
            torch.from_numpy(x).float(),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
            torch.tensor(int(self.file_ids[idx]), dtype=torch.long),
        )


@dataclass
class SegmentBundle:
    X: np.ndarray
    segment_file_ids: np.ndarray
    file_df: pd.DataFrame
    skipped_df: pd.DataFrame


def _parse_args():
    p = argparse.ArgumentParser(description="Deep-learning health classification with GPU.")
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", type=str, default=None, help="Default: health_deep_<timestamp>")
    p.add_argument("--window_sec", type=float, default=60.0, help="Primary segment window.")
    p.add_argument("--target_points", type=int, default=300)
    p.add_argument("--min_valid_ratio", type=float, default=0.6)
    p.add_argument("--feature_clip_low_pct", type=float, default=0.5)
    p.add_argument("--feature_clip_high_pct", type=float, default=99.5)
    p.add_argument("--median_window", type=int, default=5)
    p.add_argument("--smooth_window", type=int, default=9)
    p.add_argument("--target", type=str, default="both", choices=["coarse", "fine", "both"])
    p.add_argument("--max_epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--model_variant", type=str, default="enhanced_reslstm", choices=["enhanced_reslstm", "baseline_cnn_bilstm"])
    p.add_argument("--tta_shifts", type=str, default="0,-4,4", help="Comma-separated time-axis shifts for TTA.")
    p.add_argument(
        "--cv_mode",
        type=str,
        default="group_by_day",
        choices=["group_by_day", "stratified"],
        help="File-level CV split mode.",
    )
    p.add_argument("--merge_fine_labels", dest="merge_fine_labels", action="store_true", default=True)
    p.add_argument("--no_merge_fine_labels", dest="merge_fine_labels", action="store_false")
    p.add_argument("--use_balanced_sampler", dest="use_balanced_sampler", action="store_true", default=True)
    p.add_argument("--no_balanced_sampler", dest="use_balanced_sampler", action="store_false")
    p.add_argument("--pipeline", type=str, default="single", choices=["single", "two_stage_ensemble"])
    p.add_argument("--ensemble_seeds", type=str, default="42,52,62", help="Comma-separated seeds for ensemble.")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline_scan_dir", type=str, default=None)
    return p.parse_args()


def _resolve_output_dir(output_dir: str = None) -> Path:
    if output_dir:
        return ensure_dir(Path(output_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(DEFAULT_OUT_BASE) / f"health_deep_{ts}")


def _extract_segments_for_file(
    zip_path: Path,
    window_sec: float,
    target_points: int,
    min_valid_ratio: float,
    clip_low_pct: float,
    clip_high_pct: float,
    median_window: int,
    smooth_window: int,
) -> List[np.ndarray]:
    signal = load_signal_from_zip(zip_path)
    frame = canonicalize_signal(signal.frame)
    segs = segment_signal(
        frame,
        window_sec=float(window_sec),
        target_points=int(target_points),
        min_valid_ratio=float(min_valid_ratio),
        offset_sec=0.0,
        clip_low_pct=float(clip_low_pct),
        clip_high_pct=float(clip_high_pct),
        median_window=int(median_window),
        smooth_window=int(smooth_window),
        detrend=True,
        adaptive_boundaries=False,
    )
    if len(segs) == 0:
        duration_s = float(frame["time_s"].max()) if int(frame.shape[0]) > 0 else 0.0
        fallback_window = max(5.0, duration_s)
        segs = segment_signal(
            frame,
            window_sec=float(fallback_window),
            target_points=int(target_points),
            min_valid_ratio=float(min_valid_ratio),
            offset_sec=0.0,
            clip_low_pct=float(clip_low_pct),
            clip_high_pct=float(clip_high_pct),
            median_window=int(median_window),
            smooth_window=int(smooth_window),
            detrend=True,
            adaptive_boundaries=False,
        )
    return [seg.x.astype(np.float32) for seg in segs]


def build_segment_bundle(
    input_dir: Path,
    window_sec: float,
    target_points: int,
    min_valid_ratio: float,
    clip_low_pct: float,
    clip_high_pct: float,
    median_window: int,
    smooth_window: int,
    merge_fine_labels: bool = True,
) -> SegmentBundle:
    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise ValueError(f"No zip files found in {input_dir}")

    X_list: List[np.ndarray] = []
    seg_file_ids: List[int] = []
    file_rows: List[dict] = []
    skipped_rows: List[dict] = []

    for zp in zips:
        try:
            fine_label_raw = parse_health_label(zp.name)
            fine_label_merged = merge_fine_label(fine_label_raw, merge_enabled=bool(merge_fine_labels))
            coarse_label = map_coarse_label(fine_label_raw)
        except Exception as exc:
            skipped_rows.append({"zip_file": zp.name, "reason": f"label_error: {exc}"})
            continue

        try:
            segs = _extract_segments_for_file(
                zp,
                window_sec=window_sec,
                target_points=target_points,
                min_valid_ratio=min_valid_ratio,
                clip_low_pct=clip_low_pct,
                clip_high_pct=clip_high_pct,
                median_window=median_window,
                smooth_window=smooth_window,
            )
        except Exception as exc:
            skipped_rows.append({"zip_file": zp.name, "reason": f"segment_error: {exc}"})
            continue

        if not segs:
            skipped_rows.append({"zip_file": zp.name, "reason": "no_segments"})
            continue

        file_id = len(file_rows)
        for x in segs:
            X_list.append(x)
            seg_file_ids.append(file_id)
        file_rows.append(
            {
                "file_id": file_id,
                "zip_file": zp.name,
                "label_raw": fine_label_raw,
                "label_merged": fine_label_merged,
                "label": fine_label_merged,
                "coarse_label": coarse_label,
                "group_day": parse_group_day_from_filename(zp.name),
                "segment_count": int(len(segs)),
            }
        )

    if not file_rows or not X_list:
        raise RuntimeError("No valid files/segments for deep training.")

    X = np.stack(X_list).astype(np.float32)
    segment_file_ids = np.asarray(seg_file_ids, dtype=np.int64)
    file_df = pd.DataFrame(file_rows).sort_values("file_id").reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped_rows)
    return SegmentBundle(X=X, segment_file_ids=segment_file_ids, file_df=file_df, skipped_df=skipped_df)


def _build_file_cv_splits(file_df: pd.DataFrame, target_col: str, seed: int, cv_mode: str = "group_by_day"):
    cv_df = pd.DataFrame(
        {
            "zip_file": file_df["zip_file"].astype(str),
            target_col: file_df[target_col].astype(str),
            "group_day": file_df["group_day"].astype(str),
            "dummy_feature": np.arange(file_df.shape[0], dtype=np.float32),
        }
    )
    mode = str(cv_mode).strip().lower()
    if mode not in {"group_by_day", "stratified"}:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")
    cv = run_grouped_cv(
        cv_df,
        label_col=target_col,
        feature_cols=["dummy_feature"],
        cv_mode=("group_by_day" if mode == "group_by_day" else "stratified"),
        group_col="group_day",
        seed=int(seed),
    )
    if cv.get("status") != "ok":
        raise RuntimeError(f"Failed to build CV splits: {cv.get('reason', 'unknown')}")
    return cv


def _split_train_val_files(train_file_idx: np.ndarray, file_label_ids: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_file_idx = np.asarray(train_file_idx, dtype=np.int64)
    if train_file_idx.size <= 2:
        if train_file_idx.size <= 1:
            return train_file_idx, train_file_idx
        return train_file_idx[:1], train_file_idx[1:]

    y = file_label_ids[train_file_idx]
    min_count = int(np.bincount(y).min()) if y.size else 0
    stratify = y if min_count >= 2 and int(np.unique(y).size) >= 2 else None
    test_size = max(1, int(round(float(val_ratio) * float(train_file_idx.size))))
    test_size = min(test_size, int(train_file_idx.size - 1))
    try:
        tr, va = train_test_split(
            train_file_idx,
            test_size=test_size,
            random_state=int(seed),
            stratify=stratify,
        )
    except Exception:
        perm = np.random.RandomState(int(seed)).permutation(train_file_idx)
        va = perm[:test_size]
        tr = perm[test_size:]
    if tr.size == 0:
        tr, va = va, tr
    if va.size == 0:
        va = tr[-1:]
        tr = tr[:-1] if tr.size > 1 else tr
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    seg_file_ids: np.ndarray,
    train_seg_idx: np.ndarray,
    val_seg_idx: np.ndarray,
    test_seg_idx: np.ndarray,
    batch_size: int,
    use_balanced_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HealthSegDataset(X, y, seg_file_ids, train_seg_idx, augment=True)
    val_ds = HealthSegDataset(X, y, seg_file_ids, val_seg_idx, augment=False)
    test_ds = HealthSegDataset(X, y, seg_file_ids, test_seg_idx, augment=False)
    if bool(use_balanced_sampler):
        train_y = y[train_seg_idx]
        counts = np.bincount(train_y, minlength=max(1, int(np.max(train_y)) + 1)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        sample_weights = 1.0 / counts[train_y]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=int(train_seg_idx.size),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size) * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size) * 2, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def _mixup_batch(xb: torch.Tensor, yb: torch.Tensor, alpha: float):
    if float(alpha) <= 0 or int(xb.shape[0]) < 2:
        return xb, yb, yb, 1.0
    lam = float(np.random.beta(float(alpha), float(alpha)))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(int(xb.shape[0]), device=xb.device)
    mixed = lam * xb + (1.0 - lam) * xb[perm]
    return mixed, yb, yb[perm], lam


def _autocast_ctx(device: torch.device, enabled: bool = True):
    on = bool(enabled) and str(device.type) == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=str(device.type), enabled=on)
    return torch.cuda.amp.autocast(enabled=on)


def _run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if float(mixup_alpha) > 0 and float(mixup_prob) > 0 and np.random.rand() < float(mixup_prob):
            xb, y_a, y_b, lam = _mixup_batch(xb, yb, alpha=float(mixup_alpha))
        else:
            y_a, y_b, lam = yb, yb, 1.0
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, enabled=(scaler is not None)):
            logits = model(xb)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = int(xb.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
    return loss_sum / max(1, n)


@torch.no_grad()
def _eval_loss(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with _autocast_ctx(device, enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        bs = int(xb.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
    return loss_sum / max(1, n)


@torch.no_grad()
def _predict_logits(model, loader, device, tta_shifts: List[int] = None):
    model.eval()
    shifts = list(tta_shifts or [0])
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_file_ids: List[np.ndarray] = []
    for xb, yb, fb in loader:
        xb = xb.to(device, non_blocking=True)
        logits_acc = None
        for shift in shifts:
            x_in = xb if int(shift) == 0 else torch.roll(xb, shifts=int(shift), dims=2)
            with _autocast_ctx(device, enabled=(device.type == "cuda")):
                logits = model(x_in)
            logits_acc = logits if logits_acc is None else (logits_acc + logits)
        logits = (logits_acc / float(len(shifts))).detach().cpu().numpy().astype(np.float32)
        all_logits.append(logits)
        all_y.append(yb.numpy().astype(np.int64))
        all_file_ids.append(fb.numpy().astype(np.int64))
    if not all_logits:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_file_ids, axis=0),
    )


def _segment_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    if y_true.size == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "macro_recall": float("nan")}
    rec = recall_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for i, r in enumerate(rec.tolist()):
        out[f"recall_class_{i}"] = float(r)
    return out


def _file_level_logits_df(
    logits: np.ndarray,
    seg_file_ids: np.ndarray,
    file_label_ids: np.ndarray,
    eval_file_ids: np.ndarray,
    n_classes: int,
    class_names: List[str] = None,
    file_names: np.ndarray = None,
) -> pd.DataFrame:
    eval_set = set(np.asarray(eval_file_ids, dtype=np.int64).tolist())
    bucket: Dict[int, List[np.ndarray]] = {}
    for i in range(int(logits.shape[0])):
        fid = int(seg_file_ids[i])
        if fid not in eval_set:
            continue
        bucket.setdefault(fid, []).append(logits[i])

    rows: List[dict] = []
    for fid in sorted(bucket.keys()):
        mean_logits = np.mean(np.stack(bucket[fid], axis=0), axis=0)
        pred = int(np.argmax(mean_logits))
        true = int(file_label_ids[fid])
        row = {
            "file_id": int(fid),
            "true_id": int(true),
            "pred_id": int(pred),
        }
        if class_names is not None and int(true) < len(class_names):
            row["true_label"] = str(class_names[int(true)])
        if class_names is not None and int(pred) < len(class_names):
            row["pred_label"] = str(class_names[int(pred)])
        if file_names is not None and int(fid) < int(len(file_names)):
            row["zip_file"] = str(file_names[int(fid)])
        for c in range(int(n_classes)):
            row[f"logit_{c}"] = float(mean_logits[c])
        rows.append(row)
    return pd.DataFrame(rows)


def _file_metrics(
    logits: np.ndarray,
    seg_file_ids: np.ndarray,
    file_label_ids: np.ndarray,
    eval_file_ids: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    file_df = _file_level_logits_df(
        logits=logits,
        seg_file_ids=seg_file_ids,
        file_label_ids=file_label_ids,
        eval_file_ids=eval_file_ids,
        n_classes=n_classes,
    )
    if file_df.empty:
        return {"n_files_eval": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "macro_recall": float("nan")}

    y_true_arr = file_df["true_id"].to_numpy(dtype=np.int64)
    y_pred_arr = file_df["pred_id"].to_numpy(dtype=np.int64)
    out = _segment_metrics(y_true_arr, y_pred_arr, n_classes=n_classes)
    out["n_files_eval"] = int(y_true_arr.size)
    return out


def _train_cv_for_target(
    bundle: SegmentBundle,
    target_col: str,
    out_dir: Path,
    seed: int,
    cv_mode: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    mixup_alpha: float,
    mixup_prob: float,
    model_variant: str,
    val_ratio: float,
    tta_shifts: List[int],
    use_balanced_sampler: bool,
    device: torch.device,
    checkpoint_context: Dict[str, object] = None,
) -> Dict[str, object]:
    file_df = bundle.file_df.copy()
    class_names = sorted(file_df[target_col].astype(str).unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    file_df["target_id"] = file_df[target_col].map(class_to_id).astype(int)
    file_label_ids = file_df["target_id"].to_numpy(dtype=np.int64)
    file_names = file_df["zip_file"].astype(str).to_numpy()
    seg_targets = file_label_ids[bundle.segment_file_ids]
    n_classes = int(len(class_names))

    cv = _build_file_cv_splits(file_df, target_col=target_col, seed=int(seed), cv_mode=str(cv_mode))
    fold_rows: List[dict] = []
    file_oof_rows: List[pd.DataFrame] = []
    epoch_rows: List[dict] = []
    checkpoint_context = dict(checkpoint_context or {})
    fold_ckpt_dir = ensure_dir(out_dir / "deploy_checkpoints" / str(target_col))

    for fold_idx, fold in enumerate(cv["splits"]):
        test_file_idx = np.asarray(fold["test_idx"], dtype=np.int64)
        train_pool_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        train_file_idx, val_file_idx = _split_train_val_files(
            train_pool_idx,
            file_label_ids=file_label_ids,
            val_ratio=float(val_ratio),
            seed=int(seed + fold_idx),
        )

        train_mask = np.isin(bundle.segment_file_ids, train_file_idx)
        val_mask = np.isin(bundle.segment_file_ids, val_file_idx)
        test_mask = np.isin(bundle.segment_file_ids, test_file_idx)
        train_seg_idx = np.where(train_mask)[0]
        val_seg_idx = np.where(val_mask)[0]
        test_seg_idx = np.where(test_mask)[0]
        if train_seg_idx.size == 0 or val_seg_idx.size == 0 or test_seg_idx.size == 0:
            log(f"Skip fold {fold_idx} ({target_col}): empty split")
            continue

        train_loader, val_loader, test_loader = _make_loaders(
            X=bundle.X,
            y=seg_targets,
            seg_file_ids=bundle.segment_file_ids,
            train_seg_idx=train_seg_idx,
            val_seg_idx=val_seg_idx,
            test_seg_idx=test_seg_idx,
            batch_size=int(batch_size),
            use_balanced_sampler=bool(use_balanced_sampler),
        )
        train_eval_loader = DataLoader(
            HealthSegDataset(bundle.X, seg_targets, bundle.segment_file_ids, train_seg_idx, augment=False),
            batch_size=int(batch_size) * 2,
            shuffle=False,
            num_workers=0,
        )

        model = MorseCharModel(num_classes=n_classes, input_channels=2, variant=model_variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(max_epochs)),
            eta_min=float(learning_rate) * 0.05,
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))
            except TypeError:
                scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        counts = np.bincount(seg_targets[train_seg_idx], minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        class_weights = counts.sum() / (counts * float(n_classes))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=float(max(0.0, label_smoothing)),
        )

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        wait = 0

        for epoch in range(1, int(max_epochs) + 1):
            train_loss = _run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                mixup_alpha=float(mixup_alpha),
                mixup_prob=float(mixup_prob),
            )
            scheduler.step()
            train_eval_loss = _eval_loss(model, train_eval_loader, criterion, device)
            val_loss = _eval_loss(model, val_loader, criterion, device)
            val_logits, val_y, val_seg_file = _predict_logits(model, val_loader, device, tta_shifts=tta_shifts)
            val_seg_pred = np.argmax(val_logits, axis=1).astype(np.int64)
            val_seg_m = _segment_metrics(val_y, val_seg_pred, n_classes=n_classes)
            val_file = _file_metrics(
                logits=val_logits,
                seg_file_ids=val_seg_file,
                file_label_ids=file_label_ids,
                eval_file_ids=val_file_idx,
                n_classes=n_classes,
            )
            train_logits, train_y_eval, train_seg_file_eval = _predict_logits(
                model,
                train_eval_loader,
                device,
                tta_shifts=tta_shifts,
            )
            train_seg_pred_eval = np.argmax(train_logits, axis=1).astype(np.int64)
            train_seg_m = _segment_metrics(train_y_eval, train_seg_pred_eval, n_classes=n_classes)
            train_file = _file_metrics(
                logits=train_logits,
                seg_file_ids=train_seg_file_eval,
                file_label_ids=file_label_ids,
                eval_file_ids=train_file_idx,
                n_classes=n_classes,
            )
            val_f1 = float(val_file["macro_f1"]) if np.isfinite(val_file["macro_f1"]) else -1.0
            log(
                f"[{target_col}] fold={fold_idx} epoch={epoch} "
                f"train_loss={train_loss:.4f} val_file_f1={val_f1:.4f} "
                f"train_file_acc={float(train_file.get('accuracy', np.nan)):.4f} "
                f"val_file_acc={float(val_file.get('accuracy', np.nan)):.4f} "
                f"val_loss={float(val_loss):.4f}"
            )
            epoch_rows.append(
                {
                    "target_col": str(target_col),
                    "fold_idx": int(fold_idx),
                    "fold_name": str(fold["fold_name"]),
                    "epoch": int(epoch),
                    "train_loss_aug": float(train_loss),
                    "train_loss_eval": float(train_eval_loss),
                    "val_loss": float(val_loss),
                    "train_file_accuracy": float(train_file.get("accuracy", np.nan)),
                    "train_file_macro_f1": float(train_file.get("macro_f1", np.nan)),
                    "train_segment_accuracy": float(train_seg_m.get("accuracy", np.nan)),
                    "train_segment_macro_f1": float(train_seg_m.get("macro_f1", np.nan)),
                    "val_file_accuracy": float(val_file.get("accuracy", np.nan)),
                    "val_file_macro_f1": float(val_file.get("macro_f1", np.nan)),
                    "val_segment_accuracy": float(val_seg_m.get("accuracy", np.nan)),
                    "val_segment_macro_f1": float(val_seg_m.get("macro_f1", np.nan)),
                }
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break

        best_ckpt_path = ""
        best_label_map_path = ""
        if best_state is not None:
            model.load_state_dict(best_state)
            best_ckpt_path = str(fold_ckpt_dir / f"seed{int(seed)}_{target_col}_fold{int(fold_idx)}_best.pt")
            best_label_map_path = str(
                fold_ckpt_dir / f"seed{int(seed)}_{target_col}_fold{int(fold_idx)}_label_map.json"
            )
            label_map = {str(name): int(i) for i, name in enumerate(class_names)}
            checkpoint_payload = {
                "state_dict": best_state,
                "num_classes": int(n_classes),
                "input_channels": int(bundle.X.shape[1]),
                "target_points": int(bundle.X.shape[2]),
                "label_map": label_map,
                "config": {
                    "task": "health",
                    "target_col": str(target_col),
                    "model_variant": str(model_variant),
                    "cv_mode": str(cv_mode),
                    "seed": int(seed),
                    "fold_idx": int(fold_idx),
                    "window_sec": float(checkpoint_context.get("window_sec", np.nan)),
                    "min_valid_ratio": float(checkpoint_context.get("min_valid_ratio", np.nan)),
                    "clip_low_pct": float(checkpoint_context.get("clip_low_pct", np.nan)),
                    "clip_high_pct": float(checkpoint_context.get("clip_high_pct", np.nan)),
                    "median_window": int(checkpoint_context.get("median_window", 0)),
                    "smooth_window": int(checkpoint_context.get("smooth_window", 0)),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "label_smoothing": float(label_smoothing),
                    "mixup_alpha": float(mixup_alpha),
                    "mixup_prob": float(mixup_prob),
                    "tta_shifts": [int(x) for x in tta_shifts],
                    "merge_fine_labels": bool(checkpoint_context.get("merge_fine_labels", True)),
                },
            }
            torch.save(checkpoint_payload, best_ckpt_path)
            save_json(best_label_map_path, label_map)

        test_logits, test_y_seg, test_seg_file = _predict_logits(model, test_loader, device, tta_shifts=tta_shifts)
        test_seg_pred = np.argmax(test_logits, axis=1).astype(np.int64)
        seg_m = _segment_metrics(test_y_seg, test_seg_pred, n_classes=n_classes)
        file_oof_df = _file_level_logits_df(
            logits=test_logits,
            seg_file_ids=test_seg_file,
            file_label_ids=file_label_ids,
            eval_file_ids=test_file_idx,
            n_classes=n_classes,
            class_names=class_names,
            file_names=file_names,
        )
        if not file_oof_df.empty:
            file_oof_df["target_col"] = str(target_col)
            file_oof_df["fold_idx"] = int(fold_idx)
            file_oof_df["fold_name"] = str(fold["fold_name"])
            file_oof_rows.append(file_oof_df)
        file_m = _file_metrics(
            logits=test_logits,
            seg_file_ids=test_seg_file,
            file_label_ids=file_label_ids,
            eval_file_ids=test_file_idx,
            n_classes=n_classes,
        )

        row = {
            "target_col": target_col,
            "fold_idx": int(fold_idx),
            "fold_name": str(fold["fold_name"]),
            "best_epoch": int(best_epoch),
            "best_checkpoint_path": str(best_ckpt_path),
            "best_label_map_path": str(best_label_map_path),
            "n_train_files": int(train_file_idx.size),
            "n_val_files": int(val_file_idx.size),
            "n_test_files": int(test_file_idx.size),
            "n_train_segments": int(train_seg_idx.size),
            "n_val_segments": int(val_seg_idx.size),
            "n_test_segments": int(test_seg_idx.size),
            "val_file_macro_f1_best": float(best_val_f1),
            "test_file_accuracy": float(file_m["accuracy"]),
            "test_file_macro_f1": float(file_m["macro_f1"]),
            "test_file_macro_recall": float(file_m["macro_recall"]),
            "test_segment_accuracy": float(seg_m["accuracy"]),
            "test_segment_macro_f1": float(seg_m["macro_f1"]),
            "test_segment_macro_recall": float(seg_m["macro_recall"]),
        }
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_csv = out_dir / f"deep_cv_{target_col}_folds.csv"
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    epoch_df = pd.DataFrame(epoch_rows)
    epoch_csv = out_dir / f"deep_cv_{target_col}_epochs.csv"
    epoch_df.to_csv(epoch_csv, index=False, encoding="utf-8-sig")
    file_oof_csv = out_dir / f"deep_cv_{target_col}_file_oof.csv"
    if file_oof_rows:
        pd.concat(file_oof_rows, axis=0, ignore_index=True).to_csv(file_oof_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(file_oof_csv, index=False, encoding="utf-8-sig")

    if fold_df.empty:
        return {
            "target_col": target_col,
            "class_names": class_names,
            "status": "failed",
            "reason": "no_valid_folds_after_training",
            "fold_csv": str(fold_csv),
            "epoch_csv": str(epoch_csv),
            "file_oof_csv": str(file_oof_csv),
            "deploy_model_path": "",
            "deploy_label_map_path": "",
        }

    best_row_idx = int(
        fold_df.sort_values(by=["val_file_macro_f1_best", "test_file_macro_f1"], ascending=[False, False]).index[0]
    )
    best_row = fold_df.loc[best_row_idx]
    deploy_model_path = out_dir / f"deploy_{target_col}_best_model.pt"
    deploy_label_map_path = out_dir / f"deploy_{target_col}_label_map.json"
    src_ckpt = Path(str(best_row.get("best_checkpoint_path", "")))
    src_label = Path(str(best_row.get("best_label_map_path", "")))
    if src_ckpt.exists():
        shutil.copy2(src_ckpt, deploy_model_path)
    if src_label.exists():
        shutil.copy2(src_label, deploy_label_map_path)

    summary = {
        "target_col": target_col,
        "class_names": class_names,
        "n_folds": int(fold_df.shape[0]),
        "file_accuracy_mean": float(fold_df["test_file_accuracy"].mean()),
        "file_accuracy_std": float(fold_df["test_file_accuracy"].std(ddof=0)),
        "file_macro_f1_mean": float(fold_df["test_file_macro_f1"].mean()),
        "file_macro_f1_std": float(fold_df["test_file_macro_f1"].std(ddof=0)),
        "segment_accuracy_mean": float(fold_df["test_segment_accuracy"].mean()),
        "segment_macro_f1_mean": float(fold_df["test_segment_macro_f1"].mean()),
        "fold_csv": str(fold_csv),
        "epoch_csv": str(epoch_csv),
        "file_oof_csv": str(file_oof_csv),
        "split_strategy": str(cv.get("split_strategy", "unknown")),
        "deploy_fold_idx": int(best_row.get("fold_idx", -1)),
        "deploy_model_path": str(deploy_model_path),
        "deploy_label_map_path": str(deploy_label_map_path),
        "fold_checkpoint_paths": [
            str(x)
            for x in fold_df["best_checkpoint_path"].astype(str).tolist()
            if str(x).strip()
        ],
    }
    return summary


def _load_baseline_metrics(baseline_scan_dir: Path) -> Dict[str, dict]:
    csv_path = baseline_scan_dir / "classical_model_metrics.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty or "status" not in df.columns:
        return {}
    out = {}
    for target_col in ["coarse_label", "label"]:
        sub = df[(df["label_col"] == target_col) & (df["status"] == "ok")].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        best = sub.iloc[0].to_dict()
        out[target_col] = {
            "model": str(best.get("model", "")),
            "accuracy": float(best.get("accuracy", np.nan)),
            "macro_f1": float(best.get("macro_f1", np.nan)),
        }
    return out


def run_health_deep(
    input_dir: Path,
    output_dir: Path,
    window_sec: float = 60.0,
    target_points: int = 300,
    min_valid_ratio: float = 0.6,
    clip_low_pct: float = 0.5,
    clip_high_pct: float = 99.5,
    median_window: int = 5,
    smooth_window: int = 9,
    target: str = "both",
    max_epochs: int = 35,
    patience: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    mixup_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    model_variant: str = "enhanced_reslstm",
    tta_shifts: List[int] = None,
    cv_mode: str = "group_by_day",
    merge_fine_labels: bool = True,
    use_balanced_sampler: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
    baseline_scan_dir: Path = None,
):
    set_seed(int(seed))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required, but torch.cuda.is_available() is False.")
    device = torch.device("cuda:0")
    log(f"Using GPU: {torch.cuda.get_device_name(0)}")

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    out_dir = ensure_dir(Path(output_dir))

    bundle = build_segment_bundle(
        input_dir=input_dir,
        window_sec=float(window_sec),
        target_points=int(target_points),
        min_valid_ratio=float(min_valid_ratio),
        clip_low_pct=float(clip_low_pct),
        clip_high_pct=float(clip_high_pct),
        median_window=int(median_window),
        smooth_window=int(smooth_window),
        merge_fine_labels=bool(merge_fine_labels),
    )

    bundle.file_df.to_csv(out_dir / "deep_file_manifest.csv", index=False, encoding="utf-8-sig")
    bundle.skipped_df.to_csv(out_dir / "deep_skipped_files.csv", index=False, encoding="utf-8-sig")

    targets: List[str] = []
    if target in ("coarse", "both"):
        targets.append("coarse_label")
    if target in ("fine", "both"):
        targets.append("label")

    summaries: Dict[str, dict] = {}
    for target_col in targets:
        summaries[target_col] = _train_cv_for_target(
            bundle=bundle,
            target_col=target_col,
            out_dir=out_dir,
            seed=int(seed),
            cv_mode=str(cv_mode),
            max_epochs=int(max_epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            label_smoothing=float(label_smoothing),
            mixup_alpha=float(mixup_alpha),
            mixup_prob=float(mixup_prob),
            model_variant=str(model_variant),
            val_ratio=float(val_ratio),
            tta_shifts=list(tta_shifts or [0]),
            use_balanced_sampler=bool(use_balanced_sampler),
            device=device,
            checkpoint_context={
                "window_sec": float(window_sec),
                "min_valid_ratio": float(min_valid_ratio),
                "clip_low_pct": float(clip_low_pct),
                "clip_high_pct": float(clip_high_pct),
                "median_window": int(median_window),
                "smooth_window": int(smooth_window),
                "merge_fine_labels": bool(merge_fine_labels),
            },
        )

    baseline = _load_baseline_metrics(Path(baseline_scan_dir)) if baseline_scan_dir else {}
    compare = {}
    if "coarse_label" in summaries and "coarse_label" in baseline and summaries["coarse_label"].get("status", "ok") != "failed":
        compare["coarse_label"] = {
            "deep_file_macro_f1_mean": float(summaries["coarse_label"]["file_macro_f1_mean"]),
            "baseline_macro_f1": float(baseline["coarse_label"]["macro_f1"]),
            "delta_macro_f1": float(summaries["coarse_label"]["file_macro_f1_mean"] - baseline["coarse_label"]["macro_f1"]),
            "deep_file_accuracy_mean": float(summaries["coarse_label"]["file_accuracy_mean"]),
            "baseline_accuracy": float(baseline["coarse_label"]["accuracy"]),
            "delta_accuracy": float(summaries["coarse_label"]["file_accuracy_mean"] - baseline["coarse_label"]["accuracy"]),
        }
    if "label" in summaries and "label" in baseline and summaries["label"].get("status", "ok") != "failed":
        compare["label"] = {
            "deep_file_macro_f1_mean": float(summaries["label"]["file_macro_f1_mean"]),
            "baseline_macro_f1": float(baseline["label"]["macro_f1"]),
            "delta_macro_f1": float(summaries["label"]["file_macro_f1_mean"] - baseline["label"]["macro_f1"]),
            "deep_file_accuracy_mean": float(summaries["label"]["file_accuracy_mean"]),
            "baseline_accuracy": float(baseline["label"]["accuracy"]),
            "delta_accuracy": float(summaries["label"]["file_accuracy_mean"] - baseline["label"]["accuracy"]),
        }

    summary = {
        "task": "health_signal_deep_learning_cv",
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "n_files": int(bundle.file_df.shape[0]),
        "n_segments": int(bundle.X.shape[0]),
        "skipped_files": int(bundle.skipped_df.shape[0]),
        "window_sec": float(window_sec),
        "target_points": int(target_points),
        "target": target,
        "model_variant": model_variant,
        "cv_mode": str(cv_mode),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "label_smoothing": float(label_smoothing),
        "mixup_alpha": float(mixup_alpha),
        "mixup_prob": float(mixup_prob),
        "tta_shifts": list(tta_shifts or [0]),
        "merge_fine_labels": bool(merge_fine_labels),
        "use_balanced_sampler": bool(use_balanced_sampler),
        "seed": int(seed),
        "targets": summaries,
        "baseline_best": baseline,
        "compare_to_baseline": compare,
    }
    save_json(out_dir / "deep_summary.json", summary)
    return summary


def main():
    args = _parse_args()
    out_dir = _resolve_output_dir(args.output_dir)
    tta_shifts = parse_tta_shifts(args.tta_shifts)
    summary = run_health_deep(
        input_dir=Path(args.input_dir),
        output_dir=out_dir,
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
        tta_shifts=tta_shifts,
        cv_mode=str(args.cv_mode),
        merge_fine_labels=bool(args.merge_fine_labels),
        use_balanced_sampler=bool(args.use_balanced_sampler),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        baseline_scan_dir=Path(args.baseline_scan_dir) if args.baseline_scan_dir else None,
    )
    print("output_dir:", summary["output_dir"])
    print("gpu_name:", summary["gpu_name"])
    print("n_files:", summary["n_files"])
    print("n_segments:", summary["n_segments"])
    if "coarse_label" in summary["targets"] and summary["targets"]["coarse_label"].get("status", "ok") != "failed":
        print(
            "coarse_file_macro_f1_mean:",
            f"{summary['targets']['coarse_label']['file_macro_f1_mean']:.6f}",
        )
    if "label" in summary["targets"] and summary["targets"]["label"].get("status", "ok") != "failed":
        print(
            "fine_file_macro_f1_mean:",
            f"{summary['targets']['label']['file_macro_f1_mean']:.6f}",
        )


if __name__ == "__main__":
    main()
