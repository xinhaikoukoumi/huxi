import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

from .io_zip import load_signal_from_zip
from .preprocess import segment_signal
from .utils import log, save_json


LETTER_LABEL_PATTERN = re.compile(r"AB[,\uFF0C]([A-Z])[,\uFF0C]")
DIGIT_LABEL_PATTERN = re.compile(r"AB[,\uFF0C]([0-9])[,\uFF0C]")
VALID_LABEL_MODES = ("letters", "digits")
VALID_CHANNEL_MODES = ("dual", "ch1", "ch2")


@dataclass
class CacheBundle:
    X: np.ndarray
    y: np.ndarray
    source_files: np.ndarray
    segment_idx: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    label_map: Dict[str, int]


class MorseSegmentDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        augment: bool = False,
        shift_max: int = 8,
        noise_std: float = 0.01,
        scale_min: float = 0.85,
        scale_max: float = 1.15,
        drift_max: float = 0.08,
        time_mask_prob: float = 0.35,
        time_mask_max_width: int = 24,
    ):
        self.X = X[indices]
        self.y = y[indices]
        self.augment = augment
        self.shift_max = int(max(0, shift_max))
        self.noise_std = float(max(0.0, noise_std))
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if self.scale_max < self.scale_min:
            self.scale_min, self.scale_max = self.scale_max, self.scale_min
        self.drift_max = float(max(0.0, drift_max))
        self.time_mask_prob = float(min(max(time_mask_prob, 0.0), 1.0))
        self.time_mask_max_width = int(max(4, time_mask_max_width))

    def __len__(self):
        return int(self.y.shape[0])

    @staticmethod
    def _time_mask(x: np.ndarray, max_width: int = 24) -> np.ndarray:
        t = x.shape[1]
        if t < 16:
            return x
        width = int(np.random.randint(4, max_width + 1))
        width = min(width, t - 1)
        start = int(np.random.randint(0, t - width))
        x[:, start : start + width] = 0.0
        return x

    def __getitem__(self, idx: int):
        x_np = self.X[idx].copy()
        if self.augment:
            if self.shift_max > 0:
                shift = int(np.random.randint(-self.shift_max, self.shift_max + 1))
                if shift != 0:
                    x_np = np.roll(x_np, shift=shift, axis=1)

            # Random amplitude scaling per channel.
            scales = np.random.uniform(
                self.scale_min,
                self.scale_max,
                size=(x_np.shape[0], 1),
            ).astype(np.float32)
            x_np = x_np * scales

            # Random small baseline drift.
            t = np.linspace(-1.0, 1.0, num=x_np.shape[1], dtype=np.float32)
            slope = np.random.uniform(
                -self.drift_max,
                self.drift_max,
                size=(x_np.shape[0], 1),
            ).astype(np.float32)
            x_np = x_np + slope * t.reshape(1, -1)

            if np.random.rand() < self.time_mask_prob:
                x_np = self._time_mask(x_np, max_width=self.time_mask_max_width)

            if self.noise_std > 0:
                std = np.random.uniform(0.3, 1.0) * self.noise_std
                noise = np.random.normal(0.0, std, size=x_np.shape).astype(np.float32)
                x_np = x_np + noise
        x = torch.from_numpy(x_np).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def channel_mode_to_indices(channel_mode: str):
    mode = str(channel_mode).lower()
    if mode == "dual":
        return [0, 1]
    if mode == "ch1":
        return [0]
    if mode == "ch2":
        return [1]
    raise ValueError(f"Invalid channel_mode={channel_mode}, must be one of {VALID_CHANNEL_MODES}")


def select_channel_mode(x: np.ndarray, channel_mode: str) -> np.ndarray:
    mode = str(channel_mode).lower()
    if x.ndim != 2:
        raise ValueError(f"Expected x shape=(C,T), got ndim={x.ndim}")
    if x.shape[0] < 2:
        raise ValueError(f"Expected at least 2 input channels in raw segment, got {x.shape[0]}")
    idx = channel_mode_to_indices(mode)
    return x[idx, :].astype(np.float32)


def extract_label_from_filename(filename: str, label_mode: str = "letters") -> Optional[str]:
    mode = str(label_mode).lower()
    if mode not in VALID_LABEL_MODES:
        raise ValueError(f"Invalid label_mode={label_mode}, must be one of {VALID_LABEL_MODES}")
    pattern = LETTER_LABEL_PATTERN if mode == "letters" else DIGIT_LABEL_PATTERN
    match = pattern.search(filename)
    if not match:
        return None
    return match.group(1)


def scan_training_zips(train_dir) -> List[Path]:
    train_dir = Path(train_dir)
    return sorted(train_dir.glob("*.zip"))


def parse_segment_offsets(offsets_sec=None) -> List[float]:
    if offsets_sec is None:
        return [0.0]

    if isinstance(offsets_sec, str):
        raw_values = offsets_sec.split(",")
    else:
        raw_values = list(offsets_sec)

    parsed: List[float] = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        val = float(text)
        if val < 0:
            raise ValueError("segment offsets must be >= 0")
        parsed.append(val)

    if not parsed:
        return [0.0]
    return sorted(set(parsed))


def build_cache(
    train_dir,
    cache_path,
    meta_path,
    label_map_path,
    window_sec: float = 30.0,
    target_points: int = 300,
    min_valid_ratio: float = 0.6,
    clip_low_pct: float = 0.5,
    clip_high_pct: float = 99.5,
    median_window: int = 5,
    smooth_window: int = 9,
    detrend: bool = True,
    label_mode: str = "letters",
    channel_mode: str = "dual",
    offsets_sec=None,
) -> CacheBundle:
    train_dir = Path(train_dir)
    cache_path = Path(cache_path)
    meta_path = Path(meta_path)
    label_map_path = Path(label_map_path)

    mode_label = str(label_mode).lower()
    if mode_label not in VALID_LABEL_MODES:
        raise ValueError(f"Invalid label_mode={label_mode}, must be one of {VALID_LABEL_MODES}")
    mode = str(channel_mode).lower()
    _ = channel_mode_to_indices(mode)
    segment_offsets = parse_segment_offsets(offsets_sec)

    zips = scan_training_zips(train_dir)
    if not zips:
        raise ValueError(f"No zip files found in {train_dir}")

    labeled_files: List[Tuple[Path, str]] = []
    labels = set()
    for zip_path in zips:
        label = extract_label_from_filename(zip_path.name, label_mode=mode_label)
        if label is None:
            continue
        labels.add(label)
        labeled_files.append((zip_path, label))

    if not labeled_files:
        if mode_label == "digits":
            raise ValueError("No labeled zip files matched AB,<DIGIT>, pattern")
        raise ValueError("No labeled zip files matched AB,<LETTER>, pattern")

    label_names = sorted(labels)
    label_map = {lab: idx for idx, lab in enumerate(label_names)}
    save_json(label_map_path, label_map)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    source_files: List[str] = []
    segment_indices: List[int] = []
    start_s_list: List[float] = []
    end_s_list: List[float] = []
    meta_rows: List[dict] = []

    for zip_path, label in labeled_files:
        try:
            signal = load_signal_from_zip(zip_path)
        except Exception as exc:
            log(f"Skip bad zip {zip_path.name}: {exc}")
            continue

        for offset_sec in segment_offsets:
            try:
                segs = segment_signal(
                    signal.frame,
                    window_sec=window_sec,
                    target_points=target_points,
                    min_valid_ratio=min_valid_ratio,
                    offset_sec=float(offset_sec),
                    clip_low_pct=clip_low_pct,
                    clip_high_pct=clip_high_pct,
                    median_window=median_window,
                    smooth_window=smooth_window,
                    detrend=detrend,
                    adaptive_boundaries=False,
                )
            except Exception as exc:
                log(f"Skip bad offset for {zip_path.name} (offset={offset_sec}): {exc}")
                continue

            for seg in segs:
                item_id = len(X_list)
                X_list.append(select_channel_mode(seg.x, mode))
                y_list.append(label_map[label])
                source_files.append(zip_path.name)
                segment_indices.append(seg.segment_idx)
                start_s_list.append(seg.start_s)
                end_s_list.append(seg.end_s)
                meta_rows.append(
                    {
                        "id": item_id,
                        "label": label,
                        "label_id": label_map[label],
                        "source_file": zip_path.name,
                        "segment_idx": seg.segment_idx,
                        "offset_sec": float(offset_sec),
                        "start_s": seg.start_s,
                        "end_s": seg.end_s,
                        "valid_ratio": seg.valid_ratio,
                        "source_format": signal.source_format,
                    }
                )

    if not X_list:
        raise ValueError("No valid segments were created")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    source_files_arr = np.asarray(source_files, dtype="<U256")
    segment_idx_arr = np.asarray(segment_indices, dtype=np.int64)
    start_s_arr = np.asarray(start_s_list, dtype=np.float32)
    end_s_arr = np.asarray(end_s_list, dtype=np.float32)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        source_files=source_files_arr,
        segment_idx=segment_idx_arr,
        start_s=start_s_arr,
        end_s=end_s_arr,
    )
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False, encoding="utf-8")

    log(f"Cache saved: {cache_path} (segments={len(X_list)})")
    return CacheBundle(
        X=X,
        y=y,
        source_files=source_files_arr,
        segment_idx=segment_idx_arr,
        start_s=start_s_arr,
        end_s=end_s_arr,
        label_map=label_map,
    )


def load_cache(cache_path, label_map_path) -> CacheBundle:
    cache_path = Path(cache_path)
    label_map_path = Path(label_map_path)
    loaded = np.load(cache_path)

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return CacheBundle(
        X=loaded["X"].astype(np.float32),
        y=loaded["y"].astype(np.int64),
        source_files=loaded["source_files"],
        segment_idx=loaded["segment_idx"].astype(np.int64),
        start_s=loaded["start_s"].astype(np.float32),
        end_s=loaded["end_s"].astype(np.float32),
        label_map=label_map,
    )


def stratified_random_split(
    y: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )
    y_train_val = y[train_val_idx]
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_relative,
        random_state=seed,
        stratify=y_train_val,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def group_by_file_split(
    y: np.ndarray,
    source_files: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = np.arange(len(y))
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(indices, y, groups=source_files))

    relative_val = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_sub, val_sub = next(
        gss_val.split(
            train_val_idx,
            y[train_val_idx],
            groups=source_files[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_sub]
    val_idx = train_val_idx[val_sub]
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def stratified_group_by_file_split(
    y: np.ndarray,
    source_files: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Grouped split that preserves label coverage as much as possible.
    Each file belongs to a single label in this dataset, so splitting by
    label-specific file buckets avoids severe class dropout in val/test.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    y = np.asarray(y)
    source_files = np.asarray(source_files)
    rng = np.random.RandomState(seed)

    # Map each file(group) -> label. Files are expected to be single-label.
    group_to_label: Dict[str, int] = {}
    for g in np.unique(source_files):
        labels = np.unique(y[source_files == g])
        if labels.size == 0:
            continue
        if labels.size > 1:
            # Fallback to majority label if mixed (should not happen here).
            counts = np.bincount(y[source_files == g])
            group_to_label[g] = int(np.argmax(counts))
        else:
            group_to_label[g] = int(labels[0])

    label_to_groups: Dict[int, List[str]] = {}
    for g, lab in group_to_label.items():
        label_to_groups.setdefault(lab, []).append(g)

    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    for lab, groups in label_to_groups.items():
        groups = groups.copy()
        rng.shuffle(groups)
        n = len(groups)
        if n == 1:
            train_groups.extend(groups)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0

        # Keep at least one file in train.
        while n_test + n_val > n - 1:
            if n_val > 0:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        test_groups.extend(groups[:n_test])
        val_groups.extend(groups[n_test : n_test + n_val])
        train_groups.extend(groups[n_test + n_val :])

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    # Ensure disjoint.
    val_set = val_set - train_set
    test_set = test_set - train_set - val_set

    indices = np.arange(len(y))
    train_idx = indices[np.array([g in train_set for g in source_files])]
    val_idx = indices[np.array([g in val_set for g in source_files])]
    test_idx = indices[np.array([g in test_set for g in source_files])]

    # Safety fallback: avoid empty splits.
    if len(val_idx) == 0:
        val_idx = train_idx[: max(1, min(64, len(train_idx) // 10))]
        train_idx = np.setdiff1d(train_idx, val_idx, assume_unique=False)
    if len(test_idx) == 0:
        test_idx = train_idx[: max(1, min(64, len(train_idx) // 10))]
        train_idx = np.setdiff1d(train_idx, test_idx, assume_unique=False)

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
