from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SegmentItem:
    x: np.ndarray
    segment_idx: int
    start_s: float
    end_s: float
    valid_ratio: float


def canonicalize_signal(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"time_s", "ch1", "ch2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = frame.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["ch1"] = pd.to_numeric(df["ch1"], errors="coerce")
    df["ch2"] = pd.to_numeric(df["ch2"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["time_s"])

    # Duplicate timestamps can appear in source files; merge them to keep interpolation stable.
    df = df.groupby("time_s", as_index=False).mean(numeric_only=True)
    df = df.sort_values("time_s").reset_index(drop=True)
    if df.empty:
        raise ValueError("Signal frame is empty after canonicalization")

    df["time_s"] = df["time_s"] - float(df["time_s"].iloc[0])
    return df


def _fill_channel(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        raise ValueError("No valid values in channel")
    if valid.sum() == 1:
        return np.full(values.shape, values[valid][0], dtype=np.float64)
    return np.interp(time_s, time_s[valid], values[valid]).astype(np.float64)


def _median_filter(signal: np.ndarray, window: int = 5) -> np.ndarray:
    window = max(1, int(window))
    return pd.Series(signal).rolling(window=window, min_periods=1, center=True).median().to_numpy(dtype=np.float64)


def _moving_average(signal: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    return pd.Series(signal).rolling(window=window, min_periods=1, center=True).mean().to_numpy(dtype=np.float64)


def _detrend(time_s: np.ndarray, signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return signal
    t = time_s - float(time_s.mean())
    var_t = float(np.var(t))
    if var_t < 1e-12:
        return signal
    slope = float(np.cov(t, signal, bias=True)[0, 1] / var_t)
    intercept = float(signal.mean() - slope * t.mean())
    trend = slope * t + intercept
    return signal - trend


def _robust_channel_preprocess(
    time_s: np.ndarray,
    values: np.ndarray,
    clip_low_pct: float,
    clip_high_pct: float,
    median_window: int,
    smooth_window: int,
    detrend: bool,
) -> np.ndarray:
    x = _fill_channel(time_s, values)

    low = float(np.percentile(x, clip_low_pct))
    high = float(np.percentile(x, clip_high_pct))
    if high > low:
        x = np.clip(x, low, high)

    x = _median_filter(x, window=median_window)
    x = _moving_average(x, window=smooth_window)
    if detrend:
        x = _detrend(time_s, x)
    return x


def _robust_normalize_per_channel(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32).copy()
    for idx in range(out.shape[0]):
        median = float(np.median(out[idx]))
        mad = float(np.median(np.abs(out[idx] - median)))
        scale = 1.4826 * mad
        if scale < 1e-6:
            out[idx] = 0.0
        else:
            out[idx] = (out[idx] - median) / scale
    return out


def _make_fixed_edges(offset_sec: float, segment_count: int, window_sec: float) -> List[Tuple[float, float]]:
    return [
        (offset_sec + idx * window_sec, offset_sec + (idx + 1) * window_sec)
        for idx in range(segment_count)
    ]


def _make_adaptive_edges(
    time_s: np.ndarray,
    ch1: np.ndarray,
    ch2: np.ndarray,
    offset_sec: float,
    segment_count: int,
    window_sec: float,
    boundary_search_radius: float,
    clip_low_pct: float,
    clip_high_pct: float,
    median_window: int,
    smooth_window: int,
    detrend: bool,
) -> List[Tuple[float, float]]:
    if segment_count <= 1:
        return _make_fixed_edges(offset_sec, segment_count, window_sec)

    ch1p = _robust_channel_preprocess(
        time_s,
        ch1,
        clip_low_pct=clip_low_pct,
        clip_high_pct=clip_high_pct,
        median_window=median_window,
        smooth_window=smooth_window,
        detrend=detrend,
    )
    ch2p = _robust_channel_preprocess(
        time_s,
        ch2,
        clip_low_pct=clip_low_pct,
        clip_high_pct=clip_high_pct,
        median_window=median_window,
        smooth_window=smooth_window,
        detrend=detrend,
    )

    end_time = offset_sec + segment_count * window_sec
    uniform_t = np.arange(offset_sec, end_time + 1e-6, 0.1, dtype=np.float64)
    if uniform_t.size < 10:
        return _make_fixed_edges(offset_sec, segment_count, window_sec)

    s1 = np.interp(uniform_t, time_s, ch1p)
    s2 = np.interp(uniform_t, time_s, ch2p)
    energy = np.abs(s1) + np.abs(s2)
    grad = np.abs(np.gradient(energy))

    boundaries = [offset_sec]
    min_seg = 0.65 * window_sec
    for i in range(1, segment_count):
        nominal = offset_sec + i * window_sec
        lower = max(nominal - boundary_search_radius, boundaries[-1] + min_seg)
        remain = (segment_count - i) * min_seg
        upper = min(nominal + boundary_search_radius, end_time - remain)

        if upper <= lower:
            cand = nominal
        else:
            m = (uniform_t >= lower) & (uniform_t <= upper)
            if int(m.sum()) == 0:
                cand = nominal
            else:
                local_t = uniform_t[m]
                local_g = grad[m]
                cand = float(local_t[int(np.argmin(local_g))])

        if cand <= boundaries[-1]:
            cand = boundaries[-1] + 0.05
        boundaries.append(cand)
    boundaries.append(end_time)

    edges = []
    for i in range(segment_count):
        start_s = float(boundaries[i])
        end_s = float(boundaries[i + 1])
        if end_s <= start_s:
            continue
        edges.append((start_s, end_s))
    return edges


def segment_signal(
    frame: pd.DataFrame,
    window_sec: float = 30.0,
    target_points: int = 300,
    min_valid_ratio: float = 0.6,
    offset_sec: float = 0.0,
    clip_low_pct: float = 0.5,
    clip_high_pct: float = 99.5,
    median_window: int = 5,
    smooth_window: int = 9,
    detrend: bool = True,
    adaptive_boundaries: bool = False,
    boundary_search_radius: float = 2.0,
) -> List[SegmentItem]:
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if target_points < 8:
        raise ValueError("target_points too small")
    if offset_sec < 0:
        raise ValueError("offset_sec must be >= 0")
    if not (0.0 <= clip_low_pct < clip_high_pct <= 100.0):
        raise ValueError("clip percentiles must satisfy 0 <= low < high <= 100")

    df = canonicalize_signal(frame)
    time = df["time_s"].to_numpy(dtype=np.float64)
    ch1 = df["ch1"].to_numpy(dtype=np.float64)
    ch2 = df["ch2"].to_numpy(dtype=np.float64)

    max_time = float(time.max())
    if offset_sec >= max_time:
        return []
    segment_count = int((max_time - offset_sec) // window_sec)
    if segment_count <= 0:
        return []

    if adaptive_boundaries:
        edges = _make_adaptive_edges(
            time,
            ch1,
            ch2,
            offset_sec=offset_sec,
            segment_count=segment_count,
            window_sec=window_sec,
            boundary_search_radius=boundary_search_radius,
            clip_low_pct=clip_low_pct,
            clip_high_pct=clip_high_pct,
            median_window=median_window,
            smooth_window=smooth_window,
            detrend=detrend,
        )
    else:
        edges = _make_fixed_edges(offset_sec, segment_count, window_sec)

    items: List[SegmentItem] = []
    for seg_idx, (start_s, end_s) in enumerate(edges):
        mask = (time >= start_s) & (time < end_s)
        if int(mask.sum()) < 2:
            continue

        seg_time = time[mask]
        seg_ch1 = ch1[mask]
        seg_ch2 = ch2[mask]

        non_na = np.count_nonzero(~np.isnan(seg_ch1)) + np.count_nonzero(~np.isnan(seg_ch2))
        valid_ratio = float(non_na) / float(seg_time.size * 2)
        if valid_ratio < min_valid_ratio:
            continue

        try:
            ch1_filled = _robust_channel_preprocess(
                seg_time,
                seg_ch1,
                clip_low_pct=clip_low_pct,
                clip_high_pct=clip_high_pct,
                median_window=median_window,
                smooth_window=smooth_window,
                detrend=detrend,
            )
            ch2_filled = _robust_channel_preprocess(
                seg_time,
                seg_ch2,
                clip_low_pct=clip_low_pct,
                clip_high_pct=clip_high_pct,
                median_window=median_window,
                smooth_window=smooth_window,
                detrend=detrend,
            )
        except ValueError:
            continue

        target_t = np.linspace(start_s, end_s, num=target_points, endpoint=False, dtype=np.float64)
        ch1_rs = np.interp(target_t, seg_time, ch1_filled)
        ch2_rs = np.interp(target_t, seg_time, ch2_filled)

        x = np.stack([ch1_rs, ch2_rs], axis=0).astype(np.float32)
        if np.isnan(x).any():
            continue
        x = _robust_normalize_per_channel(x)

        items.append(
            SegmentItem(
                x=x,
                segment_idx=seg_idx,
                start_s=float(start_s),
                end_s=float(end_s),
                valid_ratio=valid_ratio,
            )
        )

    return items

