import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .io_zip import load_signal_from_zip
from .preprocess import canonicalize_signal, segment_signal


HEALTH_LABEL_PATTERN = re.compile(r"AB[,\uFF0C]([^,\uFF0C]+)[,\uFF0C]")
TIMESTAMP_PATTERN = re.compile(r"-(\d{6,14})(?:\(\d+\))?$")
COARSE_LABEL_MAP = {
    "\u53e3\u547c\u5438": "normal",
    "\u9f3b\u5b50\u547c\u5438": "normal",
    "\u5de6\u9f3b\u585e": "nasal_block",
    "\u53f3\u9f3b\u585e": "nasal_block",
    "\u5c0f\u8dd1": "exercise",
    "\u8dd1\u6b65": "exercise",
    "\u54b3\u55fd": "cough",
}
RESP_BAND = (0.08, 0.7)
HIGH_BAND = (0.8, 2.0)
FEATURE_VERSION = "v2_freq_time_hybrid"
META_COLUMNS = {"zip_file", "label", "coarse_label", "source_format", "group_day", "feature_set", "feature_version"}


def parse_health_label(filename) -> str:
    name = Path(filename).name
    m = HEALTH_LABEL_PATTERN.search(name)
    if not m:
        raise ValueError(f"Failed to parse health label from filename: {name}")
    return m.group(1).strip()


def map_coarse_label(label: str) -> str:
    key = str(label).strip()
    if key not in COARSE_LABEL_MAP:
        raise ValueError(f"Unknown health label for coarse mapping: {label}")
    return COARSE_LABEL_MAP[key]


def parse_group_day_from_filename(filename: str) -> str:
    m = TIMESTAMP_PATTERN.search(Path(filename).stem)
    if not m:
        return "unknown"
    token = m.group(1)
    return token[:4] if len(token) >= 4 else token


def _estimate_fs_hz(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(1.0 / np.median(dt)) if int(dt.size) > 0 else 0.0


def _fill_channel(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if int(valid.sum()) == 0:
        raise ValueError("Channel has no valid values")
    if int(valid.sum()) == 1:
        return np.full(values.shape, float(values[valid][0]), dtype=np.float64)
    return np.interp(time_s, time_s[valid], values[valid]).astype(np.float64)


def _safe_skew(x: np.ndarray) -> float:
    std = float(np.std(x))
    if std < 1e-12:
        return 0.0
    z = (x - float(np.mean(x))) / std
    return float(np.mean(z ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    std = float(np.std(x))
    if std < 1e-12:
        return 0.0
    z = (x - float(np.mean(x))) / std
    return float(np.mean(z ** 4) - 3.0)


def _rolling_mean(x: np.ndarray, win: int = 5) -> np.ndarray:
    w = int(max(1, win))
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def _local_maxima_indices(x: np.ndarray, min_distance: int = 1) -> np.ndarray:
    if x.size < 3:
        return np.asarray([], dtype=np.int64)
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if idx.size <= 1:
        return idx.astype(np.int64)
    min_dist = int(max(1, min_distance))
    kept = [int(idx[0])]
    for i in idx[1:]:
        if int(i) - int(kept[-1]) >= min_dist:
            kept.append(int(i))
        elif x[int(i)] > x[int(kept[-1])]:
            kept[-1] = int(i)
    return np.asarray(kept, dtype=np.int64)


def _frequency_features(x: np.ndarray, fs_hz: float) -> Dict[str, float]:
    n = int(x.size)
    if n < 8 or fs_hz <= 0:
        return {"dominant_freq": 0.0, "spectral_entropy": 0.0, "hf_ratio": 0.0, "bandpower_ratio": 0.0, "centroid": 0.0}
    x0 = x - float(np.mean(x))
    spec = np.abs(np.fft.rfft(x0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    total = float(np.sum(spec))
    if total <= 1e-12:
        return {"dominant_freq": 0.0, "spectral_entropy": 0.0, "hf_ratio": 0.0, "bandpower_ratio": 0.0, "centroid": 0.0}
    spec_no_dc = spec.copy()
    if int(spec_no_dc.size) > 0:
        spec_no_dc[0] = 0.0
    dom = float(freqs[int(np.argmax(spec_no_dc))])
    pnorm = spec / total
    ent = float(-np.sum(pnorm * np.log(pnorm + 1e-12)) / np.log(max(2, pnorm.size)))
    hf_mask = (freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])
    hf = float(np.sum(spec[hf_mask]) / total)
    rb_mask = (freqs >= RESP_BAND[0]) & (freqs <= RESP_BAND[1])
    rb = float(np.sum(spec[rb_mask]))
    rb_ratio = float(rb / total)
    centroid = float(np.sum(freqs[rb_mask] * spec[rb_mask]) / rb) if rb > 1e-12 else 0.0
    return {"dominant_freq": dom, "spectral_entropy": ent, "hf_ratio": hf, "bandpower_ratio": rb_ratio, "centroid": centroid}


def _periodic_features(x: np.ndarray, time_s: np.ndarray, fs_hz: float) -> Dict[str, float]:
    if int(x.size) < 8 or fs_hz <= 0:
        return {"peak_interval_mean_s": 0.0, "peak_interval_std_s": 0.0, "peak_interval_cv": 0.0, "breaths_per_min": 0.0}
    smooth = _rolling_mean(x, win=max(3, int(fs_hz * 0.4)))
    peaks = _local_maxima_indices(smooth, min_distance=max(1, int(fs_hz * 0.8)))
    if peaks.size < 2:
        return {"peak_interval_mean_s": 0.0, "peak_interval_std_s": 0.0, "peak_interval_cv": 0.0, "breaths_per_min": 0.0}
    intervals = np.diff(time_s[peaks]).astype(np.float64)
    intervals = intervals[(intervals > 0.3) & (intervals < 15.0)]
    if intervals.size == 0:
        return {"peak_interval_mean_s": 0.0, "peak_interval_std_s": 0.0, "peak_interval_cv": 0.0, "breaths_per_min": 0.0}
    mean_i = float(np.mean(intervals))
    std_i = float(np.std(intervals))
    cv_i = float(std_i / (mean_i + 1e-12))
    bpm = float(60.0 / (mean_i + 1e-12))
    return {"peak_interval_mean_s": mean_i, "peak_interval_std_s": std_i, "peak_interval_cv": cv_i, "breaths_per_min": bpm}


def _cross_channel_features(ch1: np.ndarray, ch2: np.ndarray, fs_hz: float) -> Dict[str, float]:
    x1 = ch1 - float(np.mean(ch1))
    x2 = ch2 - float(np.mean(ch2))
    s1 = float(np.std(x1))
    s2 = float(np.std(x2))
    corr = float(np.corrcoef(x1, x2)[0, 1]) if (s1 >= 1e-12 and s2 >= 1e-12) else 0.0
    e1 = float(np.sum(x1 ** 2))
    e2 = float(np.sum(x2 ** 2))
    er = float((e1 + 1e-12) / (e2 + 1e-12))
    if fs_hz <= 0 or x1.size < 8:
        return {"corr_ch1_ch2": corr, "energy_ratio_ch1_ch2": er, "xcorr_peak_ch1_ch2": 0.0, "xcorr_lag_s_ch1_ch2": 0.0}
    max_lag = int(max(1, min(x1.size // 2, fs_hz * 3.0)))
    full = np.correlate(x1, x2, mode="full")
    c = x1.size - 1
    local = full[c - max_lag : c + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    den = float(np.sqrt(np.sum(x1 ** 2) * np.sum(x2 ** 2)) + 1e-12)
    idx = int(np.argmax(np.abs(local)))
    return {
        "corr_ch1_ch2": corr,
        "energy_ratio_ch1_ch2": er,
        "xcorr_peak_ch1_ch2": float(np.abs(local[idx]) / den),
        "xcorr_lag_s_ch1_ch2": float(lags[idx] / fs_hz),
    }


def _channel_features(prefix: str, time_s: np.ndarray, values: np.ndarray, fs_hz: float, feature_set: str) -> Dict[str, float]:
    x = _fill_channel(time_s, values)
    out = {}
    f = _frequency_features(x, fs_hz)
    out[f"{prefix}_dominant_freq"] = float(f["dominant_freq"])
    out[f"{prefix}_spectral_entropy"] = float(f["spectral_entropy"])
    out[f"{prefix}_hf_ratio"] = float(f["hf_ratio"])
    out[f"{prefix}_bandpower_ratio"] = float(f["bandpower_ratio"])
    out[f"{prefix}_centroid"] = float(f["centroid"])
    if feature_set == "freq_only":
        return out
    out[f"{prefix}_mean"] = float(np.mean(x))
    out[f"{prefix}_std"] = float(np.std(x))
    out[f"{prefix}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    out[f"{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))
    out[f"{prefix}_skew"] = _safe_skew(x)
    out[f"{prefix}_kurtosis"] = _safe_kurtosis(x)
    signs = np.signbit(x - float(np.median(x)))
    out[f"{prefix}_zero_crossing_rate"] = float(np.mean(np.diff(signs) != 0)) if x.size > 2 else 0.0
    p = _periodic_features(x, time_s, fs_hz)
    for k, v in p.items():
        out[f"{prefix}_{k}"] = float(v)
    return out


def _aggregate_window_features(time_s: np.ndarray, ch1: np.ndarray, ch2: np.ndarray, fs_hz: float, window_sec: float, feature_set: str) -> Dict[str, float]:
    if float(window_sec) <= 0:
        return {}
    duration = float(time_s[-1] - time_s[0]) if int(time_s.size) > 0 else 0.0
    if duration <= float(window_sec):
        return {"window_count": 0.0}
    rows = []
    cur = float(time_s[0])
    end = float(time_s[-1])
    while cur + float(window_sec) <= end + 1e-9:
        m = (time_s >= cur) & (time_s < cur + float(window_sec))
        if int(m.sum()) >= 8:
            wt = time_s[m]
            w1 = ch1[m]
            w2 = ch2[m]
            row = {}
            row.update(_channel_features("ch1", wt, w1, fs_hz, feature_set))
            row.update(_channel_features("ch2", wt, w2, fs_hz, feature_set))
            row.update(_cross_channel_features(_fill_channel(wt, w1), _fill_channel(wt, w2), fs_hz))
            rows.append(row)
        cur += float(window_sec)
    if not rows:
        return {"window_count": 0.0}
    df = pd.DataFrame(rows)
    out = {"window_count": float(df.shape[0])}
    pref = f"w{int(round(float(window_sec)))}s"
    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        out[f"{pref}_{col}_mean"] = float(np.mean(x))
        out[f"{pref}_{col}_p25"] = float(np.percentile(x, 25))
        out[f"{pref}_{col}_p75"] = float(np.percentile(x, 75))
    return out


def extract_hybrid_features(zip_path, feature_set: str = "freq_time_hybrid", window_sec_for_stats: float = 10.0, segment_window_sec: float = 0.0) -> Dict[str, float]:
    mode = str(feature_set).lower()
    if mode not in ("freq_only", "freq_time_hybrid"):
        raise ValueError("feature_set must be one of: freq_only, freq_time_hybrid")
    zp = Path(zip_path)
    signal = load_signal_from_zip(zp)
    frame = canonicalize_signal(signal.frame)
    label = parse_health_label(zp.name)
    coarse = map_coarse_label(label)
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    ch1 = frame["ch1"].to_numpy(dtype=np.float64)
    ch2 = frame["ch2"].to_numpy(dtype=np.float64)
    fs_hz = _estimate_fs_hz(time_s)
    duration_s = float(time_s.max() - time_s.min()) if int(time_s.size) > 0 else 0.0
    out = {
        "zip_file": zp.name,
        "label": label,
        "coarse_label": coarse,
        "source_format": str(signal.source_format),
        "group_day": parse_group_day_from_filename(zp.name),
        "feature_set": mode,
        "feature_version": FEATURE_VERSION,
        "num_rows": int(frame.shape[0]),
        "duration_s": duration_s,
        "fs_hz": fs_hz,
        "window_sec_for_stats": float(max(0.0, window_sec_for_stats)),
        "segment_window_sec": float(max(0.0, segment_window_sec)),
    }
    ch1f = _fill_channel(time_s, ch1)
    ch2f = _fill_channel(time_s, ch2)
    out.update(_channel_features("ch1", time_s, ch1f, fs_hz, mode))
    out.update(_channel_features("ch2", time_s, ch2f, fs_hz, mode))
    out.update(_cross_channel_features(ch1f, ch2f, fs_hz))
    if float(window_sec_for_stats) > 0:
        out.update(_aggregate_window_features(time_s, ch1f, ch2f, fs_hz, float(window_sec_for_stats), mode))
    if float(segment_window_sec) > 0:
        segs = segment_signal(frame, window_sec=float(segment_window_sec), target_points=300, min_valid_ratio=0.6)
        out["segment_count"] = int(len(segs))
    else:
        out["segment_count"] = 0
    if mode == "freq_only":
        keep = set(META_COLUMNS) | {"num_rows", "duration_s", "fs_hz", "window_sec_for_stats", "segment_window_sec", "segment_count", "window_count"}
        for k in list(out.keys()):
            if k in keep:
                continue
            if any(x in k for x in ("dominant_freq", "spectral_entropy", "hf_ratio", "bandpower_ratio", "centroid", "xcorr_", "corr_ch1_ch2", "energy_ratio_ch1_ch2")):
                continue
            del out[k]
    return out


def extract_file_features(zip_path, segment_window_sec: float = 0.0, feature_set: str = "freq_time_hybrid", window_sec_for_stats: float = 10.0) -> Dict[str, float]:
    return extract_hybrid_features(zip_path, feature_set=feature_set, window_sec_for_stats=window_sec_for_stats, segment_window_sec=segment_window_sec)


def _pick_feature_columns(features_df: pd.DataFrame, extra_drop: List[str] = None) -> List[str]:
    drop = set(META_COLUMNS)
    if extra_drop:
        drop.update(extra_drop)
    cols: List[str] = []
    for col in features_df.columns:
        if col in drop:
            continue
        if pd.api.types.is_numeric_dtype(features_df[col]):
            cols.append(col)
    return cols


def _cluster_eval(X: np.ndarray, y_true: np.ndarray, pred: np.ndarray, k: int) -> Dict[str, float]:
    uniq = int(np.unique(pred).size)
    if uniq <= 1 or uniq >= int(X.shape[0]):
        sil = float("nan")
        dbi = float("nan")
    else:
        sil = float(silhouette_score(X, pred))
        dbi = float(davies_bouldin_score(X, pred))
    ari = float(adjusted_rand_score(y_true, pred))
    nmi = float(normalized_mutual_info_score(y_true, pred))
    return {"k": int(k), "n_clusters_found": uniq, "silhouette": sil, "davies_bouldin": dbi, "ari": ari, "nmi": nmi}


def run_unsupervised_analysis(features_df: pd.DataFrame, label_col: str) -> Dict[str, object]:
    if label_col not in features_df.columns:
        raise ValueError(f"label_col not found: {label_col}")
    if int(features_df.shape[0]) < 2:
        return {"status": "skipped", "reason": "insufficient_samples", "label_col": label_col}
    labels = features_df[label_col].astype(str).to_numpy()
    if int(np.unique(labels).size) < 2:
        return {"status": "skipped", "reason": "insufficient_classes", "label_col": label_col}

    feature_cols = _pick_feature_columns(features_df, extra_drop=["window_sec_for_stats", "segment_window_sec"])
    if not feature_cols:
        raise ValueError("No numeric features found for unsupervised analysis")

    X = features_df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=np.float64))

    _, y_true = np.unique(labels, return_inverse=True)
    n_samples = int(Xs.shape[0])
    k_max = min(6, n_samples - 1)
    k_values = list(range(2, k_max + 1)) if k_max >= 2 else []
    km_rows, ag_rows = [], []
    for k in k_values:
        km_pred = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xs)
        km_rows.append(_cluster_eval(Xs, y_true, km_pred, k))
        ag_pred = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        ag_rows.append(_cluster_eval(Xs, y_true, ag_pred, k))

    pca = PCA(n_components=min(2, Xs.shape[1]))
    pca_xy = pca.fit_transform(Xs)
    if pca_xy.shape[1] == 1:
        pca_xy = np.concatenate([pca_xy, np.zeros((pca_xy.shape[0], 1), dtype=np.float64)], axis=1)
    dist = np.sqrt(np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2))
    pairs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairs.append((i, j, float(dist[i, j])))
    pairs.sort(key=lambda x: x[2])
    return {
        "status": "ok",
        "label_col": label_col,
        "n_samples": n_samples,
        "n_labels": int(np.unique(labels).size),
        "labels": [str(x) for x in labels.tolist()],
        "zip_files": features_df["zip_file"].astype(str).tolist(),
        "feature_cols": feature_cols,
        "k_values": k_values,
        "kmeans": km_rows,
        "agglomerative": ag_rows,
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
        "pca_points": [{"zip_file": str(features_df.iloc[i]["zip_file"]), "label": str(labels[i]), "pc1": float(pca_xy[i, 0]), "pc2": float(pca_xy[i, 1])} for i in range(n_samples)],
        "pairwise_distance": [[float(x) for x in row] for row in dist.tolist()],
        "nearest_pairs": [{"i": int(i), "j": int(j), "zip_i": str(features_df.iloc[i]["zip_file"]), "zip_j": str(features_df.iloc[j]["zip_file"]), "distance": float(d)} for i, j, d in pairs],
    }


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(df.shape[0], float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default)).to_numpy(dtype=np.float64)


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.mean(x))) / std


def run_rule_based_classifier(features_df: pd.DataFrame) -> pd.DataFrame:
    if int(features_df.shape[0]) == 0:
        return pd.DataFrame(columns=["zip_file", "label", "coarse_label", "pred_label", "trigger_rule", "is_correct"])
    df = features_df.copy().reset_index(drop=True)
    duration = _safe_col(df, "duration_s")
    std_mean = (_safe_col(df, "ch1_std") + _safe_col(df, "ch2_std")) / 2.0
    iqr_mean = (_safe_col(df, "ch1_iqr") + _safe_col(df, "ch2_iqr")) / 2.0
    amp_score = 0.6 * std_mean + 0.4 * iqr_mean
    hf_ratio_max = np.maximum(_safe_col(df, "ch1_hf_ratio"), _safe_col(df, "ch2_hf_ratio"))
    dom_freq_max = np.maximum(_safe_col(df, "ch1_dominant_freq"), _safe_col(df, "ch2_dominant_freq"))
    kurtosis_max = np.maximum(np.abs(_safe_col(df, "ch1_kurtosis")), np.abs(_safe_col(df, "ch2_kurtosis")))
    zcr_mean = (_safe_col(df, "ch1_zero_crossing_rate") + _safe_col(df, "ch2_zero_crossing_rate")) / 2.0
    bpm_mean = (_safe_col(df, "ch1_breaths_per_min") + _safe_col(df, "ch2_breaths_per_min")) / 2.0
    activity_score = _safe_zscore(std_mean) + _safe_zscore(zcr_mean) + _safe_zscore(bpm_mean)
    q25 = float(np.percentile(duration, 25))
    q75 = float(np.percentile(duration, 75))
    duration_thr = max(0.0, float(np.median(duration) - 1.5 * (q75 - q25)))
    dom_thr = float(np.percentile(dom_freq_max, 70))
    hf_thr = float(np.percentile(hf_ratio_max, 70))
    kurt_thr = float(np.percentile(kurtosis_max, 70))
    amp_thr = float(np.percentile(amp_score, 35))
    act_thr = float(np.percentile(activity_score, 65))

    rows = []
    for i, row in df.iterrows():
        is_cough = bool((duration[i] <= duration_thr) and ((dom_freq_max[i] >= dom_thr) or (hf_ratio_max[i] >= hf_thr) or (kurtosis_max[i] >= kurt_thr)))
        is_nasal = bool(amp_score[i] <= amp_thr)
        is_exercise = bool(activity_score[i] >= act_thr)
        if is_cough:
            pred, trig = "cough", "rule_cough_short_highfreq"
        elif is_nasal:
            pred, trig = "nasal_block", "rule_nasal_low_amplitude"
        elif is_exercise:
            pred, trig = "exercise", "rule_exercise_high_activity"
        else:
            pred, trig = "normal", "rule_default_normal"
        true_label = str(row.get("coarse_label", ""))
        rows.append(
            {
                "zip_file": str(row.get("zip_file", "")),
                "label": str(row.get("label", "")),
                "coarse_label": true_label,
                "pred_label": pred,
                "is_correct": int(pred == true_label) if true_label else np.nan,
                "trigger_rule": trig,
                "key_feature_values": json.dumps(
                    {
                        "duration_s": float(duration[i]),
                        "amp_score": float(amp_score[i]),
                        "dom_freq_max": float(dom_freq_max[i]),
                        "hf_ratio_max": float(hf_ratio_max[i]),
                        "kurtosis_max": float(kurtosis_max[i]),
                        "activity_score": float(activity_score[i]),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.attrs["thresholds"] = {
        "cough_duration_max": duration_thr,
        "cough_dom_freq_min": dom_thr,
        "cough_hf_ratio_min": hf_thr,
        "cough_kurtosis_min": kurt_thr,
        "nasal_amp_score_max": amp_thr,
        "exercise_activity_score_min": act_thr,
    }
    return out_df


def _make_cv_splits(y: np.ndarray, groups: np.ndarray, cv_mode: str, seed: int):
    mode = str(cv_mode).lower()
    splits: List[Tuple[np.ndarray, np.ndarray, str]] = []
    strategy = "none"
    reason = ""
    if mode == "group_by_day":
        uniq = sorted(set(groups.tolist()))
        if len(uniq) >= 2:
            for g in uniq:
                test_idx = np.where(groups == g)[0]
                train_idx = np.where(groups != g)[0]
                if int(test_idx.size) == 0 or int(train_idx.size) == 0:
                    continue
                if int(np.unique(y[train_idx]).size) < 2:
                    continue
                splits.append((train_idx, test_idx, f"group:{g}"))
            if splits:
                strategy = "group_by_day"
            else:
                reason = "no_valid_group_folds"
        else:
            reason = "insufficient_groups"

    if not splits:
        counts = np.bincount(y)
        nonzero = counts[counts > 0]
        min_class = int(nonzero.min()) if nonzero.size else 0
        if int(y.size) >= 4 and min_class >= 2:
            n_splits = int(min(3, min_class))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            for fold_idx, (tr, te) in enumerate(skf.split(np.zeros_like(y), y)):
                if int(np.unique(y[tr]).size) < 2:
                    continue
                splits.append((tr, te, f"stratified:{fold_idx}"))
            if splits:
                strategy = "stratified_kfold"
                reason = reason or "fallback_from_group_by_day"

    if not splits and int(y.size) >= 3 and int(np.unique(y).size) >= 2:
        idx = np.arange(y.size, dtype=np.int64)
        counts = np.bincount(y)
        nonzero = counts[counts > 0]
        stratify_vec = y if (nonzero.size > 0 and int(nonzero.min()) >= 2) else None
        tr, te = train_test_split(idx, test_size=0.34, random_state=int(seed), stratify=stratify_vec)
        if int(np.unique(y[tr]).size) >= 2:
            splits.append((tr, te, "holdout"))
            strategy = "stratified_holdout"
            reason = reason or "fallback_from_group_by_day"
    return strategy, reason, splits


def run_grouped_cv(
    features_df: pd.DataFrame,
    label_col: str = "coarse_label",
    feature_cols: List[str] = None,
    cv_mode: str = "group_by_day",
    group_col: str = "group_day",
    seed: int = 42,
) -> Dict[str, object]:
    if label_col not in features_df.columns:
        raise ValueError(f"label_col not found: {label_col}")
    if group_col not in features_df.columns:
        raise ValueError(f"group_col not found: {group_col}")
    if int(features_df.shape[0]) < 3:
        return {"status": "skipped", "reason": "insufficient_samples", "n_folds": 0, "split_strategy": "none", "splits": []}
    labels = features_df[label_col].astype(str).to_numpy()
    classes = sorted(np.unique(labels).tolist())
    if len(classes) < 2:
        return {"status": "skipped", "reason": "insufficient_classes", "n_folds": 0, "split_strategy": "none", "splits": []}
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.asarray([c2i[x] for x in labels], dtype=np.int64)
    groups = features_df[group_col].astype(str).to_numpy()
    strategy, reason, split_tuples = _make_cv_splits(y, groups, cv_mode=cv_mode, seed=int(seed))
    splits = []
    for tr, te, name in split_tuples:
        splits.append(
            {
                "fold_name": str(name),
                "train_idx": tr.astype(np.int64),
                "test_idx": te.astype(np.int64),
                "train_size": int(tr.size),
                "test_size": int(te.size),
                "test_groups": sorted(set(groups[te].tolist())),
            }
        )
    return {
        "status": "ok" if splits else "skipped",
        "reason": reason if not splits else "",
        "n_folds": int(len(splits)),
        "split_strategy": strategy,
        "splits": splits,
        "classes": classes,
        "label_col": label_col,
        "group_col": group_col,
        "cv_mode": cv_mode,
        "feature_cols": feature_cols if feature_cols is not None else _pick_feature_columns(features_df),
    }


def _build_models(seed: int = 42):
    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed))),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=int(seed))),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=int(seed),
            class_weight="balanced_subsample",
            min_samples_leaf=1,
        ),
    }


def _compute_feature_importance(X: np.ndarray, y: np.ndarray, feature_cols: List[str], seed: int = 42) -> pd.DataFrame:
    if int(np.unique(y).size) < 2 or int(X.shape[0]) < 4:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    rows = []
    rf = RandomForestClassifier(n_estimators=400, random_state=int(seed), class_weight="balanced_subsample")
    rf.fit(X, y)
    for feat, val in zip(feature_cols, rf.feature_importances_):
        rows.append({"model": "random_forest", "feature": str(feat), "importance": float(val)})
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed))),
        ]
    )
    try:
        lr.fit(X, y)
        coef = lr.named_steps["clf"].coef_
        imp = np.mean(np.abs(coef), axis=0)
        for feat, val in zip(feature_cols, imp):
            rows.append({"model": "logreg", "feature": str(feat), "importance": float(val)})
    except Exception:
        pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)


def run_classical_baselines(features_df: pd.DataFrame, label_col: str = "coarse_label", cv_mode: str = "group_by_day", seed: int = 42) -> Dict[str, object]:
    feature_cols = _pick_feature_columns(features_df, extra_drop=["window_sec_for_stats", "segment_window_sec"])
    X_df = features_df[feature_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
    X = X_df.to_numpy(dtype=np.float64)
    labels = features_df[label_col].astype(str).to_numpy()
    classes = sorted(np.unique(labels).tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.asarray([c2i[x] for x in labels], dtype=np.int64)
    cv_info = run_grouped_cv(features_df, label_col=label_col, feature_cols=feature_cols, cv_mode=cv_mode, group_col="group_day", seed=seed)
    models = _build_models(seed=seed)
    metric_rows, pred_rows = [], []
    confusion_store = {}

    if cv_info["status"] != "ok":
        for model_name in models.keys():
            metric_rows.append(
                {
                    "model": model_name,
                    "label_col": label_col,
                    "status": "skipped",
                    "reason": cv_info.get("reason", "no_valid_folds"),
                    "cv_mode": cv_mode,
                    "split_strategy": cv_info.get("split_strategy", "none"),
                    "n_folds": int(cv_info.get("n_folds", 0)),
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "macro_recall": np.nan,
                }
            )
        return {
            "metrics_df": pd.DataFrame(metric_rows),
            "predictions_df": pd.DataFrame(pred_rows),
            "error_df": pd.DataFrame(pred_rows),
            "feature_importance_df": _compute_feature_importance(X, y, feature_cols, seed=seed),
            "summary": {
                "label_col": label_col,
                "cv_mode": cv_mode,
                "split_strategy": cv_info.get("split_strategy", "none"),
                "reason": cv_info.get("reason", "no_valid_folds"),
                "n_folds": int(cv_info.get("n_folds", 0)),
                "classes": classes,
                "confusion_matrix": {},
            },
        }

    for model_name, estimator in models.items():
        y_true_all, y_pred_all = [], []
        for fold_idx, fold in enumerate(cv_info["splits"]):
            tr = fold["train_idx"]
            te = fold["test_idx"]
            m = clone(estimator)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(pred.tolist())
            for local_i, sample_idx in enumerate(te.tolist()):
                true_id = int(y[sample_idx])
                pred_id = int(pred[local_i])
                pred_rows.append(
                    {
                        "model": model_name,
                        "label_col": label_col,
                        "fold_idx": int(fold_idx),
                        "fold_name": str(fold["fold_name"]),
                        "zip_file": str(features_df.iloc[sample_idx]["zip_file"]),
                        "group_day": str(features_df.iloc[sample_idx]["group_day"]),
                        "true_label": classes[true_id],
                        "pred_label": classes[pred_id],
                        "is_error": int(true_id != pred_id),
                    }
                )
        yt = np.asarray(y_true_all, dtype=np.int64)
        yp = np.asarray(y_pred_all, dtype=np.int64)
        if yt.size == 0:
            metric_rows.append(
                {
                    "model": model_name,
                    "label_col": label_col,
                    "status": "skipped",
                    "reason": "empty_predictions",
                    "cv_mode": cv_mode,
                    "split_strategy": cv_info["split_strategy"],
                    "n_folds": int(cv_info["n_folds"]),
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "macro_recall": np.nan,
                }
            )
            continue
        row = {
            "model": model_name,
            "label_col": label_col,
            "status": "ok",
            "reason": "",
            "cv_mode": cv_mode,
            "split_strategy": cv_info["split_strategy"],
            "n_folds": int(cv_info["n_folds"]),
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        }
        recs = recall_score(yt, yp, labels=list(range(len(classes))), average=None, zero_division=0)
        for class_idx, (c_name, r) in enumerate(zip(classes, recs.tolist())):
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(c_name)).strip("_")
            if not safe:
                safe = f"class_{class_idx}"
            row[f"recall_{safe}"] = float(r)
        metric_rows.append(row)
        cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
        confusion_store[model_name] = {"labels": classes, "matrix": cm.astype(int).tolist()}

    pred_df = pd.DataFrame(pred_rows)
    error_df = pred_df[pred_df["is_error"] == 1].copy() if not pred_df.empty else pd.DataFrame(columns=pred_df.columns)
    return {
        "metrics_df": pd.DataFrame(metric_rows),
        "predictions_df": pred_df,
        "error_df": error_df,
        "feature_importance_df": _compute_feature_importance(X, y, feature_cols, seed=seed),
        "summary": {
            "label_col": label_col,
            "cv_mode": cv_mode,
            "split_strategy": cv_info["split_strategy"],
            "n_folds": int(cv_info["n_folds"]),
            "classes": classes,
            "feature_cols": feature_cols,
            "confusion_matrix": confusion_store,
        },
    }


def _best_cluster_row(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    valid = [r for r in rows if np.isfinite(float(r.get("silhouette", np.nan)))]
    if not valid:
        return rows[0]
    valid.sort(key=lambda x: float(x["silhouette"]), reverse=True)
    return valid[0]


def build_health_report(
    features_df: pd.DataFrame,
    metrics_7class: Dict[str, object] = None,
    metrics_4class: Dict[str, object] = None,
    rule_df: pd.DataFrame = None,
    classical_metrics_df: pd.DataFrame = None,
    grouped_cv_summary: Dict[str, object] = None,
) -> Dict[str, object]:
    metrics_7class = metrics_7class or {}
    metrics_4class = metrics_4class or {}
    rule_df = rule_df if rule_df is not None else pd.DataFrame()
    classical_metrics_df = classical_metrics_df if classical_metrics_df is not None else pd.DataFrame()
    grouped_cv_summary = grouped_cv_summary or {}

    rows7 = (list(metrics_7class.get("kmeans", [])) + list(metrics_7class.get("agglomerative", []))) if metrics_7class.get("status") == "ok" else []
    rows4 = (list(metrics_4class.get("kmeans", [])) + list(metrics_4class.get("agglomerative", []))) if metrics_4class.get("status") == "ok" else []
    best7 = _best_cluster_row(rows7)
    best4 = _best_cluster_row(rows4)
    sil_values = [best7.get("silhouette", np.nan), best4.get("silhouette", np.nan)]
    best_sil = float(np.nanmax(sil_values)) if any(np.isfinite(x) for x in sil_values) else float("nan")
    if not np.isfinite(best_sil):
        grade = "unknown"
    elif best_sil >= 0.60:
        grade = "strong"
    elif best_sil >= 0.35:
        grade = "moderate"
    else:
        grade = "weak"

    has_rule = "is_correct" in rule_df.columns and rule_df["is_correct"].notna().any()
    rule_acc = float(rule_df["is_correct"].mean()) if has_rule else float("nan")
    nearest = list(metrics_7class.get("nearest_pairs", []))[:3] if metrics_7class.get("status") == "ok" else list(metrics_4class.get("nearest_pairs", []))[:3]
    if not classical_metrics_df.empty and "status" in classical_metrics_df.columns:
        ok_df = classical_metrics_df[classical_metrics_df["status"] == "ok"].copy()
        best_classical = ok_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0].to_dict() if not ok_df.empty else {}
    else:
        best_classical = {}

    return {
        "task": "health_signal_non_dl_scan",
        "n_files": int(features_df.shape[0]),
        "labels": sorted(features_df["label"].astype(str).unique().tolist()) if "label" in features_df.columns else [],
        "coarse_labels": sorted(features_df["coarse_label"].astype(str).unique().tolist()) if "coarse_label" in features_df.columns else [],
        "separability_grade": grade,
        "unsupervised_best": {"class7_best": best7, "class4_best": best4},
        "closest_pairs": nearest,
        "rule_based": {
            "accuracy_vs_coarse_label": rule_acc,
            "pred_counts": {str(k): int(v) for k, v in rule_df["pred_label"].value_counts().items()} if "pred_label" in rule_df.columns else {},
            "thresholds": dict(rule_df.attrs.get("thresholds", {})),
        },
        "classical_models": {"best_run": best_classical, "grouped_cv_summary": grouped_cv_summary},
        "recommendations": [
            "Collect >=8 files per coarse class for stable 4-class model.",
            "Keep normal/nasal/exercise around 60s and cough around 30s.",
            "Collect across multiple days and randomize collection order.",
        ],
    }


def metrics_to_distance_frame(metrics: Dict[str, object]) -> pd.DataFrame:
    if not isinstance(metrics, dict) or metrics.get("status") != "ok":
        return pd.DataFrame()
    zips = [str(x) for x in metrics.get("zip_files", [])]
    mat = np.asarray(metrics.get("pairwise_distance", []), dtype=np.float64)
    if mat.size == 0 or not zips:
        return pd.DataFrame()
    return pd.DataFrame(mat, index=zips, columns=zips)


def plot_pca_scatter(metrics: Dict[str, object], out_png) -> Path:
    out_png = Path(out_png)
    if metrics.get("status") != "ok":
        raise ValueError("metrics status is not ok")
    points = pd.DataFrame(metrics.get("pca_points", []))
    if points.empty:
        raise ValueError("No PCA points to plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = points["label"].astype(str).unique().tolist()
    label_alias = {lab: f"L{idx + 1}" for idx, lab in enumerate(labels)}
    sample_alias = {z: f"S{idx + 1}" for idx, z in enumerate(points["zip_file"].astype(str).tolist())}
    cmap = plt.colormaps.get_cmap("tab10")
    for idx, lab in enumerate(labels):
        sub = points[points["label"] == lab]
        ax.scatter(sub["pc1"], sub["pc2"], s=55, color=cmap(idx % 10), label=label_alias[lab])
        for _, r in sub.iterrows():
            ax.text(float(r["pc1"]) + 0.02, float(r["pc2"]) + 0.02, sample_alias[str(r["zip_file"])], fontsize=8)
    ax.set_title(f"PCA Scatter ({metrics.get('label_col', 'label')})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best", title="Label Alias")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_distance_heatmap(distance_df: pd.DataFrame, out_png) -> Path:
    out_png = Path(out_png)
    if distance_df.empty:
        raise ValueError("Distance matrix is empty")
    aliases = [f"S{i + 1}" for i in range(distance_df.shape[0])]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(distance_df.to_numpy(dtype=np.float64), cmap="viridis")
    ax.set_xticks(np.arange(distance_df.shape[1]))
    ax.set_yticks(np.arange(distance_df.shape[0]))
    ax.set_xticklabels(aliases, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(aliases, fontsize=8)
    ax.set_title("Pairwise Distance Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_feature_boxplots(features_df: pd.DataFrame, out_png) -> Path:
    out_png = Path(out_png)
    if "coarse_label" not in features_df.columns:
        raise ValueError("coarse_label missing from features")
    cols = ["duration_s", "ch1_std", "ch2_std", "ch1_bandpower_ratio", "ch2_bandpower_ratio", "corr_ch1_ch2"]
    cols = [c for c in cols if c in features_df.columns]
    if not cols:
        raise ValueError("No feature columns available for boxplot")
    groups = sorted(features_df["coarse_label"].astype(str).unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        ax = axes[i]
        box = [features_df.loc[features_df["coarse_label"] == g, col].to_numpy(dtype=np.float64) for g in groups]
        ax.boxplot(box, labels=groups, showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    for i in range(len(cols), 6):
        axes[i].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
