import math
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_health_scan import run_health_scan
from src.health_analysis import (
    extract_file_features,
    extract_hybrid_features,
    map_coarse_label,
    parse_health_label,
    run_grouped_cv,
    run_rule_based_classifier,
)


def _write_health_zip(zip_path: Path, duration_s: float, amp1: float, amp2: float, freq_hz: float, burst: bool = False):
    t = np.arange(0.0, duration_s + 1e-6, 0.1, dtype=np.float64)
    ch1 = amp1 * np.sin(2.0 * np.pi * freq_hz * t)
    ch2 = amp2 * np.cos(2.0 * np.pi * freq_hz * t)
    if burst:
        pulse = np.zeros_like(t)
        pulse[(t > 4.0) & (t < 5.0)] = amp1 * 2.5
        pulse[(t > 12.0) & (t < 13.0)] = -amp1 * 2.0
        ch1 = ch1 + pulse
        ch2 = ch2 + 0.8 * pulse
    lines = [",Ch1,Ch2", "Time (s),dR/R0 (%),dR/R0 (%)"]
    for idx in range(t.size):
        lines.append(f"{t[idx]:.2f},{ch1[idx]:.6f},{ch2[idx]:.6f}")
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("signal.csv", blob)


class TestHealthAnalysis(unittest.TestCase):
    def test_parse_label_and_coarse_mapping(self):
        name = "AB\uff0c\u53e3\u547c\u5438\uff0c\u4e00\u5206\u949f-xxx.zip"
        label = parse_health_label(name)
        self.assertEqual(label, "\u53e3\u547c\u5438")
        self.assertEqual(map_coarse_label(label), "normal")
        with self.assertRaises(ValueError):
            parse_health_label("bad_name.zip")
        with self.assertRaises(ValueError):
            map_coarse_label("\u672a\u77e5\u573a\u666f")

    def test_extract_file_features_stability_and_periodic_short_signal(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="health_feat_"))
        try:
            zip_path = tmpdir / "AB\uff0c\u5de6\u9f3b\u585e\uff0c\u6d4b\u8bd5-0205120000.zip"
            _write_health_zip(zip_path, duration_s=8.0, amp1=0.4, amp2=0.3, freq_hz=0.22)
            feat = extract_file_features(zip_path, segment_window_sec=0.0, feature_set="freq_time_hybrid", window_sec_for_stats=5.0)
            need_cols = [
                "zip_file",
                "label",
                "coarse_label",
                "duration_s",
                "fs_hz",
                "ch1_bandpower_ratio",
                "ch2_bandpower_ratio",
                "ch1_peak_interval_mean_s",
                "ch2_peak_interval_mean_s",
                "corr_ch1_ch2",
                "energy_ratio_ch1_ch2",
            ]
            for col in need_cols:
                self.assertIn(col, feat)
                if isinstance(feat[col], float):
                    self.assertTrue(math.isfinite(feat[col]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_feature_set_switch(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="health_feat_set_"))
        try:
            zip_path = tmpdir / "AB\uff0c\u8dd1\u6b65\uff0c\u6d4b\u8bd5-0206120000.zip"
            _write_health_zip(zip_path, duration_s=61.0, amp1=1.4, amp2=1.2, freq_hz=0.35)
            freq_only = extract_hybrid_features(zip_path, feature_set="freq_only", window_sec_for_stats=10.0)
            hybrid = extract_hybrid_features(zip_path, feature_set="freq_time_hybrid", window_sec_for_stats=10.0)
            self.assertEqual(freq_only["feature_set"], "freq_only")
            self.assertEqual(hybrid["feature_set"], "freq_time_hybrid")
            self.assertIn("ch1_std", hybrid)
            self.assertNotIn("ch1_std", freq_only)
            self.assertIn("ch1_bandpower_ratio", freq_only)
            self.assertIn("ch1_bandpower_ratio", hybrid)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_grouped_cv_group_day(self):
        df = pd.DataFrame(
            {
                "zip_file": ["a", "b", "c", "d", "e", "f"],
                "label": ["x", "y", "x", "y", "x", "y"],
                "coarse_label": ["x", "y", "x", "y", "x", "y"],
                "group_day": ["0205", "0205", "0206", "0206", "0207", "0207"],
                "source_format": ["csv", "csv", "csv", "csv", "csv", "csv"],
                "f1": [0.1, 0.8, 0.2, 0.9, 0.3, 1.0],
                "f2": [0.0, 0.9, 0.1, 1.0, 0.2, 0.95],
            }
        )
        cv = run_grouped_cv(df, label_col="coarse_label", cv_mode="group_by_day", group_col="group_day", seed=42)
        self.assertEqual(cv["status"], "ok")
        self.assertEqual(cv["split_strategy"], "group_by_day")
        self.assertGreaterEqual(int(cv["n_folds"]), 2)

    def test_rule_classifier_outputs_required_columns(self):
        rows = [
            {
                "zip_file": "a.zip",
                "label": "\u53e3\u547c\u5438",
                "coarse_label": "normal",
                "duration_s": 60.0,
                "ch1_std": 3.2,
                "ch2_std": 2.8,
                "ch1_iqr": 4.1,
                "ch2_iqr": 3.9,
                "ch1_hf_ratio": 0.01,
                "ch2_hf_ratio": 0.01,
                "ch1_dominant_freq": 0.20,
                "ch2_dominant_freq": 0.20,
                "ch1_kurtosis": 0.1,
                "ch2_kurtosis": 0.2,
                "ch1_zero_crossing_rate": 0.05,
                "ch2_zero_crossing_rate": 0.05,
                "ch1_breaths_per_min": 12.0,
                "ch2_breaths_per_min": 12.0,
            },
            {
                "zip_file": "b.zip",
                "label": "\u54b3\u55fd",
                "coarse_label": "cough",
                "duration_s": 24.0,
                "ch1_std": 2.9,
                "ch2_std": 2.5,
                "ch1_iqr": 3.5,
                "ch2_iqr": 3.1,
                "ch1_hf_ratio": 0.08,
                "ch2_hf_ratio": 0.07,
                "ch1_dominant_freq": 0.9,
                "ch2_dominant_freq": 0.8,
                "ch1_kurtosis": 2.3,
                "ch2_kurtosis": 2.1,
                "ch1_zero_crossing_rate": 0.16,
                "ch2_zero_crossing_rate": 0.14,
                "ch1_breaths_per_min": 30.0,
                "ch2_breaths_per_min": 28.0,
            },
        ]
        pred_df = run_rule_based_classifier(pd.DataFrame(rows))
        for col in ["pred_label", "trigger_rule", "key_feature_values"]:
            self.assertIn(col, pred_df.columns)
        self.assertIn("thresholds", pred_df.attrs)

    def test_run_health_scan_smoke(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="health_scan_"))
        try:
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "out"
            input_dir.mkdir(parents=True, exist_ok=True)

            _write_health_zip(input_dir / "AB\uff0c\u53e3\u547c\u5438\uff0c\u4e00\u5206\u949f-0205120001.zip", 61.0, 2.8, 2.2, 0.22)
            _write_health_zip(input_dir / "AB\uff0c\u5de6\u9f3b\u585e\uff0c1\u5206\u949f-0205120002.zip", 62.0, 0.7, 0.6, 0.21)
            _write_health_zip(input_dir / "AB\uff0c\u5c0f\u8dd1\uff0c1\u5206\u949f-0206120003.zip", 61.0, 3.2, 3.0, 0.75)
            _write_health_zip(input_dir / "AB\uff0c\u54b3\u55fd\uff0c5\u6b21-0206120004.zip", 24.0, 2.0, 1.8, 0.95, burst=True)
            _write_health_zip(input_dir / "bad_name.zip", 30.0, 1.0, 1.0, 0.2)

            result = run_health_scan(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=False,
                segment_window_sec=0.0,
                feature_set="freq_time_hybrid",
                window_sec_for_stats=10.0,
                eval_mode="both",
                cv_mode="group_by_day",
                seed=42,
                save_plots=True,
            )

            out_path = Path(result["output_dir"])
            must_files = [
                "file_features.csv",
                "pairwise_distance.csv",
                "unsupervised_7class_metrics.json",
                "unsupervised_4class_metrics.json",
                "rule_predictions.csv",
                "rule_thresholds.json",
                "classical_model_metrics.csv",
                "grouped_cv_summary.json",
                "feature_importance.csv",
                "error_cases.csv",
                "health_summary.json",
                "health_brief.md",
                "skipped_files.csv",
            ]
            for name in must_files:
                self.assertTrue((out_path / name).exists(), name)

            fig_dir = out_path / "figures"
            for name in ["pca_7class.png", "pca_4class.png", "distance_heatmap.png", "feature_boxplots.png"]:
                self.assertTrue((fig_dir / name).exists(), name)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
