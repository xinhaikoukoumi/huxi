import io
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import build_cache, extract_label_from_filename, select_channel_mode


def _make_sample_zip(zip_path: Path):
    lines = [",Ch1,Ch2", "Time (s),ΔR/R0 (%),ΔR/R0 (%)"]
    for t in range(0, 61):
        ch1 = 0.1 * np.sin(t / 5.0)
        ch2 = 0.1 * np.cos(t / 5.0)
        lines.append(f"{float(t):.1f},{ch1:.6f},{ch2:.6f}")
    csv_blob = ("\n".join(lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.csv", csv_blob)


class TestChannelMode(unittest.TestCase):
    def test_extract_label_mode_letters_and_digits(self):
        self.assertEqual(extract_label_from_filename("AB,A,unit.zip", label_mode="letters"), "A")
        self.assertEqual(extract_label_from_filename("AB，7，unit.zip", label_mode="digits"), "7")
        self.assertIsNone(extract_label_from_filename("AB，7，unit.zip", label_mode="letters"))
        self.assertIsNone(extract_label_from_filename("AB,A,unit.zip", label_mode="digits"))

        with self.assertRaises(ValueError):
            extract_label_from_filename("AB,A,unit.zip", label_mode="bad")

    def test_select_channel_mode(self):
        x = np.asarray([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
        dual = select_channel_mode(x, "dual")
        ch1 = select_channel_mode(x, "ch1")
        ch2 = select_channel_mode(x, "ch2")

        self.assertEqual(dual.shape, (2, 3))
        self.assertEqual(ch1.shape, (1, 3))
        self.assertEqual(ch2.shape, (1, 3))
        self.assertTrue(np.array_equal(ch1[0], x[0]))
        self.assertTrue(np.array_equal(ch2[0], x[1]))

        with self.assertRaises(ValueError):
            select_channel_mode(x, "bad_mode")

    def test_build_cache_channel_dimension(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="morse_channel_mode_"))
        try:
            train_dir = tmpdir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            _make_sample_zip(train_dir / "AB,A,unit.zip")

            dual_out = tmpdir / "dual"
            dual_out.mkdir(parents=True, exist_ok=True)
            dual_bundle = build_cache(
                train_dir=train_dir,
                cache_path=dual_out / "segments_cache.npz",
                meta_path=dual_out / "meta.csv",
                label_map_path=dual_out / "label_map.json",
                window_sec=30.0,
                target_points=64,
                min_valid_ratio=0.6,
                channel_mode="dual",
            )
            self.assertEqual(int(dual_bundle.X.shape[1]), 2)
            self.assertGreater(int(dual_bundle.X.shape[0]), 0)

            ch1_out = tmpdir / "ch1"
            ch1_out.mkdir(parents=True, exist_ok=True)
            ch1_bundle = build_cache(
                train_dir=train_dir,
                cache_path=ch1_out / "segments_cache.npz",
                meta_path=ch1_out / "meta.csv",
                label_map_path=ch1_out / "label_map.json",
                window_sec=30.0,
                target_points=64,
                min_valid_ratio=0.6,
                channel_mode="ch1",
            )
            self.assertEqual(int(ch1_bundle.X.shape[1]), 1)
            self.assertEqual(int(ch1_bundle.X.shape[0]), int(dual_bundle.X.shape[0]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_cache_digits_mode(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="morse_digits_mode_"))
        try:
            train_dir = tmpdir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            _make_sample_zip(train_dir / "AB，7，unit.zip")

            out_dir = tmpdir / "digits"
            out_dir.mkdir(parents=True, exist_ok=True)
            bundle = build_cache(
                train_dir=train_dir,
                cache_path=out_dir / "segments_cache.npz",
                meta_path=out_dir / "meta.csv",
                label_map_path=out_dir / "label_map.json",
                window_sec=30.0,
                target_points=64,
                min_valid_ratio=0.6,
                label_mode="digits",
                channel_mode="dual",
            )
            self.assertEqual(set(bundle.label_map.keys()), {"7"})
            self.assertEqual(int(bundle.X.shape[1]), 2)
            self.assertGreater(int(bundle.X.shape[0]), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
