import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_zip import load_signal_from_zip
from src.preprocess import segment_signal


def _write_signal_zip(zip_path: Path, duration_s: float, dt_s: float = 0.1) -> None:
    t = np.arange(0.0, duration_s + 1e-9, dt_s, dtype=np.float64)
    ch1 = 0.6 * np.sin(2.0 * np.pi * 0.25 * t) + 0.2 * np.sin(2.0 * np.pi * 0.05 * t)
    ch2 = 0.5 * np.cos(2.0 * np.pi * 0.22 * t) + 0.15 * np.cos(2.0 * np.pi * 0.04 * t)
    lines = [",Ch1,Ch2", "Time (s),dR/R0 (%),dR/R0 (%)"]
    for idx in range(t.size):
        lines.append(f"{t[idx]:.2f},{ch1[idx]:.6f},{ch2[idx]:.6f}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("signal.csv", payload)


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="preprocess_test_"))
        cls.zip_help = cls.tmpdir / "AB，HELP，4组-0206191411.zip"
        cls.zip_eat = cls.tmpdir / "AB，EAT，3组-0206191606.zip"
        cls.zip_help_me = cls.tmpdir / "AB，HELP ME，6组-0205183330.zip"
        _write_signal_zip(cls.zip_help, duration_s=120.0)
        _write_signal_zip(cls.zip_eat, duration_s=95.0)
        _write_signal_zip(cls.zip_help_me, duration_s=180.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_segment_count_help(self):
        signal = load_signal_from_zip(self.zip_help)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 4)

    def test_segment_count_eat(self):
        signal = load_signal_from_zip(self.zip_eat)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 3)

    def test_segment_count_help_me(self):
        signal = load_signal_from_zip(self.zip_help_me)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 6)

    def test_resample_shape(self):
        signal = load_signal_from_zip(self.zip_help_me)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertGreater(len(segs), 0)
        self.assertEqual(segs[0].x.shape, (2, 300))


if __name__ == "__main__":
    unittest.main()
