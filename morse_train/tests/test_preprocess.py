from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_zip import load_signal_from_zip
from src.preprocess import segment_signal


def _find_dirs():
    root = Path(__file__).resolve().parents[2]
    train_dir = None
    for p in root.rglob("AD+BC"):
        if p.is_dir() and any(p.glob("*.zip")):
            train_dir = p
            break
    if train_dir is None:
        raise RuntimeError("Failed to locate AD+BC")
    pred_dir = train_dir.parent
    return train_dir, pred_dir


def _find_zip_by_stamp(folder: Path, stamp: str) -> Path:
    for zp in folder.glob("*.zip"):
        if stamp in zp.name:
            return zp
    raise RuntimeError(f"zip not found for stamp={stamp}")


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.pred_dir = _find_dirs()

    def test_segment_count_help(self):
        zip_path = _find_zip_by_stamp(self.pred_dir, "0206191411")
        signal = load_signal_from_zip(zip_path)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 4)

    def test_segment_count_eat(self):
        zip_path = _find_zip_by_stamp(self.pred_dir, "0206191606")
        signal = load_signal_from_zip(zip_path)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 3)

    def test_segment_count_help_me(self):
        zip_path = _find_zip_by_stamp(self.pred_dir, "0205183330")
        signal = load_signal_from_zip(zip_path)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertEqual(len(segs), 6)

    def test_resample_shape(self):
        zip_path = _find_zip_by_stamp(self.pred_dir, "0205183330")
        signal = load_signal_from_zip(zip_path)
        segs = segment_signal(signal.frame, window_sec=30.0, target_points=300, min_valid_ratio=0.6)
        self.assertGreater(len(segs), 0)
        self.assertEqual(segs[0].x.shape, (2, 300))


if __name__ == "__main__":
    unittest.main()
