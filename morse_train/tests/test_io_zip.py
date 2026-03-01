import zipfile
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_zip import load_signal_from_zip


def _find_train_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    for p in root.rglob("AD+BC"):
        if p.is_dir() and any(p.glob("*.zip")):
            return p
    raise RuntimeError("Failed to locate AD+BC training directory")


class TestZipLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dir = _find_train_dir()
        cls.zip_files = sorted(cls.train_dir.glob("*.zip"))
        if not cls.zip_files:
            raise RuntimeError("No training zip files found")

    def test_csv_dual_header_parse(self):
        csv_zip = None
        for zp in self.zip_files:
            with zipfile.ZipFile(zp, "r") as zf:
                names = [n.lower() for n in zf.namelist()]
            if any(n.endswith(".csv") for n in names):
                csv_zip = zp
                break
        self.assertIsNotNone(csv_zip)
        signal = load_signal_from_zip(csv_zip)
        self.assertEqual(signal.source_format, "csv")
        self.assertTrue({"time_s", "ch1", "ch2"}.issubset(signal.frame.columns))
        self.assertGreater(len(signal.frame), 100)

    def test_xlsx_fallback_parse(self):
        xlsx_only = None
        for zp in self.zip_files:
            with zipfile.ZipFile(zp, "r") as zf:
                names = [n.lower() for n in zf.namelist()]
            has_csv = any(n.endswith(".csv") for n in names)
            has_xlsx = any(n.endswith(".xlsx") for n in names)
            if (not has_csv) and has_xlsx:
                xlsx_only = zp
                break
        self.assertIsNotNone(xlsx_only)
        signal = load_signal_from_zip(xlsx_only)
        self.assertEqual(signal.source_format, "xlsx")
        self.assertTrue({"time_s", "ch1", "ch2"}.issubset(signal.frame.columns))
        self.assertGreater(len(signal.frame), 100)


if __name__ == "__main__":
    unittest.main()
