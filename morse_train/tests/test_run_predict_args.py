from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_predict import validate_input_args


class TestRunPredictArgs(unittest.TestCase):
    def test_accept_input_dir_only(self):
        validate_input_args("d:/some_dir", None)

    def test_accept_input_file_only(self):
        validate_input_args(None, "d:/some_file.zip")

    def test_reject_both_missing(self):
        with self.assertRaises(ValueError):
            validate_input_args(None, None)

    def test_reject_both_provided(self):
        with self.assertRaises(ValueError):
            validate_input_args("d:/a", "d:/b.zip")


if __name__ == "__main__":
    unittest.main()
