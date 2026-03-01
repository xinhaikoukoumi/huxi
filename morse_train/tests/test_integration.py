import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_directory
from src.train import train_pipeline


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


def _find_digit_dirs():
    root = Path(__file__).resolve().parents[2]
    digit_root = None
    digit_train = None
    target_file = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        sub_train = None
        for x in d.iterdir():
            if x.is_dir() and x.name.startswith("AD+BC") and any(x.glob("*.zip")):
                sub_train = x
                break
        if sub_train is None:
            continue

        target = None
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() == ".zip" and "0-9" in f.name:
                target = f
                break
        if target is None:
            continue

        digit_root = d
        digit_train = sub_train
        target_file = target
        break

    if digit_root is None or digit_train is None or target_file is None:
        raise RuntimeError("Failed to locate digit train directory and 0-9 target file")
    return digit_train, digit_root, target_file


@unittest.skipUnless(os.environ.get("MORSE_RUN_SLOW_TESTS") == "1", "Set MORSE_RUN_SLOW_TESTS=1 to run")
class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.train_dir, self.pred_dir = _find_dirs()
        self.digit_train_dir, self.digit_pred_dir, self.digit_target_file = _find_digit_dirs()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="morse_train_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_smoke_train_and_predict(self):
        artifacts = self.tmpdir / "artifacts"
        preds = self.tmpdir / "predictions"

        result = train_pipeline(
            train_dir=self.train_dir,
            out_dir=artifacts,
            seed=42,
            max_epochs=1,
            batch_size=128,
            rebuild_cache=True,
        )
        model_path = Path(result["artifacts"]["model_path"])
        label_map_path = Path(result["artifacts"]["label_map_path"])
        self.assertTrue(model_path.exists())
        self.assertTrue(label_map_path.exists())

        predict_directory(
            model_path=model_path,
            label_map_path=label_map_path,
            input_dir=self.pred_dir,
            output_dir=preds,
            batch_size=128,
        )

        hit = None
        for csv_path in preds.glob("*0205183330*_segments.csv"):
            hit = csv_path
            break
        self.assertIsNotNone(hit)
        df = pd.read_csv(hit)
        self.assertEqual(len(df), 6)

    def test_smoke_single_channel_train_and_predict(self):
        artifacts = self.tmpdir / "artifacts_ch1"
        preds = self.tmpdir / "predictions_ch1"

        result = train_pipeline(
            train_dir=self.train_dir,
            out_dir=artifacts,
            seed=42,
            channel_mode="ch1",
            max_epochs=1,
            batch_size=128,
            rebuild_cache=True,
        )
        model_path = Path(result["artifacts"]["model_path"])
        label_map_path = Path(result["artifacts"]["label_map_path"])
        ckpt = torch.load(model_path, map_location="cpu")
        self.assertEqual(int(ckpt.get("input_channels", -1)), 1)
        self.assertEqual(str(ckpt.get("config", {}).get("channel_mode")), "ch1")

        predict_directory(
            model_path=model_path,
            label_map_path=label_map_path,
            input_dir=self.pred_dir,
            output_dir=preds,
            batch_size=128,
        )

        hit = None
        for csv_path in preds.glob("*0205183330*_segments.csv"):
            hit = csv_path
            break
        self.assertIsNotNone(hit)
        df = pd.read_csv(hit)
        self.assertEqual(len(df), 6)

    def test_smoke_compare_channels_script(self):
        out_root = self.tmpdir / "channel_compare"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_compare_channels.py"),
            "--train_dir",
            str(self.train_dir),
            "--out_root",
            str(out_root),
            "--reference_config",
            str(PROJECT_ROOT / "artifacts_delivery_final" / "train_config.json"),
            "--max_epochs",
            "1",
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

        metrics_path = out_root / "comparison_metrics.csv"
        self.assertTrue(metrics_path.exists())
        df = pd.read_csv(metrics_path)
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["channel_mode"].tolist()), {"dual", "ch1", "ch2"})
        dual_row = df[df["channel_mode"] == "dual"].iloc[0]
        self.assertAlmostEqual(float(dual_row["delta_accuracy_vs_dual"]), 0.0, places=9)
        self.assertAlmostEqual(float(dual_row["delta_macro_f1_vs_dual"]), 0.0, places=9)

    def test_smoke_digits_train_and_single_file_predict(self):
        artifacts = self.tmpdir / "artifacts_digits"
        preds = self.tmpdir / "predictions_digits_single"

        result = train_pipeline(
            train_dir=self.digit_train_dir,
            out_dir=artifacts,
            seed=42,
            label_mode="digits",
            channel_mode="dual",
            max_epochs=1,
            batch_size=128,
            rebuild_cache=True,
        )
        model_path = Path(result["artifacts"]["model_path"])
        label_map_path = Path(result["artifacts"]["label_map_path"])
        self.assertTrue(model_path.exists())
        self.assertTrue(label_map_path.exists())

        predict_directory(
            model_path=model_path,
            label_map_path=label_map_path,
            output_dir=preds,
            input_file=self.digit_target_file,
            batch_size=128,
            use_lexicon_decoder=False,
            auto_lexicon=False,
        )
        summary_path = preds / "decoding_summary.csv"
        self.assertTrue(summary_path.exists())
        df = pd.read_csv(summary_path)
        self.assertEqual(len(df), 1)
        self.assertIn("0-9", str(df.iloc[0]["zip_file"]))

    def test_smoke_run_digit_pipeline_script(self):
        out_root = self.tmpdir / "digit_pipeline"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_digit_pipeline.py"),
            "--train_dir",
            str(self.digit_train_dir),
            "--target_file",
            str(self.digit_target_file),
            "--out_root",
            str(out_root),
            "--batch_size",
            "128",
            "--stage1_max_epochs",
            "1",
            "--stage1_patience",
            "1",
            "--stage1_limit",
            "1",
            "--skip_stage2",
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        self.assertTrue((out_root / "stage1_results.csv").exists())
        self.assertTrue((out_root / "best_run.json").exists())
        self.assertTrue((out_root / "prediction_final" / "decoding_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
