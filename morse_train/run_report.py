import argparse
import base64
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.decoder import normalize_phrase, phrase_letters
from src.utils import load_json
from src.visualize import plot_confusion_matrix, plot_prediction_directory, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Build delivery report.html from final artifacts.")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Training artifacts directory")
    parser.add_argument("--prediction_dir", type=str, required=True, help="Prediction output directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Input zip directory for plotting")
    parser.add_argument(
        "--asset_mode",
        type=str,
        choices=["inline", "external"],
        default="inline",
        help="Asset reference mode: inline embeds image/csv as data URI; external uses relative paths.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default=None,
        help="Output report html path, defaults to <artifacts_dir>/report_standalone.html",
    )
    return parser.parse_args()


def _extract_gt_phrase(zip_name: str) -> str:
    text = str(zip_name)
    if not text.upper().startswith("AB"):
        return ""
    seps = [i for i, ch in enumerate(text) if ch in {",", "\uFF0C"}]
    if len(seps) >= 2:
        candidate = text[seps[0] + 1 : seps[1]].strip()
        return normalize_phrase(candidate)
    return ""


def _letters_only(text: str) -> str:
    import re

    return "".join(re.findall(r"[A-Z]", str(text).upper()))


def _edit_distance(a: str, b: str) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1, dtype=np.int32)
    dp[0, :] = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def _char_score(gt_letters: str, pred_letters: str) -> float:
    denom = max(len(gt_letters), len(pred_letters), 1)
    dist = _edit_distance(gt_letters, pred_letters)
    return float(max(0.0, 1.0 - dist / denom))


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace("\\", "/")


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".csv":
        return "text/csv"
    return "application/octet-stream"


def _file_to_data_uri(path: Path) -> Tuple[str, int]:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = _mime_type(path)
    return f"data:{mime};base64,{encoded}", len(data)


class AssetResolver:
    def __init__(self, asset_mode: str, base_dir: Path):
        self.asset_mode = str(asset_mode)
        self.base_dir = Path(base_dir)
        self._embedded_bytes = 0
        self._embedded_files = set()

    @property
    def embedded_bytes(self) -> int:
        return int(self._embedded_bytes)

    @property
    def embedded_size_mb(self) -> float:
        return float(self._embedded_bytes) / (1024.0 * 1024.0)

    def _mark_embedded(self, path: Path, size: int) -> None:
        key = str(path.resolve())
        if key not in self._embedded_files:
            self._embedded_files.add(key)
            self._embedded_bytes += int(size)

    def img_src(self, path: Path) -> Optional[str]:
        path = Path(path)
        if not path.exists():
            return None
        if self.asset_mode == "inline":
            uri, size = _file_to_data_uri(path)
            self._mark_embedded(path, size)
            return uri
        return _rel(path, self.base_dir)

    def csv_href(self, path: Path) -> Optional[str]:
        path = Path(path)
        if not path.exists():
            return None
        if self.asset_mode == "inline":
            uri, size = _file_to_data_uri(path)
            self._mark_embedded(path, size)
            return uri
        return _rel(path, self.base_dir)


def _build_summary_with_gt(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        zip_file = str(row.get("zip_file", ""))
        gt_phrase = _extract_gt_phrase(zip_file)
        pred_phrase = normalize_phrase(str(row.get("decoded_phrase", "")))
        gt_letters = phrase_letters(gt_phrase)
        pred_letters = _letters_only(str(row.get("final_sequence", "")))
        exact = int(gt_letters == pred_letters and gt_letters != "")
        char_acc = _char_score(gt_letters, pred_letters) if gt_letters else np.nan
        rows.append(
            {
                "zip_file": zip_file,
                "gt_phrase": gt_phrase,
                "decoded_phrase": pred_phrase,
                "raw_sequence": str(row.get("raw_sequence", "")),
                "final_sequence": str(row.get("final_sequence", "")),
                "segments": int(row.get("segments", 0)),
                "offset_sec": float(row.get("offset_sec", 0.0)),
                "mean_confidence": float(row.get("mean_confidence", 0.0)),
                "decode_score": float(row.get("decode_score", 0.0)),
                "exact_match": exact,
                "char_accuracy": float(char_acc) if not np.isnan(char_acc) else np.nan,
                "output_csv": str(row.get("output_csv", "")),
            }
        )
    return pd.DataFrame(rows)


def _resolve_output_csv(output_csv_str: str, prediction_dir: Path) -> Path:
    p = Path(str(output_csv_str))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p

    c1 = prediction_dir / p.name
    if c1.exists():
        return c1

    c2 = prediction_dir / p
    if c2.exists():
        return c2
    return c1


def _render_image_block(path: Path, alt: str, resolver: AssetResolver) -> str:
    src = resolver.img_src(path)
    if src is None:
        return f"<div class=\"missing\">missing image: {html.escape(str(path))}</div>"
    return f"<img src=\"{html.escape(src)}\" alt=\"{html.escape(alt)}\" />"


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    prediction_dir = Path(args.prediction_dir)
    input_dir = Path(args.input_dir)
    output_html = Path(args.output_html) if args.output_html else artifacts_dir / "report_standalone.html"

    history_path = artifacts_dir / "train_history.csv"
    metrics_path = artifacts_dir / "metrics.json"
    config_path = artifacts_dir / "train_config.json"
    cm_path = artifacts_dir / "confusion_matrix.csv"
    summary_path = prediction_dir / "decoding_summary.csv"

    train_fig_dir = artifacts_dir / "figures"
    pred_fig_dir = prediction_dir / "figures"
    train_figs = plot_training_history(history_path, train_fig_dir)
    cm_figs = plot_confusion_matrix(
        cm_path,
        train_fig_dir / "confusion_matrix.png",
        annotate_nonzero=True,
        show_normalized=True,
    )
    pred_figs = plot_prediction_directory(input_dir=input_dir, prediction_dir=prediction_dir, out_dir=pred_fig_dir)

    metrics = load_json(metrics_path)
    config = load_json(config_path)
    history_df = pd.read_csv(history_path)
    summary_df = pd.read_csv(summary_path)
    summary_gt_df = _build_summary_with_gt(summary_df)

    exact_match = float(summary_gt_df["exact_match"].mean()) if not summary_gt_df.empty else 0.0
    char_accuracy = float(summary_gt_df["char_accuracy"].dropna().mean()) if not summary_gt_df.empty else 0.0

    recall_items = list(metrics.get("per_class_recall", {}).items())
    recall_sorted = sorted(recall_items, key=lambda x: x[1])
    worst_recall = recall_sorted[:8]

    aug_below_val = None
    if "train_acc_aug" in history_df.columns and "val_accuracy" in history_df.columns:
        aug_below_val = int((history_df["train_acc_aug"] < history_df["val_accuracy"]).sum())
    elif "train_accuracy" in history_df.columns and "val_accuracy" in history_df.columns:
        aug_below_val = int((history_df["train_accuracy"] < history_df["val_accuracy"]).sum())

    resolver = AssetResolver(args.asset_mode, output_html.parent)
    config_html = html.escape(pd.Series(config).to_string())

    show_cols = [
        "zip_file",
        "gt_phrase",
        "decoded_phrase",
        "raw_sequence",
        "final_sequence",
        "segments",
        "offset_sec",
        "mean_confidence",
        "exact_match",
        "char_accuracy",
    ]
    table_html = summary_gt_df[show_cols].to_html(index=False, float_format=lambda x: f"{x:.4f}")

    pred_cards = []
    by_stem = {Path(p).stem.replace("_prediction", ""): Path(p) for p in pred_figs}
    for _, row in summary_gt_df.iterrows():
        stem = Path(str(row["zip_file"])).stem
        img = by_stem.get(stem)

        out_csv = _resolve_output_csv(str(row["output_csv"]), prediction_dir)
        csv_href = resolver.csv_href(out_csv)
        if csv_href is None:
            csv_html = f"<span class=\"missing\">missing csv: {html.escape(str(out_csv))}</span>"
        else:
            csv_html = (
                f"<a href=\"{html.escape(csv_href)}\" download=\"{html.escape(out_csv.name)}\">segments.csv</a>"
            )

        if img is None:
            img_html = "<div class=\"missing\">missing image: prediction figure not found</div>"
        else:
            img_html = _render_image_block(img, stem, resolver)

        pred_cards.append(
            f"""
            <div class=\"card\">
              <div><b>{html.escape(str(row['zip_file']))}</b></div>
              <div>GT: {html.escape(str(row['gt_phrase']))} | Pred: {html.escape(str(row['decoded_phrase']))}</div>
              <div>Seq: {html.escape(str(row['final_sequence']))} | mean_conf={float(row['mean_confidence']):.3f}</div>
              <div>{csv_html}</div>
              {img_html}
            </div>
            """
        )

    cm_imgs_html = "".join([f"<div>{_render_image_block(Path(p), 'confusion', resolver)}</div>" for p in cm_figs])

    train_imgs_html = "".join([f"<div>{_render_image_block(Path(p), 'training', resolver)}</div>" for p in train_figs])

    explain_text = (
        "训练增强口径(train_aug)长期低于验证口径(val)在当前配置下是可解释的："
        "训练时启用了mixup、label smoothing和随机增强，损失与准确率统计更难；"
        "验证集未做这些扰动，指标通常更高。"
    )
    if aug_below_val is not None:
        explain_text += f" 本次共有 {aug_below_val}/{len(history_df)} 轮出现 train_aug_acc < val_acc。"

    splits = metrics.get("splits", {})
    train_segments = int(splits.get("train", 0))
    val_segments = int(splits.get("val", 0))
    test_segments = int(splits.get("test", 0))

    html_text = f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <title>Morse Delivery Report</title>
  <style>
    body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 24px; line-height: 1.45; color: #1f2937; }}
    h1, h2 {{ margin: 10px 0; }}
    .meta {{ color: #4b5563; font-size: 13px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 14px 0 20px; }}
    .kpi {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; background: #f8fafc; }}
    .kpi .v {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    .section {{ margin-top: 24px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; margin: 6px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f8fafc; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; overflow: auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; background: #fff; }}
    .missing {{ color: #b91c1c; font-size: 12px; border: 1px dashed #ef4444; padding: 4px 6px; border-radius: 4px; display: inline-block; }}
  </style>
</head>
<body>
  <h1>Morse Delivery Report</h1>
  <div class=\"meta\">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  <div class=\"meta\">Artifacts: {html.escape(str(artifacts_dir))} | Predictions: {html.escape(str(prediction_dir))}</div>
  <div class=\"meta\">asset_mode: {html.escape(args.asset_mode)} | embedded_size_mb: {resolver.embedded_size_mb:.2f}</div>

  <div class=\"kpis\">
    <div class=\"kpi\"><div>Best Epoch</div><div class=\"v\">{int(metrics.get('best_epoch', -1))}</div></div>
    <div class=\"kpi\"><div>Best Val Macro-F1</div><div class=\"v\">{float(metrics.get('best_val_macro_f1', 0.0)):.4f}</div></div>
    <div class=\"kpi\"><div>Test Accuracy</div><div class=\"v\">{float(metrics.get('test_accuracy', 0.0)):.4f}</div></div>
    <div class=\"kpi\"><div>Test Macro-F1</div><div class=\"v\">{float(metrics.get('test_macro_f1', 0.0)):.4f}</div></div>
    <div class=\"kpi\"><div>Command Exact</div><div class=\"v\">{exact_match:.4f}</div></div>
    <div class=\"kpi\"><div>Command Char Accuracy</div><div class=\"v\">{char_accuracy:.4f}</div></div>
    <div class=\"kpi\"><div>Train Segments</div><div class=\"v\">{train_segments}</div></div>
    <div class=\"kpi\"><div>Val/Test Segments</div><div class=\"v\">{val_segments}/{test_segments}</div></div>
  </div>

  <div class=\"section\">
    <h2>Phenomenon Explanation</h2>
    <p>{html.escape(explain_text)}</p>
  </div>

  <div class=\"section\">
    <h2>Training Curves</h2>
    {train_imgs_html}
  </div>

  <div class=\"section\">
    <h2>Confusion Matrix</h2>
    {cm_imgs_html}
  </div>

  <div class=\"section\">
    <h2>Command-Level Summary</h2>
    {table_html}
  </div>

  <div class=\"section\">
    <h2>Prediction Visualization</h2>
    <div class=\"cards\">
      {''.join(pred_cards)}
    </div>
  </div>

  <div class=\"section\">
    <h2>Worst Per-Class Recall</h2>
    <pre>{html.escape(pd.DataFrame(worst_recall, columns=['label', 'recall']).to_string(index=False))}</pre>
  </div>

  <div class=\"section\">
    <h2>Train Config Snapshot</h2>
    <pre>{config_html}</pre>
  </div>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print("report_html:", output_html)
    print("asset_mode:", args.asset_mode)
    print("embedded_size_mb:", f"{resolver.embedded_size_mb:.2f}")


if __name__ == "__main__":
    main()
