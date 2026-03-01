from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd

from .io_zip import load_signal_from_zip
from .utils import ensure_dir, log


matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_csv, out_dir) -> List[Path]:
    history_csv = Path(history_csv)
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(history_csv)
    outputs = []

    train_loss_clean_col = "train_loss_clean" if "train_loss_clean" in df.columns else None
    train_loss_aug_col = "train_loss_aug" if "train_loss_aug" in df.columns else None
    if "train_loss" in df.columns and train_loss_aug_col is None:
        train_loss_aug_col = "train_loss"

    fig_loss, ax_loss = plt.subplots(figsize=(8.8, 4.8))
    if train_loss_clean_col:
        ax_loss.plot(df["epoch"], df[train_loss_clean_col], label="train_loss_clean", linewidth=2.0)
    if train_loss_aug_col:
        ax_loss.plot(
            df["epoch"],
            df[train_loss_aug_col],
            label="train_loss_aug",
            linewidth=1.6,
            linestyle="--",
            alpha=0.85,
        )
    ax_loss.plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=2.0)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training/Validation Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    loss_path = out_dir / "loss_curve.png"
    _save_fig(fig_loss, loss_path)
    outputs.append(loss_path)

    train_acc_clean_col = "train_acc_clean" if "train_acc_clean" in df.columns else None
    train_acc_aug_col = "train_acc_aug" if "train_acc_aug" in df.columns else None
    if "train_accuracy" in df.columns and train_acc_aug_col is None:
        train_acc_aug_col = "train_accuracy"

    if (train_acc_clean_col or train_acc_aug_col) and "val_accuracy" in df.columns:
        fig_acc, ax_acc = plt.subplots(figsize=(8.8, 4.8))
        if train_acc_clean_col:
            ax_acc.plot(df["epoch"], df[train_acc_clean_col], label="train_acc_clean", linewidth=2.0)
        if train_acc_aug_col:
            ax_acc.plot(
                df["epoch"],
                df[train_acc_aug_col],
                label="train_acc_aug",
                linewidth=1.6,
                linestyle="--",
                alpha=0.85,
            )
        ax_acc.plot(df["epoch"], df["val_accuracy"], label="val_accuracy", linewidth=2.0)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.set_title("Training/Validation Accuracy")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend()
        acc_path = out_dir / "accuracy_curve.png"
        _save_fig(fig_acc, acc_path)
        outputs.append(acc_path)

    if "val_macro_f1" in df.columns:
        fig_f1, ax_f1 = plt.subplots(figsize=(8.8, 4.8))
        ax_f1.plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1", linewidth=2.0)
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_ylim(0.0, 1.0)
        ax_f1.set_title("Validation Macro-F1")
        ax_f1.grid(alpha=0.3)
        ax_f1.legend()
        f1_path = out_dir / "val_macro_f1_curve.png"
        _save_fig(fig_f1, f1_path)
        outputs.append(f1_path)

    return outputs


def plot_confusion_matrix(
    cm_csv,
    out_path,
    title: str = "Confusion Matrix",
    annotate_nonzero: bool = True,
    show_normalized: bool = True,
) -> List[Path]:
    cm_csv = Path(cm_csv)
    out_path = Path(out_path)
    cm_df = pd.read_csv(cm_csv, index_col=0)
    labels = cm_df.index.astype(str).tolist()
    cm_count = cm_df.to_numpy(dtype=np.float64)
    outputs: List[Path] = []

    def _draw_matrix(values: np.ndarray, out_file: Path, fig_title: str, as_percent: bool):
        fig, ax = plt.subplots(figsize=(12.0, 10.0))
        im = ax.imshow(values, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(fig_title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        if annotate_nonzero:
            thresh = values.max() * 0.5 if values.size else 0.0
            fontsize = 7 if len(labels) > 20 else 8
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if cm_count[i, j] <= 0:
                        continue
                    text_val = f"{values[i, j]:.1f}%" if as_percent else f"{int(round(cm_count[i, j]))}"
                    ax.text(
                        j,
                        i,
                        text_val,
                        ha="center",
                        va="center",
                        color="white" if values[i, j] > thresh else "black",
                        fontsize=fontsize,
                    )

        _save_fig(fig, out_file)
        outputs.append(out_file)

    _draw_matrix(cm_count, out_path, f"{title} (Count)", as_percent=False)

    if show_normalized:
        row_sum = np.sum(cm_count, axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        cm_norm = (cm_count / row_sum) * 100.0
        norm_path = out_path.with_name(f"{out_path.stem}_normalized{out_path.suffix}")
        _draw_matrix(cm_norm, norm_path, f"{title} (Row-Normalized %)", as_percent=True)

    return outputs


def _resolve_output_csv(prediction_dir: Path, csv_path_str: str, zip_file: str) -> Path:
    candidate = Path(csv_path_str)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        maybe = prediction_dir / candidate.name
        if maybe.exists():
            return maybe
    stem = Path(zip_file).stem
    fallback = prediction_dir / f"{stem}_segments.csv"
    return fallback


def _resolve_zip_file(input_dir: Path, zip_name: str) -> Path:
    exact = input_dir / zip_name
    if exact.exists():
        return exact
    stem = Path(zip_name).stem
    matches = list(input_dir.glob(f"{stem}.zip"))
    if matches:
        return matches[0]
    all_zips = list(input_dir.glob("*.zip"))
    for zp in all_zips:
        if zp.stem == stem:
            return zp
    return exact


def plot_prediction_signal(zip_path, segments_csv, out_png, title: str = "") -> Path:
    zip_path = Path(zip_path)
    segments_csv = Path(segments_csv)
    out_png = Path(out_png)

    signal = load_signal_from_zip(zip_path)
    frame = signal.frame.copy()
    frame = frame.sort_values("time_s").reset_index(drop=True)

    seg_df = pd.read_csv(segments_csv)

    t = frame["time_s"].to_numpy(dtype=np.float64)
    ch1 = frame["ch1"].to_numpy(dtype=np.float64)
    ch2 = frame["ch2"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(16.0, 7.2), sharex=True)
    ax1, ax2 = axes
    ax1.plot(t, ch1, color="#2b6cb0", linewidth=1.2, label="Ch1 DeltaR/R0 (%)")
    ax2.plot(t, ch2, color="#2f855a", linewidth=1.2, label="Ch2 DeltaR/R0 (%)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax1.grid(alpha=0.25)
    ax2.grid(alpha=0.25)
    ax1.set_ylabel("Ch1")
    ax2.set_ylabel("Ch2")
    ax2.set_xlabel("Time (s)")

    colors = ["#fed7aa", "#fde68a", "#bbf7d0", "#bfdbfe", "#ddd6fe", "#fecdd3"]
    y_min, y_max = np.nanmin(ch1), np.nanmax(ch1)
    span = max(1e-6, y_max - y_min)

    for idx, row in seg_df.iterrows():
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        pred = str(row.get("pred_label", ""))
        conf = float(row.get("confidence", 0.0))
        color = colors[idx % len(colors)]

        for ax in axes:
            ax.axvspan(start_s, end_s, color=color, alpha=0.18)
            ax.axvline(start_s, color="#4a5568", alpha=0.25, linewidth=0.8)

        x_mid = 0.5 * (start_s + end_s)
        y_text = y_max - (idx % 3) * (0.12 * span)
        ax1.text(
            x_mid,
            y_text,
            f"{pred} ({conf:.2f})",
            ha="center",
            va="top",
            fontsize=8,
            color="#1a202c",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
        )

    fig.suptitle(title if title else f"{zip_path.name} Prediction Segments", fontsize=12)
    _save_fig(fig, out_png)
    return out_png


def plot_prediction_directory(input_dir, prediction_dir, out_dir=None) -> List[Path]:
    input_dir = Path(input_dir)
    prediction_dir = Path(prediction_dir)
    out_dir = ensure_dir(out_dir if out_dir is not None else prediction_dir / "figures")

    summary_path = prediction_dir / "decoding_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    outputs = []
    for _, row in summary_df.iterrows():
        zip_name = str(row["zip_file"])
        final_seq = str(row.get("final_sequence", ""))
        seg_csv = _resolve_output_csv(prediction_dir, str(row.get("output_csv", "")), zip_name)
        zip_path = _resolve_zip_file(input_dir, zip_name)
        if not seg_csv.exists() or not zip_path.exists():
            log(f"Skip plot, missing files: zip={zip_path} csv={seg_csv}")
            continue

        out_png = out_dir / f"{Path(zip_name).stem}_prediction.png"
        title = f"{zip_name} | final={final_seq}"
        outputs.append(plot_prediction_signal(zip_path, seg_csv, out_png, title=title))

    return outputs
