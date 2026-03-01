import argparse
from pathlib import Path

from src.visualize import plot_confusion_matrix, plot_prediction_directory, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training/prediction visualization figures.")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Training artifacts directory")
    parser.add_argument("--prediction_dir", type=str, required=True, help="Prediction output directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Input zip directory for signal plotting")
    parser.add_argument(
        "--figure_dir",
        type=str,
        default=None,
        help="Optional root figure directory; defaults to artifacts/prediction figure subdirs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    prediction_dir = Path(args.prediction_dir)
    input_dir = Path(args.input_dir)

    if args.figure_dir:
        figure_root = Path(args.figure_dir)
        train_fig_dir = figure_root / "training"
        pred_fig_dir = figure_root / "prediction"
    else:
        train_fig_dir = artifacts_dir / "figures"
        pred_fig_dir = prediction_dir / "figures"

    history_path = artifacts_dir / "train_history.csv"
    cm_path = artifacts_dir / "confusion_matrix.csv"

    train_figs = plot_training_history(history_path, train_fig_dir)
    cm_figs = plot_confusion_matrix(
        cm_path,
        train_fig_dir / "confusion_matrix.png",
        annotate_nonzero=True,
        show_normalized=True,
    )
    pred_figs = plot_prediction_directory(input_dir=input_dir, prediction_dir=prediction_dir, out_dir=pred_fig_dir)

    print("train_figures:", len(train_figs))
    for p in train_figs:
        print(" -", p)
    print("confusion_matrix_figures:", len(cm_figs))
    for p in cm_figs:
        print(" -", p)
    print("prediction_figures:", len(pred_figs))
    for p in pred_figs:
        print(" -", p)


if __name__ == "__main__":
    main()
