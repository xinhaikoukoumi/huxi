import argparse
from pathlib import Path

from src.train import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train Morse letter classifier from AD+BC zip dataset.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training zips directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Artifacts output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="letters",
        choices=["letters", "digits"],
        help="Label mode: letters(A-Z) or digits(0-9).",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="dual",
        choices=["dual", "ch1", "ch2"],
        help="Input channel mode: dual(ch1+ch2), ch1 only, or ch2 only.",
    )
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional override for max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for batch size")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Optional override for early stopping patience",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild segments_cache.npz and meta.csv even if cache exists",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random_segments",
        choices=["random_segments", "group_by_file", "group_by_file_stratified"],
        help="Data split mode for validation/testing.",
    )
    parser.add_argument("--clip_low_pct", type=float, default=None, help="Lower percentile for spike clipping.")
    parser.add_argument("--clip_high_pct", type=float, default=None, help="Upper percentile for spike clipping.")
    parser.add_argument("--median_window", type=int, default=None, help="Median filter window size.")
    parser.add_argument("--smooth_window", type=int, default=None, help="Moving-average filter window size.")
    parser.add_argument(
        "--segment_offsets_sec",
        type=str,
        default=None,
        help="Comma-separated segment offsets in seconds for cache building, e.g. '0,10,20'.",
    )
    parser.add_argument("--learning_rate", type=float, default=None, help="Optional override learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Optional override weight decay.")
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["ce", "focal"],
        help="Optional loss type: ce or focal.",
    )
    parser.add_argument("--focal_gamma", type=float, default=None, help="Optional focal loss gamma.")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Optional override label smoothing.")
    parser.add_argument("--mixup_alpha", type=float, default=None, help="Optional mixup beta alpha.")
    parser.add_argument("--mixup_prob", type=float, default=None, help="Optional probability to apply mixup.")
    parser.add_argument("--aug_shift_max", type=int, default=None, help="Optional max random time shift (points).")
    parser.add_argument("--aug_noise_std", type=float, default=None, help="Optional gaussian noise std.")
    parser.add_argument("--aug_scale_min", type=float, default=None, help="Optional minimum channel scale.")
    parser.add_argument("--aug_scale_max", type=float, default=None, help="Optional maximum channel scale.")
    parser.add_argument("--aug_drift_max", type=float, default=None, help="Optional max baseline drift slope.")
    parser.add_argument("--aug_time_mask_prob", type=float, default=None, help="Optional time mask probability.")
    parser.add_argument(
        "--aug_time_mask_max_width",
        type=int,
        default=None,
        help="Optional max width for time masking (points).",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        choices=["enhanced_reslstm", "baseline_cnn_bilstm"],
        help="Optional model architecture variant.",
    )
    parser.add_argument("--init_model_path", type=str, default=None, help="Optional checkpoint path for fine-tuning.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_pipeline(
        train_dir=Path(args.train_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        label_mode=args.label_mode,
        channel_mode=args.channel_mode,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        split_mode=args.split_mode,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        aug_shift_max=args.aug_shift_max,
        aug_noise_std=args.aug_noise_std,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_drift_max=args.aug_drift_max,
        aug_time_mask_prob=args.aug_time_mask_prob,
        aug_time_mask_max_width=args.aug_time_mask_max_width,
        model_variant=args.model_variant,
        init_model_path=Path(args.init_model_path) if args.init_model_path else None,
        clip_low_pct=args.clip_low_pct,
        clip_high_pct=args.clip_high_pct,
        median_window=args.median_window,
        smooth_window=args.smooth_window,
        segment_offsets_sec=args.segment_offsets_sec,
        rebuild_cache=args.rebuild_cache,
    )
    print("best_epoch:", result["best_epoch"])
    print("best_val_macro_f1:", f"{result['best_val_macro_f1']:.6f}")
    print("test_macro_f1:", f"{result['test_macro_f1']:.6f}")
    print("test_accuracy:", f"{result['test_accuracy']:.6f}")
    print("model_path:", result["artifacts"]["model_path"])


if __name__ == "__main__":
    main()
