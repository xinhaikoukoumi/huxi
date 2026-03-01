import argparse
from pathlib import Path

from src.predict import predict_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Morse letter segments from zip files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label_map.json")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory with zip files")
    parser.add_argument("--input_file", type=str, default=None, help="Optional single input zip file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for segment CSV files")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument(
        "--offset_step_sec",
        type=float,
        default=1.0,
        help="Offset search step in seconds within [0, 30).",
    )
    parser.add_argument(
        "--disable_offset_search",
        action="store_true",
        help="Disable offset search and use 0s fixed alignment.",
    )
    parser.add_argument(
        "--disable_adaptive_boundaries",
        action="store_true",
        help="Disable per-boundary adaptive refinement and use fixed 30s windows.",
    )
    parser.add_argument(
        "--boundary_search_radius",
        type=float,
        default=2.0,
        help="Boundary refinement search radius in seconds.",
    )
    parser.add_argument(
        "--lexicon_file",
        type=str,
        default=None,
        help="Optional lexicon text file (one phrase per line, e.g., HELP ME).",
    )
    parser.add_argument(
        "--disable_lexicon_decoder",
        action="store_true",
        help="Disable lexicon constrained decoding.",
    )
    parser.add_argument(
        "--disable_auto_lexicon",
        action="store_true",
        help="Disable auto lexicon extraction from input zip filenames.",
    )
    parser.add_argument(
        "--disable_force_lexicon",
        action="store_true",
        help="Do not force lexicon decoding; only apply when lexicon score is close.",
    )
    parser.add_argument(
        "--lexicon_length_tolerance",
        type=int,
        default=0,
        help="Only use lexicon entries with length close to segment count (abs diff <= tolerance).",
    )
    parser.add_argument("--clip_low_pct", type=float, default=None, help="Override lower clipping percentile.")
    parser.add_argument("--clip_high_pct", type=float, default=None, help="Override upper clipping percentile.")
    parser.add_argument("--median_window", type=int, default=None, help="Override median filter window.")
    parser.add_argument("--smooth_window", type=int, default=None, help="Override moving-average window.")
    return parser.parse_args()


def validate_input_args(input_dir, input_file):
    has_dir = bool(input_dir)
    has_file = bool(input_file)
    if has_dir == has_file:
        raise ValueError("Exactly one of --input_dir or --input_file must be provided.")


def main():
    args = parse_args()
    validate_input_args(args.input_dir, args.input_file)
    results = predict_directory(
        model_path=Path(args.model_path),
        label_map_path=Path(args.label_map),
        input_dir=Path(args.input_dir) if args.input_dir else None,
        input_file=Path(args.input_file) if args.input_file else None,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        offset_step_sec=args.offset_step_sec,
        search_offset=not args.disable_offset_search,
        adaptive_boundaries=not args.disable_adaptive_boundaries,
        boundary_search_radius=args.boundary_search_radius,
        use_lexicon_decoder=not args.disable_lexicon_decoder,
        lexicon_file=Path(args.lexicon_file) if args.lexicon_file else None,
        auto_lexicon=not args.disable_auto_lexicon,
        force_lexicon=not args.disable_force_lexicon,
        lexicon_length_tolerance=args.lexicon_length_tolerance,
        clip_low_pct=args.clip_low_pct,
        clip_high_pct=args.clip_high_pct,
        median_window=args.median_window,
        smooth_window=args.smooth_window,
    )
    print("predicted_files:", len(results))


if __name__ == "__main__":
    main()
