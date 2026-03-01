import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.health_analysis import (
    build_health_report,
    extract_hybrid_features,
    metrics_to_distance_frame,
    parse_health_label,
    plot_distance_heatmap,
    plot_feature_boxplots,
    plot_pca_scatter,
    run_classical_baselines,
    run_rule_based_classifier,
    run_unsupervised_analysis,
)
from src.utils import ensure_dir, log, save_json, set_seed


DEFAULT_INPUT_DIR = "d:\\huxi\\\u5065\u5eb7\u6570\u636e"
DEFAULT_OUT_BASE = r"d:\huxi\morse_train"


def parse_args():
    p = argparse.ArgumentParser(description="Health signal separability scan (non-DL, classical features/models).")
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Input directory with health zip files.")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory, default health_scan_<timestamp>.")
    p.add_argument("--recursive", action="store_true", help="Recursively scan input_dir for zip files.")
    p.add_argument("--segment_window_sec", type=float, default=0.0, help="Optional segmentation window for count feature.")
    p.add_argument("--feature_set", type=str, default="freq_time_hybrid", choices=["freq_only", "freq_time_hybrid"])
    p.add_argument("--window_sec_for_stats", type=float, default=10.0, help="Window size for robust segment stats.")
    p.add_argument("--eval_mode", type=str, default="coarse", choices=["coarse", "fine", "both"])
    p.add_argument("--cv_mode", type=str, default="group_by_day", choices=["none", "group_by_day"])
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save_plots", dest="save_plots", action="store_true", default=True, help="Save analysis plots.")
    p.add_argument("--no_save_plots", dest="save_plots", action="store_false", help="Disable plot generation.")
    return p.parse_args()


def _resolve_output_dir(output_dir: str = None) -> Path:
    if output_dir:
        return ensure_dir(Path(output_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(DEFAULT_OUT_BASE) / f"health_scan_{ts}")


def _scan_zip_files(input_dir: Path, recursive: bool):
    return sorted(input_dir.rglob("*.zip")) if recursive else sorted(input_dir.glob("*.zip"))


def _safe_unsup(features_df: pd.DataFrame, label_col: str, enabled: bool) -> dict:
    if not enabled:
        return {"status": "skipped", "reason": "disabled_by_eval_mode", "label_col": label_col}
    try:
        return run_unsupervised_analysis(features_df, label_col=label_col)
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "label_col": label_col}


def _write_brief(out_dir: Path, summary: dict, metrics_7: dict, metrics_4: dict, classical_df: pd.DataFrame) -> Path:
    brief = out_dir / "health_brief.md"
    def _best(m):
        if m.get("status") != "ok":
            return {}
        rows = list(m.get("kmeans", [])) + list(m.get("agglomerative", []))
        if not rows:
            return {}
        rows = sorted(rows, key=lambda x: float(x.get("silhouette", float("-inf"))), reverse=True)
        return rows[0]

    best7 = _best(metrics_7)
    best4 = _best(metrics_4)
    lines = [
        "# Health Signal Brief",
        "",
        "## Summary",
        f"- separability_grade: `{summary.get('separability_grade', 'unknown')}`",
        f"- rule_accuracy_vs_coarse: `{summary.get('rule_based', {}).get('accuracy_vs_coarse_label', 'nan')}`",
        f"- best_7class_silhouette: `{best7.get('silhouette', 'nan')}`",
        f"- best_4class_silhouette: `{best4.get('silhouette', 'nan')}`",
        "",
        "## Classical Baselines",
    ]
    if classical_df.empty:
        lines.append("- no classical metrics")
    else:
        top = classical_df[classical_df["status"] == "ok"].copy()
        if top.empty:
            lines.append("- all baselines skipped")
        else:
            top = top.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).head(3)
            for _, r in top.iterrows():
                lines.append(f"- {r['label_col']} | {r['model']} | macro_f1={r['macro_f1']:.4f} | acc={r['accuracy']:.4f}")
    lines.append("")
    brief.write_text("\n".join(lines), encoding="utf-8")
    return brief


def run_health_scan(
    input_dir,
    output_dir=None,
    recursive: bool = False,
    segment_window_sec: float = 0.0,
    feature_set: str = "freq_time_hybrid",
    window_sec_for_stats: float = 10.0,
    eval_mode: str = "coarse",
    cv_mode: str = "group_by_day",
    seed: int = 42,
    save_plots: bool = True,
):
    set_seed(int(seed))
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {in_dir}")
    out_dir = _resolve_output_dir(output_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    zips = _scan_zip_files(in_dir, recursive=bool(recursive))
    if not zips:
        raise ValueError(f"No zip files found in {in_dir}")

    rows, skipped = [], []
    for zp in zips:
        try:
            _ = parse_health_label(zp.name)
        except Exception as exc:
            skipped.append({"zip_file": zp.name, "reason": f"label_parse_error: {exc}"})
            continue
        try:
            rows.append(
                extract_hybrid_features(
                    zp,
                    feature_set=feature_set,
                    window_sec_for_stats=float(window_sec_for_stats),
                    segment_window_sec=float(segment_window_sec),
                )
            )
        except Exception as exc:
            skipped.append({"zip_file": zp.name, "reason": f"feature_extract_error: {exc}"})
    if not rows:
        raise RuntimeError("No valid files after parsing/extraction")

    features_df = pd.DataFrame(rows).sort_values(["coarse_label", "label", "zip_file"]).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped)
    features_df.to_csv(out_dir / "file_features.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")

    metrics_7 = _safe_unsup(features_df, "label", enabled=eval_mode in ("fine", "both"))
    metrics_4 = _safe_unsup(features_df, "coarse_label", enabled=eval_mode in ("coarse", "both"))
    save_json(out_dir / "unsupervised_7class_metrics.json", metrics_7)
    save_json(out_dir / "unsupervised_4class_metrics.json", metrics_4)

    dist_df = metrics_to_distance_frame(metrics_7 if metrics_7.get("status") == "ok" else metrics_4)
    dist_df.to_csv(out_dir / "pairwise_distance.csv", encoding="utf-8-sig")

    rule_df = run_rule_based_classifier(features_df)
    rule_df.to_csv(out_dir / "rule_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(out_dir / "rule_thresholds.json", dict(rule_df.attrs.get("thresholds", {})))

    target_cols = []
    if eval_mode in ("coarse", "both"):
        target_cols.append("coarse_label")
    if eval_mode in ("fine", "both"):
        target_cols.append("label")
    metrics_frames, error_frames, feat_imp_frames = [], [], []
    grouped_summary = {}
    for label_col in target_cols:
        out = run_classical_baselines(features_df, label_col=label_col, cv_mode=cv_mode, seed=int(seed))
        mdf = out["metrics_df"].copy()
        edf = out["error_df"].copy()
        fdf = out["feature_importance_df"].copy()
        mdf["target_label"] = label_col
        if not edf.empty:
            edf["target_label"] = label_col
        if not fdf.empty:
            fdf["target_label"] = label_col
        metrics_frames.append(mdf)
        error_frames.append(edf)
        feat_imp_frames.append(fdf)
        grouped_summary[label_col] = out["summary"]

    classical_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    error_df = pd.concat(error_frames, ignore_index=True) if error_frames and any(not x.empty for x in error_frames) else pd.DataFrame()
    feat_imp_df = pd.concat(feat_imp_frames, ignore_index=True) if feat_imp_frames and any(not x.empty for x in feat_imp_frames) else pd.DataFrame()
    classical_df.to_csv(out_dir / "classical_model_metrics.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(out_dir / "error_cases.csv", index=False, encoding="utf-8-sig")
    feat_imp_df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    save_json(out_dir / "grouped_cv_summary.json", grouped_summary)

    if save_plots:
        if metrics_7.get("status") == "ok":
            plot_pca_scatter(metrics_7, fig_dir / "pca_7class.png")
        if metrics_4.get("status") == "ok":
            plot_pca_scatter(metrics_4, fig_dir / "pca_4class.png")
        if not dist_df.empty:
            plot_distance_heatmap(dist_df, fig_dir / "distance_heatmap.png")
        plot_feature_boxplots(features_df, fig_dir / "feature_boxplots.png")

    summary = build_health_report(features_df, metrics_7class=metrics_7, metrics_4class=metrics_4, rule_df=rule_df, classical_metrics_df=classical_df, grouped_cv_summary=grouped_summary)
    summary.update(
        {
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "segment_window_sec": float(segment_window_sec),
            "window_sec_for_stats": float(window_sec_for_stats),
            "feature_set": str(feature_set),
            "eval_mode": str(eval_mode),
            "cv_mode": str(cv_mode),
            "skipped_count": int(skipped_df.shape[0]),
            "valid_count": int(features_df.shape[0]),
        }
    )
    save_json(out_dir / "health_summary.json", summary)
    brief_path = _write_brief(out_dir, summary, metrics_7, metrics_4, classical_df)

    log(f"Health scan done. output_dir={out_dir}")
    return {"output_dir": str(out_dir), "summary_json": str(out_dir / "health_summary.json"), "brief_md": str(brief_path)}


def main():
    args = parse_args()
    result = run_health_scan(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        segment_window_sec=args.segment_window_sec,
        feature_set=args.feature_set,
        window_sec_for_stats=args.window_sec_for_stats,
        eval_mode=args.eval_mode,
        cv_mode=args.cv_mode,
        seed=args.seed,
        save_plots=args.save_plots,
    )
    print("output_dir:", result["output_dir"])
    print("summary_json:", result["summary_json"])
    print("brief_md:", result["brief_md"])


if __name__ == "__main__":
    main()
