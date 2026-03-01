from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .dataset import channel_mode_to_indices, select_channel_mode
from .decoder import (
    decode_with_lexicon,
    infer_lexicon_from_filenames,
    load_lexicon,
    phrase_letters,
)
from .io_zip import load_signal_from_zip
from .model import MorseCharModel
from .preprocess import segment_signal
from .utils import ensure_dir, get_device, load_json, log


def _load_model(model_path, label_map_path, device):
    label_map = load_json(label_map_path)
    id_to_label = {int(v): k for k, v in label_map.items()}
    num_classes = len(label_map)

    checkpoint = torch.load(model_path, map_location=device)
    model_variant = checkpoint.get("config", {}).get("model_variant", "enhanced_reslstm")
    input_channels = int(checkpoint.get("input_channels", 2))
    channel_mode = checkpoint.get("config", {}).get("channel_mode")
    if channel_mode is None:
        if input_channels == 2:
            channel_mode = "dual"
        elif input_channels == 1:
            channel_mode = "ch1"
            log("Checkpoint missing channel_mode, fallback to ch1 for single-channel model")
        else:
            raise ValueError(f"Unsupported input_channels in checkpoint: {input_channels}")
    _ = channel_mode_to_indices(channel_mode)
    model = MorseCharModel(
        num_classes=num_classes,
        input_channels=input_channels,
        variant=model_variant,
    ).to(device)
    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Backward compatibility for temporary checkpoints saved with "net." prefix.
        if any(str(k).startswith("net.") for k in state_dict.keys()):
            stripped = {k[4:]: v for k, v in state_dict.items() if str(k).startswith("net.")}
            model.load_state_dict(stripped, strict=False)
        else:
            raise
    model.eval()
    train_cfg = checkpoint.get("config", {})
    train_cfg = dict(train_cfg)
    train_cfg["channel_mode"] = str(channel_mode)
    train_cfg["input_channels"] = int(input_channels)
    return model, id_to_label, train_cfg


@torch.no_grad()
def _predict_batch(model, batch_x: np.ndarray, device):
    x = torch.from_numpy(batch_x).float().to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.cpu().numpy(), conf.cpu().numpy(), probs.cpu().numpy()


@torch.no_grad()
def _predict_segments(
    model,
    segs,
    device,
    id_to_label,
    batch_size: int,
    channel_mode: str,
    expected_input_channels: int,
):
    if not segs:
        return [], np.empty((0, len(id_to_label)), dtype=np.float32)
    X = np.stack([select_channel_mode(s.x, channel_mode) for s in segs]).astype(np.float32)
    if int(X.shape[1]) != int(expected_input_channels):
        raise ValueError(
            f"Model expects {expected_input_channels} channels, but prepared input has {X.shape[1]}"
        )
    pred_all = []
    conf_all = []
    probs_all = []
    for i in range(0, X.shape[0], batch_size):
        pred, conf, probs = _predict_batch(model, X[i : i + batch_size], device)
        pred_all.append(pred)
        conf_all.append(conf)
        probs_all.append(probs)
    pred_ids = np.concatenate(pred_all)
    confidences = np.concatenate(conf_all)
    prob_matrix = np.concatenate(probs_all).astype(np.float32)
    rows = []
    for idx, seg in enumerate(segs):
        rows.append(
            {
                "segment_idx": int(seg.segment_idx),
                "start_s": float(seg.start_s),
                "end_s": float(seg.end_s),
                "raw_pred_label": id_to_label[int(pred_ids[idx])],
                "pred_label": id_to_label[int(pred_ids[idx])],
                "confidence": float(confidences[idx]),
                "raw_confidence": float(confidences[idx]),
            }
        )
    return rows, prob_matrix


def predict_directory(
    model_path,
    label_map_path,
    output_dir,
    input_dir=None,
    input_file=None,
    window_sec: float = None,
    target_points: int = None,
    min_valid_ratio: float = None,
    batch_size: int = 128,
    offset_step_sec: float = 1.0,
    search_offset: bool = True,
    adaptive_boundaries: bool = True,
    boundary_search_radius: float = 2.0,
    use_lexicon_decoder: bool = True,
    lexicon_file=None,
    auto_lexicon: bool = True,
    force_lexicon: bool = True,
    lexicon_length_tolerance: int = 0,
    clip_low_pct: float = None,
    clip_high_pct: float = None,
    median_window: int = None,
    smooth_window: int = None,
    detrend: bool = None,
) -> List[Dict]:
    has_dir = input_dir is not None
    has_file = input_file is not None
    if has_dir == has_file:
        raise ValueError("Exactly one of input_dir or input_file must be provided")

    input_dir = Path(input_dir) if input_dir is not None else None
    input_file = Path(input_file) if input_file is not None else None
    output_dir = ensure_dir(output_dir)
    device = get_device()
    log(f"Device: {device}")

    model, id_to_label, train_cfg = _load_model(model_path, label_map_path, device)
    label_to_id = {v: k for k, v in id_to_label.items()}
    channel_mode = str(train_cfg.get("channel_mode", "dual"))
    expected_input_channels = int(train_cfg.get("input_channels", 2))

    window_sec = float(window_sec if window_sec is not None else train_cfg.get("window_sec", 30.0))
    target_points = int(target_points if target_points is not None else train_cfg.get("target_points", 300))
    min_valid_ratio = float(min_valid_ratio if min_valid_ratio is not None else train_cfg.get("min_valid_ratio", 0.6))
    clip_low_pct = float(clip_low_pct if clip_low_pct is not None else train_cfg.get("clip_low_pct", 0.5))
    clip_high_pct = float(clip_high_pct if clip_high_pct is not None else train_cfg.get("clip_high_pct", 99.5))
    median_window = int(median_window if median_window is not None else train_cfg.get("median_window", 5))
    smooth_window = int(smooth_window if smooth_window is not None else train_cfg.get("smooth_window", 9))
    detrend = bool(detrend if detrend is not None else train_cfg.get("detrend", True))

    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"input_file not found: {input_file}")
        zip_paths = [input_file]
    else:
        zip_paths = sorted(input_dir.glob("*.zip"))
    lexicon = []
    if use_lexicon_decoder:
        lexicon.extend(load_lexicon(lexicon_file))
        if auto_lexicon:
            lexicon.extend(infer_lexicon_from_filenames([p.name for p in zip_paths]))
        lexicon = sorted(set([x for x in lexicon if x]))
        log(f"Lexicon size: {len(lexicon)}")

    results = []

    for zip_path in zip_paths:
        signal = load_signal_from_zip(zip_path)
        rows = []
        best_offset = 0.0
        best_decode_phrase = ""
        best_decode_score = -1e9
        raw_sequence = ""
        final_sequence = ""

        if search_offset:
            t = signal.frame["time_s"].to_numpy()
            max_time = float(np.nanmax(t)) - float(np.nanmin(t))
            nominal_segments = int(max_time // window_sec)
            offsets = np.arange(0.0, window_sec, max(offset_step_sec, 0.1), dtype=np.float32)
            candidates = []
            for offset in offsets:
                segs = segment_signal(
                    signal.frame,
                    window_sec=window_sec,
                    target_points=target_points,
                    min_valid_ratio=min_valid_ratio,
                    offset_sec=float(offset),
                    clip_low_pct=clip_low_pct,
                    clip_high_pct=clip_high_pct,
                    median_window=median_window,
                    smooth_window=smooth_window,
                    detrend=detrend,
                    adaptive_boundaries=adaptive_boundaries,
                    boundary_search_radius=boundary_search_radius,
                )
                candidate_rows, candidate_probs = _predict_segments(
                    model,
                    segs,
                    device,
                    id_to_label,
                    batch_size,
                    channel_mode=channel_mode,
                    expected_input_channels=expected_input_channels,
                )
                if not candidate_rows:
                    continue
                mean_conf = float(np.mean([r["confidence"] for r in candidate_rows]))
                sorted_probs = np.sort(candidate_probs, axis=1)
                mean_margin = float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])) if candidate_probs.shape[1] > 1 else 0.0
                raw_seq = "".join([r["raw_pred_label"] for r in candidate_rows])
                decoded_phrase = ""
                decoded_letters = raw_seq
                decode_score = float(np.log(max(1e-8, mean_conf)))
                nolex_score = mean_conf + 0.4 * mean_margin
                if use_lexicon_decoder and lexicon:
                    filtered_lexicon = [
                        p
                        for p in lexicon
                        if abs(len(phrase_letters(p)) - len(candidate_rows)) <= int(lexicon_length_tolerance)
                    ]
                    if not filtered_lexicon:
                        filtered_lexicon = lexicon
                    decoded = decode_with_lexicon(
                        candidate_probs,
                        raw_seq,
                        filtered_lexicon,
                        force_lexicon=force_lexicon,
                    )
                    decoded_phrase = decoded.decoded_phrase
                    decoded_letters = decoded.decoded_letters
                    decode_score = float(decoded.score)

                candidates.append(
                    (
                        float(offset),
                        candidate_rows,
                        candidate_probs,
                        mean_conf,
                        len(candidate_rows),
                        raw_seq,
                        decoded_phrase,
                        decoded_letters,
                        decode_score,
                        nolex_score,
                    )
                )

            if candidates:
                same_len = [c for c in candidates if c[4] == nominal_segments]
                pool = same_len if same_len else candidates
                if use_lexicon_decoder and lexicon:
                    best = max(pool, key=lambda x: x[8])
                else:
                    best = max(pool, key=lambda x: x[9])
                best_offset = best[0]
                rows = best[1]
                raw_sequence = best[5]
                best_decode_phrase = best[6]
                decoded_letters = best[7]
                best_decode_score = best[8]

                if decoded_letters and len(decoded_letters) == len(rows):
                    for i in range(len(rows)):
                        rows[i]["pred_label"] = decoded_letters[i]
                        decoded_id = label_to_id.get(decoded_letters[i], None)
                        if decoded_id is not None:
                            rows[i]["confidence"] = float(best[2][i, decoded_id])
                final_sequence = "".join([r["pred_label"] for r in rows])
        else:
            segs = segment_signal(
                signal.frame,
                window_sec=window_sec,
                target_points=target_points,
                min_valid_ratio=min_valid_ratio,
                offset_sec=0.0,
                clip_low_pct=clip_low_pct,
                clip_high_pct=clip_high_pct,
                median_window=median_window,
                smooth_window=smooth_window,
                detrend=detrend,
                adaptive_boundaries=adaptive_boundaries,
                boundary_search_radius=boundary_search_radius,
            )
            rows, prob_matrix = _predict_segments(
                model,
                segs,
                device,
                id_to_label,
                batch_size,
                channel_mode=channel_mode,
                expected_input_channels=expected_input_channels,
            )
            raw_sequence = "".join([r["raw_pred_label"] for r in rows])
            final_sequence = raw_sequence
            best_decode_score = float(np.log(max(1e-8, np.mean([r["confidence"] for r in rows]) if rows else 1e-8)))
            if use_lexicon_decoder and lexicon and rows:
                filtered_lexicon = [
                    p
                    for p in lexicon
                    if abs(len(phrase_letters(p)) - len(rows)) <= int(lexicon_length_tolerance)
                ]
                if not filtered_lexicon:
                    filtered_lexicon = lexicon
                decoded = decode_with_lexicon(
                    prob_matrix,
                    raw_sequence,
                    filtered_lexicon,
                    force_lexicon=force_lexicon,
                )
                best_decode_phrase = decoded.decoded_phrase
                best_decode_score = decoded.score
                if decoded.decoded_letters and len(decoded.decoded_letters) == len(rows):
                    for i in range(len(rows)):
                        rows[i]["pred_label"] = decoded.decoded_letters[i]
                        decoded_id = label_to_id.get(decoded.decoded_letters[i], None)
                        if decoded_id is not None:
                            rows[i]["confidence"] = float(prob_matrix[i, decoded_id])
                    final_sequence = decoded.decoded_letters

        out_file = output_dir / f"{zip_path.stem}_segments.csv"
        for r in rows:
            r["offset_sec"] = best_offset
        pd.DataFrame(
            rows,
            columns=[
                "segment_idx",
                "start_s",
                "end_s",
                "pred_label",
                "confidence",
                "raw_confidence",
                "raw_pred_label",
                "offset_sec",
            ],
        ).to_csv(
            out_file,
            index=False,
            encoding="utf-8",
        )

        if not final_sequence:
            final_sequence = "".join([r["pred_label"] for r in rows])
        if not raw_sequence:
            raw_sequence = "".join([r["raw_pred_label"] for r in rows])

        mean_conf = float(np.mean([r["confidence"] for r in rows])) if rows else 0.0
        log(
            f"{zip_path.name}: segments={len(rows)} offset={best_offset:.1f}s "
            f"raw={raw_sequence} final={final_sequence}"
        )
        result_row = {
            "zip_file": zip_path.name,
            "segments": len(rows),
            "offset_sec": best_offset,
            "raw_sequence": raw_sequence,
            "final_sequence": final_sequence,
            "decoded_phrase": best_decode_phrase,
            "decode_score": float(best_decode_score),
            "mean_confidence": mean_conf,
            "output_csv": str(out_file),
        }
        results.append(result_row)

    summary_path = output_dir / "decoding_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8")
    log(f"Saved summary: {summary_path}")
    return results
