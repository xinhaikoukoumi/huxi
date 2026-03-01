from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import TrainConfig
from .dataset import (
    MorseSegmentDataset,
    build_cache,
    channel_mode_to_indices,
    group_by_file_split,
    load_cache,
    stratified_group_by_file_split,
    stratified_random_split,
)
from .evaluate import compute_metrics
from .model import MorseCharModel
from .utils import ensure_dir, get_device, log, save_json, set_seed


def _compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = float(max(0.0, gamma))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_probs = probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1).clamp(1e-8, 1.0)
        focal_mod = torch.pow(1.0 - target_probs, self.gamma)
        ce = F.nll_loss(
            log_probs,
            target,
            weight=self.class_weights,
            reduction="none",
        )
        return torch.mean(focal_mod * ce)


def _run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        use_mixup = (
            mixup_alpha > 0.0
            and mixup_prob > 0.0
            and x.size(0) > 1
            and float(np.random.rand()) < mixup_prob
        )
        if use_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=device)
            x_mix = lam * x + (1.0 - lam) * x[perm]
            logits = model(x_mix)
            loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y[perm])
        else:
            logits = model(x)
            loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch = y.size(0)
        total_loss += float(loss.item()) * batch
        correct += int((pred == y).sum().item())
        total += batch
    return total_loss / max(total, 1), float(correct) / float(max(total, 1))


@torch.no_grad()
def _evaluate_loader(model, loader, criterion, device, label_ids):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        batch = y.size(0)
        total_loss += float(loss.item()) * batch
        total += batch
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true_arr = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred_arr = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    metrics = compute_metrics(y_true_arr, y_pred_arr, label_ids=label_ids)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics, y_true_arr, y_pred_arr


def train_pipeline(
    train_dir,
    out_dir,
    seed: int = 42,
    label_mode: str = None,
    channel_mode: str = None,
    max_epochs: int = None,
    batch_size: int = None,
    early_stopping_patience: int = None,
    split_mode: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
    loss_type: str = None,
    focal_gamma: float = None,
    label_smoothing: float = None,
    mixup_alpha: float = None,
    mixup_prob: float = None,
    aug_shift_max: int = None,
    aug_noise_std: float = None,
    aug_scale_min: float = None,
    aug_scale_max: float = None,
    aug_drift_max: float = None,
    aug_time_mask_prob: float = None,
    aug_time_mask_max_width: int = None,
    model_variant: str = None,
    init_model_path=None,
    clip_low_pct: float = None,
    clip_high_pct: float = None,
    median_window: int = None,
    smooth_window: int = None,
    segment_offsets_sec=None,
    rebuild_cache: bool = False,
) -> Dict:
    cfg = TrainConfig(seed=seed)
    if label_mode is not None:
        cfg.label_mode = str(label_mode)
    if cfg.label_mode not in ("letters", "digits"):
        raise ValueError("label_mode must be 'letters' or 'digits'")
    if channel_mode is not None:
        cfg.channel_mode = str(channel_mode)
    _ = channel_mode_to_indices(cfg.channel_mode)
    if max_epochs is not None:
        cfg.max_epochs = int(max_epochs)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if early_stopping_patience is not None:
        cfg.early_stopping_patience = int(early_stopping_patience)
    if split_mode is not None:
        cfg.split_mode = str(split_mode)
    if learning_rate is not None:
        cfg.learning_rate = float(learning_rate)
    if weight_decay is not None:
        cfg.weight_decay = float(weight_decay)
    if loss_type is not None:
        cfg.loss_type = str(loss_type)
    if cfg.loss_type not in ("ce", "focal"):
        raise ValueError("loss_type must be 'ce' or 'focal'")
    if focal_gamma is not None:
        cfg.focal_gamma = float(focal_gamma)
    if label_smoothing is not None:
        cfg.label_smoothing = float(label_smoothing)
    if mixup_alpha is not None:
        cfg.mixup_alpha = float(mixup_alpha)
    if mixup_prob is not None:
        cfg.mixup_prob = float(mixup_prob)
    if aug_shift_max is not None:
        cfg.aug_shift_max = int(aug_shift_max)
    if aug_noise_std is not None:
        cfg.aug_noise_std = float(aug_noise_std)
    if aug_scale_min is not None:
        cfg.aug_scale_min = float(aug_scale_min)
    if aug_scale_max is not None:
        cfg.aug_scale_max = float(aug_scale_max)
    if aug_drift_max is not None:
        cfg.aug_drift_max = float(aug_drift_max)
    if aug_time_mask_prob is not None:
        cfg.aug_time_mask_prob = float(aug_time_mask_prob)
    if aug_time_mask_max_width is not None:
        cfg.aug_time_mask_max_width = int(aug_time_mask_max_width)
    if model_variant is not None:
        cfg.model_variant = str(model_variant)
    if clip_low_pct is not None:
        cfg.clip_low_pct = float(clip_low_pct)
    if clip_high_pct is not None:
        cfg.clip_high_pct = float(clip_high_pct)
    if median_window is not None:
        cfg.median_window = int(median_window)
    if smooth_window is not None:
        cfg.smooth_window = int(smooth_window)
    if segment_offsets_sec is not None:
        cfg.segment_offsets_sec = str(segment_offsets_sec)

    out_dir = ensure_dir(out_dir)
    cache_path = out_dir / "segments_cache.npz"
    meta_path = out_dir / "meta.csv"
    label_map_path = out_dir / "label_map.json"
    config_path = out_dir / "train_config.json"
    model_path = out_dir / "morse_char_model.pt"
    history_path = out_dir / "train_history.csv"
    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.csv"
    risk_note_path = out_dir / "evaluation_note.txt"

    set_seed(cfg.seed)
    device = get_device()
    log(f"Device: {device}")

    if rebuild_cache or not cache_path.exists() or not label_map_path.exists():
        bundle = build_cache(
            train_dir=train_dir,
            cache_path=cache_path,
            meta_path=meta_path,
            label_map_path=label_map_path,
            window_sec=cfg.window_sec,
            target_points=cfg.target_points,
            min_valid_ratio=cfg.min_valid_ratio,
            clip_low_pct=cfg.clip_low_pct,
            clip_high_pct=cfg.clip_high_pct,
            median_window=cfg.median_window,
            smooth_window=cfg.smooth_window,
            detrend=cfg.detrend,
            label_mode=cfg.label_mode,
            channel_mode=cfg.channel_mode,
            offsets_sec=cfg.segment_offsets_sec,
        )
    else:
        bundle = load_cache(cache_path, label_map_path)
        expected_channels = len(channel_mode_to_indices(cfg.channel_mode))
        if int(bundle.X.shape[1]) != expected_channels:
            raise ValueError(
                "Cached channel count mismatch: "
                f"cache has {bundle.X.shape[1]} channels but channel_mode={cfg.channel_mode} "
                f"expects {expected_channels}. Use --rebuild_cache or a separate --out_dir."
            )
        if not meta_path.exists():
            log(f"meta.csv not found at {meta_path}, proceeding with cache only")

    save_json(config_path, cfg.to_dict())

    X = bundle.X
    y = bundle.y
    label_map = bundle.label_map
    if cfg.label_mode == "digits":
        if any((not str(k).isdigit()) for k in label_map.keys()):
            raise ValueError(
                "Loaded cache/labels do not match digits mode. "
                "Use --rebuild_cache or separate --out_dir."
            )
    else:
        if any((not str(k).isalpha()) for k in label_map.keys()):
            raise ValueError(
                "Loaded cache/labels do not match letters mode. "
                "Use --rebuild_cache or separate --out_dir."
            )
    id_to_label = {v: k for k, v in label_map.items()}
    label_ids = sorted(id_to_label.keys())
    num_classes = len(label_ids)

    if cfg.split_mode == "group_by_file":
        train_idx, val_idx, test_idx = group_by_file_split(
            y,
            bundle.source_files,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
    elif cfg.split_mode == "group_by_file_stratified":
        train_idx, val_idx, test_idx = stratified_group_by_file_split(
            y,
            bundle.source_files,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
    else:
        train_idx, val_idx, test_idx = stratified_random_split(
            y,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )

    train_ds = MorseSegmentDataset(
        X,
        y,
        train_idx,
        augment=True,
        shift_max=cfg.aug_shift_max,
        noise_std=cfg.aug_noise_std,
        scale_min=cfg.aug_scale_min,
        scale_max=cfg.aug_scale_max,
        drift_max=cfg.aug_drift_max,
        time_mask_prob=cfg.aug_time_mask_prob,
        time_mask_max_width=cfg.aug_time_mask_max_width,
    )
    train_eval_ds = MorseSegmentDataset(X, y, train_idx, augment=False)
    val_ds = MorseSegmentDataset(X, y, val_idx, augment=False)
    test_ds = MorseSegmentDataset(X, y, test_idx, augment=False)

    pin_memory = device.type == "cuda"
    train_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
    }
    if cfg.split_mode in ("group_by_file", "group_by_file_stratified"):
        y_train = y[train_idx]
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        sample_weights = (1.0 / class_counts[y_train]).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **train_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = MorseCharModel(
        num_classes=num_classes,
        input_channels=int(X.shape[1]),
        variant=cfg.model_variant,
    ).to(device)
    if init_model_path is not None:
        init_model_path = Path(init_model_path)
        if init_model_path.exists():
            init_ckpt = torch.load(init_model_path, map_location=device)
            init_state = init_ckpt["state_dict"]
            if any(str(k).startswith("net.") for k in init_state.keys()):
                init_state = {k[4:]: v for k, v in init_state.items() if str(k).startswith("net.")}
            model.load_state_dict(init_state, strict=False)
            log(f"Initialized model from: {init_model_path}")
        else:
            log(f"init_model_path not found, training from scratch: {init_model_path}")
    log(
        "Train setup: "
        f"loss_type={cfg.loss_type} focal_gamma={cfg.focal_gamma:.2f} "
        f"aug_shift_max={cfg.aug_shift_max} aug_noise_std={cfg.aug_noise_std:.4f} "
        f"aug_time_mask_prob={cfg.aug_time_mask_prob:.2f}"
    )
    class_weights = _compute_class_weights(y[train_idx], num_classes=num_classes).to(device)
    if cfg.loss_type == "focal":
        criterion = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=cfg.focal_gamma,
        )
    else:
        label_smoothing = cfg.label_smoothing
        if cfg.split_mode in ("group_by_file", "group_by_file_stratified"):
            label_smoothing = min(label_smoothing, 0.05)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            mixup_alpha=cfg.mixup_alpha,
            mixup_prob=cfg.mixup_prob,
        )
        train_clean_metrics, _, _ = _evaluate_loader(model, train_eval_loader, criterion, device, label_ids)
        val_metrics, _, _ = _evaluate_loader(model, val_loader, criterion, device, label_ids)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss_aug": train_loss,
            "train_acc_aug": train_acc,
            "train_loss_clean": train_clean_metrics["loss"],
            "train_acc_clean": train_clean_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        log(
            f"Epoch {epoch:03d}/{cfg.max_epochs}: "
            f"train_aug_loss={train_loss:.4f} train_aug_acc={train_acc:.4f} "
            f"train_clean_loss={train_clean_metrics['loss']:.4f} "
            f"train_clean_acc={train_clean_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_channels": int(X.shape[1]),
                "target_points": cfg.target_points,
                "label_map": label_map,
                "config": asdict(cfg),
            }
            torch.save(checkpoint, model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                log("Early stopping triggered")
                break

    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, y_true, y_pred = _evaluate_loader(model, test_loader, criterion, device, label_ids)

    cm = np.array(test_metrics["confusion_matrix"], dtype=np.int64)
    cm_df = pd.DataFrame(
        cm,
        index=[id_to_label[i] for i in label_ids],
        columns=[id_to_label[i] for i in label_ids],
    )
    cm_df.to_csv(cm_path, encoding="utf-8")

    per_class_recall = {
        id_to_label[label_ids[i]]: float(test_metrics["per_class_recall"][i])
        for i in range(len(label_ids))
    }

    result = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "per_class_recall": per_class_recall,
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_mode": cfg.split_mode,
        "artifacts": {
            "model_path": str(model_path),
            "cache_path": str(cache_path),
            "meta_path": str(meta_path),
            "label_map_path": str(label_map_path),
            "history_path": str(history_path),
            "metrics_path": str(metrics_path),
            "cm_path": str(cm_path),
            "config_path": str(config_path),
        },
    }
    save_json(metrics_path, result)

    if cfg.split_mode in ("group_by_file", "group_by_file_stratified"):
        risk_note = (
            "This run uses file-group split (train/val/test), reducing same-file leakage.\n"
            "Metrics are more conservative and closer to real generalization.\n"
        )
    else:
        risk_note = (
            "This run uses random segment-level split (train/val/test).\n"
            "Potential same-file leakage can inflate metrics compared to file-level split.\n"
        )
    risk_note_path.write_text(risk_note, encoding="utf-8")

    log(
        f"Training done. best_val_macro_f1={best_val_f1:.4f}, "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    return result
