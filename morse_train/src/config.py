from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    label_mode: str = "letters"
    channel_mode: str = "dual"

    window_sec: float = 30.0
    target_points: int = 300
    min_valid_ratio: float = 0.6
    segment_offsets_sec: str = "0"

    batch_size: int = 64
    max_epochs: int = 120
    early_stopping_patience: int = 15
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    loss_type: str = "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    model_variant: str = "enhanced_reslstm"
    aug_shift_max: int = 12
    aug_noise_std: float = 0.02
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.15
    aug_drift_max: float = 0.08
    aug_time_mask_prob: float = 0.35
    aug_time_mask_max_width: int = 24

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_mode: str = "random_segments"

    clip_low_pct: float = 0.5
    clip_high_pct: float = 99.5
    median_window: int = 5
    smooth_window: int = 9
    detrend: bool = True

    seed: int = 42
    num_workers: int = 0

    def to_dict(self):
        return asdict(self)
