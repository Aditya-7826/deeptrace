"""Training configuration (YAML -> dataclasses)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from deeptrace.models.factory import ModelConfig


@dataclass
class DataConfig:
    root: str = "data/crops/kaggle140k"  # folder containing manifest.csv
    datasets: list[str] | None = None  # restrict to these source_dataset names
    max_train: int | None = None
    max_val: int | None = None
    balance: str = "sampler"  # "sampler", "loss" or "none"
    num_workers: int = 2


@dataclass
class OptimConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3  # head LR
    backbone_lr_mult: float = 0.1  # stage-2 backbone LR = lr * mult
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01
    freeze_epochs: int = 1  # stage 1: head only. freeze_epochs >= epochs = frozen ablation
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 3
    seed: int = 42
    threads: int = 4
    device: str = "auto"
    time_budget_hours: float | None = None  # stop cleanly before a Kaggle session limit


@dataclass
class TrainConfig:
    run_name: str
    output_root: str = "artifacts/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: bool = True
    train: OptimConfig = field(default_factory=OptimConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        raw = dict(raw)
        cfg = cls(
            run_name=raw.pop("run_name"),
            output_root=raw.pop("output_root", "artifacts/runs"),
            data=DataConfig(**raw.pop("data", {})),
            model=ModelConfig(**raw.pop("model", {})),
            augment=raw.pop("augment", True),
            train=OptimConfig(**raw.pop("train", {})),
        )
        if raw:
            raise ValueError(f"unknown config keys: {sorted(raw)}")
        if cfg.data.balance not in ("sampler", "loss", "none"):
            raise ValueError("data.balance must be sampler, loss or none")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
