"""Load a trained model from best.pt (used by evaluate, explain, export and serve)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from deeptrace.models.factory import ModelConfig, create_model


@dataclass
class LoadedModel:
    model: nn.Module
    config: ModelConfig
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    meta: dict[str, Any]


def load_trained(path: str | Path, device: str = "cpu") -> LoadedModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model_config" not in ck:
        raise ValueError(f"{path} is not a best.pt checkpoint (no model_config)")
    cfg = ModelConfig(**{**ck["model_config"], "pretrained": False})
    model = create_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    meta = {k: ck.get(k) for k in ("epoch", "val_auc", "run_name", "git_commit")}
    return LoadedModel(model, cfg, tuple(ck["mean"]), tuple(ck["std"]), meta)
