"""Model construction, parameter groups for discriminative LRs, and backbone freezing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import timm
from torch import nn

from deeptrace.models.two_branch import RGBFFTNet


@dataclass
class ModelConfig:
    name: str = "efficientnet_b0"
    arch: str = "timm"  # "timm" or "rgb_fft"
    pretrained: bool = True
    drop_rate: float = 0.2
    img_size: int = 224


def create_model(cfg: ModelConfig) -> nn.Module:
    """Binary classifier with a single-logit head (logit > 0 means fake)."""
    if cfg.arch == "rgb_fft":
        model = RGBFFTNet(cfg.name, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate)
    elif cfg.arch == "timm":
        kwargs = {"img_size": cfg.img_size} if "vit" in cfg.name else {}
        model = timm.create_model(
            cfg.name, pretrained=cfg.pretrained, num_classes=1, drop_rate=cfg.drop_rate, **kwargs
        )
    else:
        raise ValueError(f"unknown arch {cfg.arch!r}")
    init_logit_layers(model)
    return model


def init_logit_layers(model: nn.Module, std: float = 0.01) -> None:
    """Start every single-logit output layer near zero.

    timm's EfficientNet init draws the classifier from U(-1/sqrt(fan_out), ...), which is
    U(-1, 1) for one output: over 1280 pretrained features that gives initial logits large
    enough to push the starting loss to about 8 instead of about 0.7.
    """
    for head in head_modules(model):
        for m in head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.normal_(m.weight, std=std)
                nn.init.zeros_(m.bias)


def normalization(model: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(
        cfg.get("std", (0.229, 0.224, 0.225))
    )


def head_modules(model: nn.Module) -> list[nn.Module]:
    """Newly initialised parts trained from the start (and at the higher LR)."""
    if isinstance(model, RGBFFTNet):
        return [model.head, model.fft]
    return [model.get_classifier()]


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(backbone_params, head_params)."""
    head_ids = {id(p) for m in head_modules(model) for p in m.parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the backbone. Frozen backbone norm layers are kept in eval mode
    so their running statistics don't drift during head-only training."""
    backbone, _ = split_parameters(model)
    for p in backbone:
        p.requires_grad_(trainable)
    if not trainable:
        head_ids = {id(m) for h in head_modules(model) for m in h.modules()}
        for m in model.modules():
            if id(m) not in head_ids and isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def model_config_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)
