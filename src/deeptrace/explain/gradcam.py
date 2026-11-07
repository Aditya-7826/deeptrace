"""Grad-CAM for every supported architecture, and mapping heatmaps back to the original image."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch import nn

from deeptrace.models.two_branch import RGBFFTNet


def rgb_backbone(model: nn.Module) -> nn.Module:
    return model.rgb if isinstance(model, RGBFFTNet) else model


def target_layers(model: nn.Module) -> list[nn.Module]:
    """The last spatial feature layer of each architecture."""
    m = rgb_backbone(model)
    if hasattr(m, "conv_head") and hasattr(m, "bn2"):  # EfficientNet: conv_head + BN + act
        return [m.bn2]
    if hasattr(m, "layer4"):  # ResNet
        return [m.layer4[-1]]
    if hasattr(m, "blocks") and hasattr(m, "patch_embed"):  # ViT: tokens before the last attention
        return [m.blocks[-1].norm1]
    raise ValueError(f"no Grad-CAM target layer known for {type(m).__name__}")


def vit_reshape_transform(
    grid: tuple[int, int], num_prefix_tokens: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
    """B x (prefix + H*W) x C tokens -> B x C x H x W, dropping CLS and other prefix tokens."""
    h, w = grid

    def transform(tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, num_prefix_tokens:, :]
        b, n, c = patches.shape
        if n != h * w:
            raise ValueError(f"expected {h * w} patch tokens, got {n}")
        return patches.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    return transform


def reshape_transform_for(model: nn.Module):
    m = rgb_backbone(model)
    if hasattr(m, "patch_embed") and hasattr(m, "num_prefix_tokens"):
        return vit_reshape_transform(tuple(m.patch_embed.grid_size), int(m.num_prefix_tokens))
    return None


class Explainer:
    """Grad-CAM heatmaps in [0, 1], one per input image (H x W of the input)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        self.cam = GradCAM(model=model, target_layers=target_layers(model),
                           reshape_transform=reshape_transform_for(model))  # fmt: skip

    def __call__(self, x: torch.Tensor, target: str = "predicted") -> np.ndarray:
        if target == "predicted":
            with torch.no_grad():
                cats = (self.model(x).squeeze(1) > 0).long().tolist()
        else:
            cats = [1 if target == "fake" else 0] * len(x)
        maps = self.cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(c) for c in cats])
        return np.clip(np.nan_to_num(maps.astype(np.float32)), 0.0, 1.0)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Heatmap over an RGB uint8 image (cam is resized to the image if needed)."""
    if cam.shape != rgb.shape[:2]:
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return show_cam_on_image(
        rgb.astype(np.float32) / 255.0, cam, use_rgb=True, image_weight=1 - alpha
    )


def cam_to_original(
    cam: np.ndarray, crop_box: tuple[int, int, int], image_shape: tuple[int, ...]
) -> np.ndarray:
    """Place a crop's heatmap at its (left, top, side) square in the original image.
    Parts of the square outside the image (padding) are dropped; elsewhere is 0."""
    left, top, side = crop_box
    h, w = image_shape[:2]
    full = np.zeros((h, w), dtype=np.float32)
    big = cv2.resize(cam.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(left + side, w), min(top + side, h)
    if x2 > x1 and y2 > y1:
        full[y1:y2, x1:x2] = big[y1 - top : y2 - top, x1 - left : x2 - left]
    return full


def overlay_on_original(
    rgb: np.ndarray, heat: np.ndarray, crop_box: tuple[int, int, int], alpha: float = 0.45
) -> np.ndarray:
    """Blend the heatmap only inside the crop square and draw the square."""
    out = rgb.copy()
    left, top, side = crop_box
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(left + side, w), min(top + side, h)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = overlay(rgb[y1:y2, x1:x2], heat[y1:y2, x1:x2], alpha)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), max(1, side // 150))
    return out
