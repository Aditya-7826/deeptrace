"""Training-only augmentations. Validation and test use none."""

from __future__ import annotations

import albumentations as A
import cv2


def build_train_augment(enabled: bool = True) -> A.Compose | None:
    if not enabled:
        return None
    return A.Compose(
        [
            A.ImageCompression(quality_range=(50, 100), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),  # downscale then upscale back
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        ]
    )
