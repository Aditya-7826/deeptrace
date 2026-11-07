"""Deterministic perturbations applied to test crops for robustness curves.

They are applied to the 224x224 face crop (the originals are not kept after
`prepare`), which approximates degradation of the source image. Each one is a
small class, not a lambda, so it pickles into Windows (spawn) DataLoader workers.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from deeptrace.data.preprocess import decode_jpeg, encode_jpeg


class Jpeg:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return decode_jpeg(encode_jpeg(rgb, self.quality))


class Downscale:
    """Downscale by `factor`, then upscale back to the model input size."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        size = (max(1, round(w * self.factor)), max(1, round(h * self.factor)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class Blur:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(rgb, (0, 0), sigmaX=self.sigma)


class Screenshot:
    """Simulated screenshot: display rescale (bilinear), lossless PNG re-encode, back to size."""

    def __init__(self, scale: float = 0.8) -> None:
        self.scale = scale

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        size = (round(w * self.scale), round(h * self.scale))
        shown = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        buf = io.BytesIO()
        Image.fromarray(shown).save(buf, format="PNG")
        with Image.open(io.BytesIO(buf.getvalue())) as img:
            back = np.asarray(img.convert("RGB"))
        return cv2.resize(back, (w, h), interpolation=cv2.INTER_LINEAR)


# family -> list of (level label, numeric level, function)
PERTURBATIONS: dict[str, list[tuple[str, float, Callable[[np.ndarray], np.ndarray]]]] = {
    "jpeg_quality": [(f"q{q}", q, Jpeg(q)) for q in (90, 70, 50, 30)],
    "downscale": [(f"x{s}", s, Downscale(s)) for s in (0.75, 0.5, 0.25)],
    "gaussian_blur": [(f"sigma{s}", s, Blur(s)) for s in (1.0, 2.0, 3.0)],
    "screenshot": [("resize0.8+png", 0.8, Screenshot(0.8))],
}


class Perturb:
    """Adapter so a perturbation can be used as a CropDataset transform."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.fn = fn

    def set_random_seed(self, seed: int) -> None:  # deterministic: nothing to seed
        pass

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {"image": self.fn(image)}
