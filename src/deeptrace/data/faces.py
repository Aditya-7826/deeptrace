"""Face detection (MTCNN) and margin cropping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_CONFIDENCE = 0.90
MIN_FACE_PX = 64
MARGIN = 1.3


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    confidence: float

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


class FaceDetector:
    """Thin MTCNN wrapper. Large images are downscaled for detection only."""

    def __init__(self, device: str = "cpu", max_side: int = 1024) -> None:
        from facenet_pytorch import MTCNN

        self.device = device
        self.max_side = max_side
        self._mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    def detect(self, rgb: np.ndarray) -> list[Face]:
        from PIL import Image

        h, w = rgb.shape[:2]
        scale = 1.0
        small = rgb
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            new_size = (round(w * scale), round(h * scale))
            small = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        boxes, probs = self._mtcnn.detect(Image.fromarray(small))
        if boxes is None:
            return []
        faces = []
        for box, prob in zip(boxes, probs, strict=True):
            if prob is None:
                continue
            x1, y1, x2, y2 = (float(v) / scale for v in box)
            faces.append(Face((x1, y1, x2, y2), float(prob)))
        return faces


def filter_faces(
    faces: list[Face], min_confidence: float = MIN_CONFIDENCE, min_size: int = MIN_FACE_PX
) -> tuple[list[Face], str | None]:
    """Keep confident, large-enough faces, largest first.

    Returns (faces, reason). `reason` explains an empty result: no_face,
    low_confidence or face_too_small.
    """
    if not faces:
        return [], "no_face"
    confident = [f for f in faces if f.confidence >= min_confidence]
    if not confident:
        return [], "low_confidence"
    big = [f for f in confident if min(f.width, f.height) >= min_size]
    if not big:
        return [], "face_too_small"
    return sorted(big, key=lambda f: f.area, reverse=True), None


def select_face(
    faces: list[Face], min_confidence: float = MIN_CONFIDENCE, min_size: int = MIN_FACE_PX
) -> tuple[Face | None, str | None]:
    """Training rule: keep the largest valid face."""
    kept, reason = filter_faces(faces, min_confidence, min_size)
    return (kept[0], None) if kept else (None, reason)


def _square_region(rgb: np.ndarray, left: int, top: int, side: int) -> tuple[np.ndarray, float]:
    """Cut a square, padding with black outside the image. Returns (region, pad_frac)."""
    h, w = rgb.shape[:2]
    right, bottom = left + side, top + side
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(right, w), min(bottom, h)
    region = rgb[y1:y2, x1:x2]
    valid = max(x2 - x1, 0) * max(y2 - y1, 0)
    pads = (max(0, -top), max(0, bottom - h), max(0, -left), max(0, right - w))
    if any(pads):
        if region.size == 0:
            region = np.zeros((side, side, 3), dtype=np.uint8)
        else:
            region = cv2.copyMakeBorder(region, *pads, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return region, 1.0 - valid / float(side * side)


def _resize_square(region: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if region.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(region, (size, size), interpolation=interp)


def crop_box(
    box: tuple[float, float, float, float], margin: float = MARGIN
) -> tuple[int, int, int]:
    """Square crop (left, top, side) centred on the face box, side = margin * longest edge."""
    x1, y1, x2, y2 = box
    side = max(int(round(max(x2 - x1, y2 - y1) * margin)), 1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def crop_face(
    rgb: np.ndarray, face: Face, margin: float = MARGIN, size: int = 224
) -> tuple[np.ndarray, float]:
    left, top, side = crop_box(face.box, margin)
    region, pad_frac = _square_region(rgb, left, top, side)
    return _resize_square(region, size), pad_frac


def full_frame(rgb: np.ndarray, size: int = 224) -> tuple[np.ndarray, float]:
    """Ablation 'no face cropping': pad the whole frame to a centred square and resize."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    region, pad_frac = _square_region(rgb, (w - side) // 2, (h - side) // 2, side)
    return _resize_square(region, size), pad_frac
