"""The single preprocessing path used by data preparation, training and serving.

decode -> detect -> crop (1.3x margin, square) -> resize 224 -> JPEG q95 re-encode.
Serving runs the same re-encode in memory, so the model sees identical pixels
whether a crop came from disk (training) or from an upload (API).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from deeptrace.data.faces import (
    MARGIN,
    MIN_CONFIDENCE,
    MIN_FACE_PX,
    Face,
    FaceDetector,
    crop_face,
    filter_faces,
    full_frame,
)

CROP_SIZE = 224
JPEG_QUALITY = 95


@dataclass(frozen=True)
class CropSettings:
    margin: float = MARGIN
    size: int = CROP_SIZE
    jpeg_quality: int = JPEG_QUALITY
    min_confidence: float = MIN_CONFIDENCE
    min_face_px: int = MIN_FACE_PX
    mode: str = "face"  # "face" or "full_frame" (ablation)


@dataclass(frozen=True)
class FaceCrop:
    face: Face
    jpeg: bytes  # standardized crop, exactly what is written to disk
    pad_frac: float


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, subsampling=0, optimize=False)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def standardize(rgb_crop: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """In-memory equivalent of 'save crop to disk and read it back'."""
    return decode_jpeg(encode_jpeg(rgb_crop, quality))


def extract_crops(
    rgb: np.ndarray,
    detector: FaceDetector,
    settings: CropSettings = CropSettings(),
    largest_only: bool = True,
) -> tuple[list[FaceCrop], str | None]:
    """Detect faces and return standardized crops (largest first) or a skip reason."""
    faces, reason = filter_faces(detector.detect(rgb), settings.min_confidence, settings.min_face_px)
    if not faces:
        return [], reason
    if largest_only:
        faces = faces[:1]
    crops = []
    for face in faces:
        if settings.mode == "full_frame":
            pixels, pad = full_frame(rgb, settings.size)
        else:
            pixels, pad = crop_face(rgb, face, settings.margin, settings.size)
        crops.append(FaceCrop(face, encode_jpeg(pixels, settings.jpeg_quality), pad))
    return crops, None
