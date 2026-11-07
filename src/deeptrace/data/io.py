"""Safe image decoding shared by data preparation and serving.

Every image goes through `load_image`, so real and fake images (and API uploads)
are decoded identically: EXIF orientation applied, pixel count capped, and any
colour format converted to 8-bit RGB.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MAX_PIXELS = 40_000_000  # ~6300x6300; guards against decompression bombs

# libjpeg's standard luminance quantization table (quality 50).
_STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64,
)  # fmt: skip


class ImageLoadError(ValueError):
    """Raised for files that cannot be decoded safely. `reason` is a short code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class ImageMeta:
    width: int
    height: int
    format: str | None
    mode: str
    jpeg_quality: int | None


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Estimate the libjpeg quality setting from the luminance quantization table."""
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    actual = float(np.sum(np.asarray(tables[0], dtype=np.float64)))
    best_q, best_err = None, float("inf")
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        table = np.clip(np.floor((_STD_LUMA * scale + 50) / 100), 1, 255)
        err = abs(float(table.sum()) - actual)
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def to_rgb_array(img: Image.Image) -> np.ndarray:
    """Convert any PIL mode (L, LA, P, RGBA, CMYK, 16-bit, float) to HxWx3 uint8 RGB."""
    mode = img.mode
    if mode.startswith("I") or mode == "F":
        arr = np.asarray(img).astype(np.float64)
        if mode.startswith("I;16") or (mode == "I" and arr.max(initial=0) > 255):
            arr = arr / 257.0  # 0..65535 -> 0..255
        gray = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.stack([gray] * 3, axis=-1))
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    rgb = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))


def load_image(
    src: str | Path | bytes, max_pixels: int = DEFAULT_MAX_PIXELS
) -> tuple[np.ndarray, ImageMeta]:
    """Decode an image file or bytes into RGB uint8. Raises ImageLoadError."""
    if isinstance(src, (bytes, bytearray)):
        if len(src) == 0:
            raise ImageLoadError("empty_file")
        fp = io.BytesIO(src)
    else:
        path = Path(src)
        try:
            if path.stat().st_size == 0:
                raise ImageLoadError("empty_file")
            fp = path.open("rb")
        except OSError as exc:
            raise ImageLoadError("unreadable_file", str(exc)) from exc
    try:
        with Image.open(fp) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise ImageLoadError("too_many_pixels", f"{width}x{height}")
            fmt = img.format
            quality = estimate_jpeg_quality(img) if fmt == "JPEG" else None
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)  # animated GIF/WebP: first frame only
            img.load()
            mode = img.mode
            upright = ImageOps.exif_transpose(img)
            rgb = to_rgb_array(upright)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ImageLoadError("corrupt_image", str(exc)) from exc
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("too_many_pixels", str(exc)) from exc
    finally:
        fp.close()
    return rgb, ImageMeta(width, height, fmt, mode, quality)
