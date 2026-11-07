"""Build the tiny test fixture set from public-domain portraits.

Usage (PowerShell or bash):
    uv run python scripts/make_fixtures.py --portraits <folder with the 8 .jpg files>

The portraits are U.S. government (NASA / U.S. Senate) works in the public
domain, downloaded from Wikimedia Commons (see tests/fixtures/SOURCES.md).
Labels in the fixture are arbitrary: they only exercise the pipeline.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
NAMES = ["aldrin", "armstrong", "glenn", "hadfield", "jemison", "ochoa", "ride", "whitson"]


def load(portraits: Path, name: str, width: int = 520) -> Image.Image:
    img = Image.open(portraits / f"{name}.jpg").convert("RGB")
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def save_jpeg(img: Image.Image, path: Path, quality: int = 88, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, **kw)


def write_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def jitter_frames(img: Image.Image, detector, n: int = 12, size: int = 224) -> list[np.ndarray]:
    """Square clip around the detected face (face ~ 50% of the frame), with small pans."""
    arr = np.asarray(img)
    face = max(detector.detect(arr), key=lambda f: f.area)
    side = int(max(face.width, face.height) * 2.0)
    cx, cy = int((face.box[0] + face.box[2]) / 2), int((face.box[1] + face.box[3]) / 2)
    frames = []
    for k in range(n):
        dx = int(6 * np.sin(k / 2))
        x0 = max(0, min(cx - side // 2 + dx, arr.shape[1] - side))
        y0 = max(0, min(cy - side // 2, arr.shape[0] - side))
        clip = arr[y0 : y0 + side, x0 : x0 + side]
        frames.append(cv2.resize(clip, (size, size), interpolation=cv2.INTER_AREA))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits", type=Path, required=True)
    args = ap.parse_args()
    p = {n: load(args.portraits, n) for n in NAMES}
    img_real, img_fake = OUT / "images" / "real", OUT / "images" / "fake"

    # Plain images
    for n in ("aldrin", "armstrong", "hadfield"):
        save_jpeg(p[n], img_real / f"{n}.jpg")
    for n in ("ochoa", "whitson", "ride"):
        save_jpeg(p[n], img_fake / f"{n}.jpg")
    # Near-duplicate of ride (recompressed, slightly resized): must land in ride's split
    near = p["ride"].resize((500, round(p["ride"].height * 500 / 520)))
    save_jpeg(near, img_fake / "ride_recompressed.jpg", quality=70)

    # Colour formats
    rgba = p["jemison"].convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 200))
    rgba.save(img_real / "jemison_rgba.png")
    gray16 = np.asarray(p["glenn"].convert("L"), dtype=np.uint16) * 257
    Image.fromarray(gray16).save(img_real / "glenn_gray16.png")  # PIL writes I;16 PNG
    save_jpeg(p["hadfield"].convert("L"), img_fake / "hadfield_gray.jpg")
    save_jpeg(p["aldrin"].convert("CMYK"), img_fake / "aldrin_cmyk.jpg")

    # EXIF orientation 6: pixels stored rotated, viewer must rotate 90 deg clockwise
    rotated = p["armstrong"].transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    save_jpeg(rotated, img_fake / "armstrong_exif_rotated.jpg", exif=exif.tobytes())

    # Multi-face: big glenn left, small whitson right
    canvas = Image.new("RGB", (520, 400), (120, 120, 120))
    canvas.paste(p["glenn"].resize((300, 381)).crop((0, 0, 300, 381)), (0, 10))
    canvas.paste(p["whitson"].resize((150, 150)), (340, 120))
    save_jpeg(canvas, img_real / "multiface.jpg")

    # Tiny face (~35 px): detectable but below the 64 px minimum
    tiny = Image.new("RGB", (500, 500), (90, 110, 130))
    tiny.paste(p["ochoa"].resize((240, 300)), (130, 100))
    save_jpeg(tiny, img_fake / "tinyface.jpg")

    # No face: smooth gradient + noise
    rng = np.random.default_rng(0)
    grad = np.linspace(0, 255, 300, dtype=np.float32)[None, :, None].repeat(300, 0).repeat(3, 2)
    noise = rng.normal(0, 12, grad.shape)
    noface = np.clip(grad + noise, 0, 255).astype(np.uint8)
    save_jpeg(Image.fromarray(noface), img_real / "noface.jpg")

    # Corrupt (truncated JPEG) and zero-byte
    buf = io.BytesIO()
    p["aldrin"].save(buf, "JPEG", quality=88)
    (img_real / "corrupt.jpg").write_bytes(buf.getvalue()[: len(buf.getvalue()) // 3])
    (img_fake / "empty.jpg").write_bytes(b"")

    # Mini FaceForensics++ layout: 4 originals, 2 manipulations, 1 broken video
    ff = OUT / "ffpp"
    orig = ff / "original_sequences" / "youtube" / "c23" / "videos"
    df = ff / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    from deeptrace.data.faces import FaceDetector

    det = FaceDetector()
    for vid, name in (("000", "aldrin"), ("001", "hadfield"), ("002", "ochoa"), ("003", "jemison")):
        write_video(jitter_frames(p[name], det), orig / f"{vid}.mp4")
    write_video(jitter_frames(p["armstrong"], det), df / "000_001.mp4")
    write_video(jitter_frames(p["ride"], det), df / "002_003.mp4")
    (orig / "999.mp4").write_bytes(b"not a video" * 10)
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
