import io

import numpy as np
import pytest
from PIL import Image

from deeptrace.data.io import ImageLoadError, load_image, to_rgb_array
from tests.conftest import IMAGES


@pytest.mark.parametrize(
    "name",
    [
        "real/aldrin.jpg",
        "real/jemison_rgba.png",
        "real/glenn_gray16.png",
        "fake/hadfield_gray.jpg",
        "fake/aldrin_cmyk.jpg",
    ],
)
def test_every_format_becomes_8bit_rgb(name):
    rgb, meta = load_image(IMAGES / name)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.flags["C_CONTIGUOUS"]


def test_16bit_is_scaled_not_clipped():
    rgb, meta = load_image(IMAGES / "real/glenn_gray16.png")
    assert meta.mode.startswith("I")
    assert rgb.max() > 200  # naive uint8 cast of 16-bit values would saturate or wrap
    assert len(np.unique(rgb)) > 50


def test_rgba_composited_on_white():
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert (to_rgb_array(img) == 255).all()


def test_exif_orientation_applied():
    rotated, meta = load_image(IMAGES / "fake/armstrong_exif_rotated.jpg")
    upright, _ = load_image(IMAGES / "real/armstrong.jpg")
    assert (meta.width, meta.height) == (upright.shape[0], upright.shape[1])  # stored sideways
    assert rotated.shape == upright.shape  # displayed upright


def test_jpeg_quality_estimate():
    _, meta = load_image(IMAGES / "fake/ride_recompressed.jpg")
    assert meta.format == "JPEG" and abs(meta.jpeg_quality - 70) <= 2
    _, png_meta = load_image(IMAGES / "real/jemison_rgba.png")
    assert png_meta.jpeg_quality is None


@pytest.mark.parametrize(
    ("src", "reason"),
    [
        (IMAGES / "real/corrupt.jpg", "corrupt_image"),
        (IMAGES / "fake/empty.jpg", "empty_file"),
        (b"", "empty_file"),
        (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "corrupt_image"),
    ],
)
def test_bad_inputs_raise_clear_reason(src, reason):
    with pytest.raises(ImageLoadError) as exc:
        load_image(src)
    assert exc.value.reason == reason


def test_pixel_cap_checked_before_decoding():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, "PNG")
    with pytest.raises(ImageLoadError) as exc:
        load_image(buf.getvalue(), max_pixels=100 * 100)
    assert exc.value.reason == "too_many_pixels"


def test_gif_first_frame():
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    rgb, _ = load_image(buf.getvalue())
    assert rgb[0, 0, 0] > 200 and rgb[0, 0, 2] < 50
