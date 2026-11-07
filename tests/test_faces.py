import numpy as np
import pytest

from deeptrace.data.faces import Face, crop_box, crop_face, filter_faces, full_frame, select_face
from deeptrace.data.io import load_image
from deeptrace.data.preprocess import CropSettings, decode_jpeg, extract_crops, standardize
from tests.conftest import IMAGES


def test_filter_rules():
    big = Face((0, 0, 100, 120), 0.99)
    small = Face((0, 0, 40, 50), 0.99)
    unsure = Face((0, 0, 200, 200), 0.80)
    assert filter_faces([]) == ([], "no_face")
    assert filter_faces([unsure]) == ([], "low_confidence")
    assert filter_faces([small]) == ([], "face_too_small")
    assert select_face([small, big, unsure]) == (big, None)


def test_crop_margin_and_padding():
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    left, top, side = crop_box((40, 40, 60, 60), margin=1.3)
    assert side == 26 and (left, top) == (37, 37)
    crop, pad = crop_face(rgb, Face((40, 40, 60, 60), 1.0), size=224)
    assert crop.shape == (224, 224, 3) and pad == 0.0
    # Face at the corner: the 1.3x square runs off the image and is padded black
    crop, pad = crop_face(rgb, Face((0, 0, 50, 50), 1.0), size=224)
    assert 0 < pad < 0.5 and crop[0, 0].sum() == 0


def test_full_frame_ablation_keeps_whole_image():
    rgb = np.full((100, 200, 3), 50, dtype=np.uint8)
    out, pad = full_frame(rgb, 224)
    assert out.shape == (224, 224, 3) and pad == pytest.approx(0.5)


@pytest.mark.slow
def test_detect_portrait(detector):
    rgb, _ = load_image(IMAGES / "real/aldrin.jpg")
    crops, reason = extract_crops(rgb, detector)
    assert reason is None and len(crops) == 1
    pixels = decode_jpeg(crops[0].jpeg)
    assert pixels.shape == (224, 224, 3)
    np.testing.assert_array_equal(standardize(pixels), standardize(pixels))  # deterministic


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "reason"),
    [("real/noface.jpg", "no_face"), ("fake/tinyface.jpg", "face_too_small")],
)
def test_rejections(detector, name, reason):
    rgb, _ = load_image(IMAGES / name)
    assert extract_crops(rgb, detector) == ([], reason)


@pytest.mark.slow
def test_multiface_keeps_largest(detector):
    rgb, _ = load_image(IMAGES / "real/multiface.jpg")
    all_faces, _ = filter_faces(detector.detect(rgb), min_size=0)
    assert len(all_faces) >= 2
    crops, _ = extract_crops(rgb, detector, CropSettings(), largest_only=True)
    assert len(crops) == 1 and crops[0].face.box[0] < 300  # the big face is on the left


@pytest.mark.slow
def test_exif_rotation_matters_for_detection(detector):
    rgb, _ = load_image(IMAGES / "fake/armstrong_exif_rotated.jpg")
    faces, reason = filter_faces(detector.detect(rgb))
    assert reason is None
    face = faces[0]
    assert face.height >= face.width * 0.9  # upright face: not lying on its side
