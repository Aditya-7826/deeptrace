"""Grad-CAM: target layers per architecture, ViT reshape, mapping and sanity checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeptrace.explain.gradcam import (
    Explainer,
    cam_to_original,
    overlay,
    target_layers,
    vit_reshape_transform,
)
from deeptrace.explain.sanity import energy_shares, randomization_test, region_masks, spearman
from deeptrace.models.factory import ModelConfig, create_model


@pytest.mark.parametrize(
    ("name", "arch"),
    [("test_efficientnet", "timm"), ("test_resnet", "timm"), ("test_vit", "timm"),
     ("test_efficientnet", "rgb_fft")],
)  # fmt: skip
def test_gradcam_shape_and_range(name, arch):
    torch.manual_seed(0)
    model = create_model(ModelConfig(name=name, arch=arch, pretrained=False)).eval()
    x = torch.randn(3, 3, 224, 224)
    for target in ("fake", "real", "predicted"):
        cams = Explainer(model)(x, target)
        assert cams.shape == (3, 224, 224) and cams.dtype == np.float32
        assert cams.min() >= 0.0 and cams.max() <= 1.0
        assert np.isfinite(cams).all()


def test_target_layer_choices():
    eff = create_model(ModelConfig(name="test_efficientnet", pretrained=False))
    res = create_model(ModelConfig(name="test_resnet", pretrained=False))
    vit = create_model(ModelConfig(name="test_vit", pretrained=False))
    assert target_layers(eff) == [eff.bn2]
    assert target_layers(res) == [res.layer4[-1]]
    assert target_layers(vit) == [vit.blocks[-1].norm1]


def test_vit_reshape_drops_cls_token():
    grid, c = (14, 14), 8
    patches = torch.arange(196 * c, dtype=torch.float32).reshape(1, 196, c)
    cls = torch.full((1, 1, c), -999.0)  # patch values are all >= 0
    out = vit_reshape_transform(grid, 1)(torch.cat([cls, patches], dim=1))
    assert out.shape == (1, c, 14, 14)
    assert not (out == -999.0).any()  # CLS excluded
    # patch k (row-major) lands at (k // 14, k % 14)
    assert torch.equal(out[0, :, 0, 1], patches[0, 1]) and torch.equal(
        out[0, :, 1, 0], patches[0, 14]
    )
    with pytest.raises(ValueError):  # wrong prefix count is caught, not silently misaligned
        vit_reshape_transform(grid, 2)(torch.cat([cls, patches], dim=1))


def test_vit_reshape_multiple_prefix_tokens():
    tokens = torch.cat([torch.full((2, 5, 4), -1.0), torch.ones(2, 49, 4)], dim=1)
    out = vit_reshape_transform((7, 7), 5)(tokens)
    assert out.shape == (2, 4, 7, 7) and (out == 1).all()


def test_cam_maps_back_to_original_coordinates():
    cam = np.ones((224, 224), np.float32)
    full = cam_to_original(cam, (10, 20, 50), (100, 200))
    assert full.shape == (100, 200)
    assert full[20:70, 10:60].min() == 1.0 and full.sum() == pytest.approx(50 * 50)
    # crop square partly outside the image (padding) is clipped
    edge = cam_to_original(cam, (-10, -10, 40), (100, 100))
    assert edge.sum() == pytest.approx(30 * 30)
    assert overlay(np.zeros((50, 50, 3), np.uint8), cam).shape == (50, 50, 3)


def test_energy_shares_flags_background_maps():
    face, border = region_masks(224)
    centred = np.zeros((5, 224, 224), np.float32)
    centred[:, 80:150, 80:150] = 1.0
    ring = np.zeros((5, 224, 224), np.float32)
    ring[:, border] = 1.0
    good, bad = energy_shares(centred), energy_shares(ring)
    assert good["mean_face_share"] == pytest.approx(1.0) and not good["flagged_shortcut_risk"]
    assert bad["mean_face_share"] == 0.0 and bad["flagged_shortcut_risk"]
    assert 0.55 < good["uniform_map_face_share"] < 0.62  # (1/1.3)^2


def test_randomization_test():
    rng = np.random.default_rng(0)
    a = rng.random((6, 32, 32)).astype(np.float32)
    assert not randomization_test(a, a.copy())["passed"]  # identical maps: fails
    assert randomization_test(a, rng.random((6, 32, 32)).astype(np.float32))["passed"]
    assert spearman(a[0], a[0]) == pytest.approx(1.0)
