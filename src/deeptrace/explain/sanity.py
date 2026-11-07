"""Grad-CAM sanity checks.

1. Model-randomization test (Adebayo et al., 2018): heatmaps from the trained
   model must differ substantially from those of a randomly initialised model
   of the same architecture. Similar maps mean the explanation reflects the
   input/architecture, not what the model learned.
2. Energy location: the share of heatmap mass inside the face box vs on the
   background and crop borders. Mostly-background energy suggests shortcut learning.
"""

from __future__ import annotations

import numpy as np

from deeptrace.data.faces import MARGIN

MAX_RANDOMIZATION_SPEARMAN = 0.5  # pass if mean |rank correlation| is below this
MIN_FACE_SHARE = 0.5  # flag if less than half the energy is inside the face box
BORDER_FRAC = 0.08  # outer ring of the crop counted as "border"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def randomization_test(trained: np.ndarray, randomized: np.ndarray) -> dict:
    """trained/randomized: N x H x W heatmaps for the same N inputs."""
    corrs = np.array([spearman(a, b) for a, b in zip(trained, randomized, strict=True)])
    valid = corrs[~np.isnan(corrs)]
    mean_abs = float(np.abs(valid).mean()) if len(valid) else float("nan")
    return {
        "n": int(len(corrs)),
        "n_constant_maps": int(np.isnan(corrs).sum()),
        "mean_spearman": float(valid.mean()) if len(valid) else float("nan"),
        "mean_abs_spearman": mean_abs,
        "mean_abs_diff": float(np.abs(trained - randomized).mean()),
        "pass_threshold": MAX_RANDOMIZATION_SPEARMAN,
        "passed": bool(len(valid) and mean_abs < MAX_RANDOMIZATION_SPEARMAN),
    }


def region_masks(size: int, margin: float = MARGIN, border_frac: float = BORDER_FRAC):
    """Boolean masks (face, border) for a size x size crop made with `margin`."""
    lo = int(round(size * (1 - 1 / margin) / 2))
    face = np.zeros((size, size), bool)
    face[lo : size - lo, lo : size - lo] = True
    b = max(1, int(round(size * border_frac)))
    border = np.ones((size, size), bool)
    border[b : size - b, b : size - b] = False
    return face, border


def energy_shares(cams: np.ndarray, margin: float = MARGIN) -> dict:
    """Mean share of heatmap mass inside the face box / on the crop border / elsewhere."""
    size = cams.shape[-1]
    face, border = region_masks(size, margin)
    background = ~face & ~border
    totals = cams.reshape(len(cams), -1).sum(axis=1)
    ok = totals > 0
    c = cams[ok]
    t = totals[ok]
    face_share = (c[:, face].sum(axis=1) / t) if ok.any() else np.array([])
    border_share = (c[:, border].sum(axis=1) / t) if ok.any() else np.array([])
    bg_share = (c[:, background].sum(axis=1) / t) if ok.any() else np.array([])
    mean_face = float(face_share.mean()) if len(face_share) else float("nan")
    return {
        "n": int(len(cams)),
        "n_empty_maps": int((~ok).sum()),
        "mean_face_share": mean_face,
        "mean_border_share": float(border_share.mean()) if len(border_share) else float("nan"),
        "mean_background_share": float(bg_share.mean()) if len(bg_share) else float("nan"),
        "frac_images_mostly_outside_face": float((face_share < 0.5).mean())
        if len(face_share) else float("nan"),
        "uniform_map_face_share": float(face.mean()),  # what a featureless map would give
        "flag_threshold": MIN_FACE_SHARE,
        "flagged_shortcut_risk": bool(len(face_share) and mean_face < MIN_FACE_SHARE),
    }  # fmt: skip
