"""Per-class source statistics that could leak the label (resolution, JPEG quality, padding)."""

from __future__ import annotations

import statistics
from collections import defaultdict

RESOLUTION_REL_DIFF = 0.25
JPEG_QUALITY_DIFF = 10
FRACTION_DIFF = 0.20
PAD_DIFF = 0.10


def _num(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def class_stats(rows: list[dict]) -> dict:
    stats: dict = {}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    for label, items in sorted(by_class.items()):
        res = [(_num(r["orig_width"]) * _num(r["orig_height"])) ** 0.5 for r in items]
        qualities = [q for r in items if (q := _num(r["jpeg_quality"])) is not None]
        pads = [_num(r["pad_frac"]) or 0.0 for r in items]
        stats[label] = {
            "count": len(items),
            "median_resolution": statistics.median(res) if res else None,
            "jpeg_fraction": len(qualities) / len(items) if items else None,
            "median_jpeg_quality": statistics.median(qualities) if qualities else None,
            "mean_pad_frac": statistics.fmean(pads) if pads else None,
        }
    return stats


def shortcut_warnings(stats: dict, dataset: str) -> list[str]:
    if "real" not in stats or "fake" not in stats:
        return []
    r, f = stats["real"], stats["fake"]
    out = []
    if r["median_resolution"] and f["median_resolution"]:
        rel = abs(r["median_resolution"] - f["median_resolution"]) / max(
            r["median_resolution"], f["median_resolution"]
        )
        if rel > RESOLUTION_REL_DIFF:
            out.append(
                f"{dataset}: original resolution differs by class "
                f"(real median {r['median_resolution']:.0f}px, fake {f['median_resolution']:.0f}px)"
            )
    if abs((r["jpeg_fraction"] or 0) - (f["jpeg_fraction"] or 0)) > FRACTION_DIFF:
        out.append(
            f"{dataset}: share of JPEG sources differs by class "
            f"(real {r['jpeg_fraction']:.2f}, fake {f['jpeg_fraction']:.2f})"
        )
    if r["median_jpeg_quality"] is not None and f["median_jpeg_quality"] is not None:
        if abs(r["median_jpeg_quality"] - f["median_jpeg_quality"]) > JPEG_QUALITY_DIFF:
            out.append(
                f"{dataset}: JPEG quality differs by class "
                f"(real median {r['median_jpeg_quality']:.0f}, fake {f['median_jpeg_quality']:.0f})"
            )
    if abs((r["mean_pad_frac"] or 0) - (f["mean_pad_frac"] or 0)) > PAD_DIFF:
        out.append(
            f"{dataset}: crop border padding differs by class "
            f"(real {r['mean_pad_frac']:.2f}, fake {f['mean_pad_frac']:.2f})"
        )
    return out
