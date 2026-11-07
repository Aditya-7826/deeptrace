"""`deeptrace prepare`: raw datasets -> standardized face crops + manifest.csv.

Resumable: each source file's result rows are appended to prepare_journal.csv
only after the file is fully processed, so an interrupted run re-does at most
one file and skips everything already in the journal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deeptrace.data.faces import FaceDetector
from deeptrace.data.io import DEFAULT_MAX_PIXELS, ImageLoadError, load_image
from deeptrace.data.phash import phash, to_hex
from deeptrace.data.preprocess import CropSettings, decode_jpeg, extract_crops
from deeptrace.data.shortcuts import class_stats, shortcut_warnings
from deeptrace.data.sources import SourceItem, enumerate_dataset
from deeptrace.data.splits import (
    assign_groups,
    find_leaks,
    identity_groups,
    near_duplicate_groups,
    remove_cross_split_near_duplicates,
)
from deeptrace.data.video import VideoReadError, sample_frames
from deeptrace.utils import resolve_device, set_cpu_threads

LOG = logging.getLogger("deeptrace.prepare")

MANIFEST_FIELDS = [
    "path", "label", "source_dataset", "manipulation_method",
    "video_id", "identity_id", "split", "phash",
]  # fmt: skip
JOURNAL_FIELDS = [
    "source_key", "status", "reason", "path", "label", "source_dataset",
    "manipulation_method", "video_id", "identity_id", "phash", "frame_index",
    "orig_width", "orig_height", "jpeg_quality", "pad_frac", "det_conf",
]  # fmt: skip


@dataclass
class DatasetSpec:
    name: str
    type: str
    root: str
    options: dict[str, Any] = field(default_factory=dict)
    max_per_class: int | None = None


@dataclass
class PrepareConfig:
    datasets: list[DatasetSpec]
    output_dir: str = "data/crops"
    raw_root: str = "data/raw"
    frames_per_video: int = 15
    crop: CropSettings = field(default_factory=CropSettings)
    max_pixels: int = DEFAULT_MAX_PIXELS
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    near_dup_distance: int = 4
    device: str = "auto"
    threads: int = 2
    detect_max_side: int = 1024

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PrepareConfig:
        raw = dict(raw)
        datasets = [DatasetSpec(**d) for d in raw.pop("datasets")]
        crop = CropSettings(**raw.pop("crop", {}))
        if "split_fractions" in raw:
            raw["split_fractions"] = tuple(raw["split_fractions"])
        cfg = cls(datasets=datasets, crop=crop, **raw)
        if abs(sum(cfg.split_fractions) - 1.0) > 1e-6:
            raise ValueError("split_fractions must sum to 1")
        if not 5 <= cfg.frames_per_video <= 50:
            LOG.warning("frames_per_video=%d is outside the usual 10-20", cfg.frames_per_video)
        return cfg


def _crop_name(item: SourceItem, frame_index: int | None) -> str:
    digest = hashlib.sha1(item.rel.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(item.rel).stem)[:40]
    suffix = f"_f{frame_index:05d}" if frame_index is not None else ""
    return f"{stem}_{digest}{suffix}.jpg"


def _base_row(item: SourceItem) -> dict[str, Any]:
    return {
        "source_key": item.key,
        "label": item.label,
        "source_dataset": item.dataset,
        "manipulation_method": item.method,
        "video_id": item.video_id,
        "identity_id": item.identity_id,
    }


def _skip(item: SourceItem, reason: str, frame_index: int | None = None) -> dict[str, Any]:
    return {**_base_row(item), "status": "skipped", "reason": reason,
            "frame_index": "" if frame_index is None else frame_index}


def process_item(
    item: SourceItem, detector: FaceDetector, cfg: PrepareConfig, out_dir: Path
) -> list[dict[str, Any]]:
    """Decode one source (image or video), crop faces, write crops. Never raises on bad data."""
    if item.kind == "image":
        try:
            rgb, meta = load_image(item.path, cfg.max_pixels)
        except ImageLoadError as exc:
            LOG.info("skip %s: %s", item.key, exc)
            return [_skip(item, exc.reason)]
        frames = [(None, rgb, meta.width, meta.height, meta.jpeg_quality)]
    else:
        frames = None

    def video_frames():
        for idx, rgb in sample_frames(item.path, cfg.frames_per_video):
            yield idx, rgb, rgb.shape[1], rgb.shape[0], None

    rows: list[dict[str, Any]] = []
    crop_dir = out_dir / "crops" / item.dataset / item.label
    try:
        for frame_index, rgb, width, height, quality in frames or video_frames():
            crops, reason = extract_crops(rgb, detector, cfg.crop, largest_only=True)
            if not crops:
                rows.append(_skip(item, reason or "no_face", frame_index))
                continue
            crop = crops[0]
            crop_dir.mkdir(parents=True, exist_ok=True)
            dest = crop_dir / _crop_name(item, frame_index)
            dest.write_bytes(crop.jpeg)
            rows.append({
                **_base_row(item),
                "status": "ok",
                "reason": "",
                "path": dest.relative_to(out_dir).as_posix(),
                "phash": to_hex(phash(decode_jpeg(crop.jpeg))),
                "frame_index": "" if frame_index is None else frame_index,
                "orig_width": width,
                "orig_height": height,
                "jpeg_quality": "" if quality is None else quality,
                "pad_frac": round(crop.pad_frac, 4),
                "det_conf": round(crop.face.confidence, 4),
            })
    except VideoReadError as exc:
        LOG.info("skip %s: %s", item.key, exc)
        rows.append(_skip(item, exc.reason))
    except Exception as exc:  # last-resort guard: log and move on, never crash the run
        LOG.warning("skip %s: unexpected %s: %s", item.key, type(exc).__name__, exc)
        rows.append(_skip(item, "unexpected_error"))
    return rows


def _read_journal(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _subsample(items: list[SourceItem], max_per_class: int | None, seed: int) -> list[SourceItem]:
    if not max_per_class:
        return items
    by_label: dict[str, list[SourceItem]] = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)
    out = []
    for label in sorted(by_label):
        group = by_label[label]
        if len(group) > max_per_class:
            group = sorted(random.Random(seed).sample(group, max_per_class), key=lambda i: i.rel)
        out.extend(group)
    return out


def enumerate_all(cfg: PrepareConfig) -> list[SourceItem]:
    items: list[SourceItem] = []
    for spec in cfg.datasets:
        root = Path(spec.root)
        if not root.is_absolute():
            root = Path(cfg.raw_root) / root
        if not root.exists():
            raise FileNotFoundError(f"dataset {spec.name!r}: root not found: {root}")
        found = enumerate_dataset(spec.type, root, spec.name, **dict(spec.options))
        found = _subsample(found, spec.max_per_class, cfg.seed)
        LOG.info("dataset %s (%s): %d source files", spec.name, spec.type, len(found))
        items.extend(found)
    return items


def assign_splits(
    rows: list[dict[str, Any]], items: dict[str, SourceItem], cfg: PrepareConfig
) -> tuple[list[dict[str, Any]], Counter]:
    """Give every ok row a split, by dataset type. Returns (rows, dropped reasons)."""
    types = {spec.name: spec.type for spec in cfg.datasets}
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dataset[r["source_dataset"]].append(r)
    out, dropped = [], Counter()
    for name, drows in sorted(by_dataset.items()):
        kind = types[name]
        if kind == "celebdf":
            for r in drows:
                r["split"] = "test"
            out.extend(drows)
            continue
        fixed = [items[r["source_key"]].fixed_split for r in drows]
        if any(f is not None for f in fixed):  # official FF++ split files were given
            for r, f in zip(drows, fixed, strict=True):
                if f in ("train", "val", "test"):
                    r["split"] = f
                    out.append(r)
                else:
                    dropped["not_in_official_split"] += 1
            continue
        if kind == "ffpp":
            groups = identity_groups([r["identity_id"] for r in drows])
            strata = ["all"] * len(drows)
        else:
            groups = near_duplicate_groups([r["phash"] for r in drows], cfg.near_dup_distance)
            strata = [r["label"] for r in drows]
        # Stratify image datasets by label; a group takes its first member's stratum.
        group_stratum: dict[str, str] = {}
        for g, s in zip(groups, strata, strict=True):
            group_stratum.setdefault(g, s)
        sizes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for g in groups:
            sizes[group_stratum[g]][g] += 1
        mapping: dict[str, str] = {}
        for stratum, gsizes in sorted(sizes.items()):
            mapping.update(assign_groups(dict(gsizes), cfg.split_fractions, cfg.seed))
        for r, g in zip(drows, groups, strict=True):
            r["split"] = mapping[g]
        out.extend(drows)
    return out, dropped


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_prepare(cfg: PrepareConfig, limit: int | None = None) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cpu_threads(cfg.threads)
    journal_path = out_dir / "prepare_journal.csv"

    items = enumerate_all(cfg)
    by_key = {it.key: it for it in items}
    done = {r["source_key"] for r in _read_journal(journal_path)}
    pending = [it for it in items if it.key not in done]
    if limit is not None:
        pending = pending[:limit]
    LOG.info("%d already processed, %d to process", len(items) - len(pending), len(pending))

    if pending:
        detector = FaceDetector(resolve_device(cfg.device), cfg.detect_max_side)
        new_file = not journal_path.exists()
        with journal_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            for n, item in enumerate(pending, 1):
                writer.writerows(process_item(item, detector, cfg, out_dir))
                fh.flush()
                if n % 100 == 0 or n == len(pending):
                    LOG.info("processed %d/%d", n, len(pending))

    journal = [r for r in _read_journal(journal_path) if r["source_key"] in by_key]
    ok_rows = [r for r in journal if r["status"] == "ok"]
    skip_reasons = Counter(r["reason"] for r in journal if r["status"] == "skipped")

    rows, dropped = assign_splits(ok_rows, by_key, cfg)
    rows, removed = remove_cross_split_near_duplicates(rows, cfg.near_dup_distance)
    if removed:
        LOG.warning("removed %d cross-split near-duplicate crops", len(removed))
        dropped["cross_split_near_duplicate"] += len(removed)
    leaks = find_leaks(rows, cfg.near_dup_distance)
    if any(leaks.values()):
        raise RuntimeError(f"split leakage detected: {leaks}")

    rows.sort(key=lambda r: r["path"])
    write_csv(out_dir / "manifest.csv", rows, MANIFEST_FIELDS)

    warnings, stats = [], {}
    for name in sorted({r["source_dataset"] for r in rows}):
        stats[name] = class_stats([r for r in rows if r["source_dataset"] == name])
        warnings.extend(shortcut_warnings(stats[name], name))
    for w in warnings:
        LOG.warning("shortcut risk: %s", w)

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["split"]][f"{r['source_dataset']}/{r['label']}"] += 1
    report = {
        "source_files": len(items),
        "crops": len(rows),
        "per_split": {s: dict(c) for s, c in sorted(counts.items())},
        "skipped": dict(skip_reasons),
        "dropped": dict(dropped),
        "class_stats": stats,
        "shortcut_warnings": warnings,
        "pending_after_run": len(by_key.keys() - {r["source_key"] for r in journal}),
    }
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("manifest: %d crops -> %s", len(rows), out_dir / "manifest.csv")
    return report
