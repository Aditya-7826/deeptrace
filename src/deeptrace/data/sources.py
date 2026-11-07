"""Dataset adapters: turn a raw dataset folder into a list of SourceItems.

Supported layouts
- image_folder: any tree where a path component is named real/ or fake/
  (e.g. Kaggle "140k Real and Fake Faces"). Given train/valid/test folders are
  ignored; we re-split with perceptual-hash deduplication.
- ffpp: FaceForensics++ (original_sequences/youtube/<c>/videos/000.mp4,
  manipulated_sequences/<Method>/<c>/videos/000_003.mp4).
- celebdf: Celeb-DF v2 (Celeb-real/, Celeb-synthesis/, YouTube-real/). Test only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


@dataclass(frozen=True)
class SourceItem:
    path: Path
    rel: str  # posix path relative to the dataset root
    kind: str  # "image" or "video"
    label: str  # "real" or "fake"
    dataset: str
    method: str
    video_id: str  # "" for still images
    identities: tuple[str, ...]
    fixed_split: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.rel}"

    @property
    def identity_id(self) -> str:
        return "|".join(self.identities)


def _files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def image_folder(root: Path, name: str, fake_method: str = "generated") -> list[SourceItem]:
    items = []
    for path in _files(root, IMAGE_EXTS):
        rel = path.relative_to(root).as_posix()
        parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
        label = next((p for p in reversed(parts) if p in ("real", "fake")), None)
        if label is None:
            continue
        stem = rel.rsplit(".", 1)[0]
        items.append(
            SourceItem(
                path=path,
                rel=rel,
                kind="image",
                label=label,
                dataset=name,
                method="real" if label == "real" else fake_method,
                video_id="",
                identities=(f"{name}:{stem}",),  # identities unknown: one per image
            )
        )
    return items


def _ffpp_official_splits(splits_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split, fname in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
        for pair in json.loads((splits_dir / fname).read_text(encoding="utf-8")):
            for vid in pair:
                mapping[str(vid)] = split
    return mapping


def ffpp(
    root: Path,
    name: str = "ffpp",
    compression: str = "c23",
    methods: tuple[str, ...] | list[str] = FFPP_METHODS,
    splits_dir: Path | None = None,
) -> list[SourceItem]:
    official = _ffpp_official_splits(splits_dir) if splits_dir else None

    def fixed(ids: list[str]) -> str | None:
        if official is None:
            return None
        splits = {official.get(i) for i in ids}
        return splits.pop() if len(splits) == 1 and None not in splits else "unassigned"

    items = []
    orig_dir = root / "original_sequences" / "youtube" / compression / "videos"
    for path in _files(orig_dir, VIDEO_EXTS) if orig_dir.exists() else []:
        vid = path.stem
        items.append(
            SourceItem(path, path.relative_to(root).as_posix(), "video", "real", name, "original",
                       f"{name}:original/{vid}", (f"{name}:{vid}",), fixed([vid]))
        )
    for method in methods:
        mdir = root / "manipulated_sequences" / method / compression / "videos"
        for path in _files(mdir, VIDEO_EXTS) if mdir.exists() else []:
            ids = path.stem.split("_")
            items.append(
                SourceItem(path, path.relative_to(root).as_posix(), "video", "fake", name, method,
                           f"{name}:{method}/{path.stem}",
                           tuple(f"{name}:{i}" for i in ids), fixed(ids))
            )
    return items


_CELEB_REAL = re.compile(r"^(id\d+)_\d+$")
_CELEB_FAKE = re.compile(r"^(id\d+)_(id\d+)_\d+$")


def celebdf(root: Path, name: str = "celebdf", test_list: Path | None = None) -> list[SourceItem]:
    allowed = None
    if test_list:
        allowed = {
            line.split()[-1].strip()
            for line in test_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    items = []
    for folder, label in (("Celeb-real", "real"), ("YouTube-real", "real"), ("Celeb-synthesis", "fake")):
        fdir = root / folder
        for path in _files(fdir, VIDEO_EXTS) if fdir.exists() else []:
            rel = path.relative_to(root).as_posix()
            if allowed is not None and rel not in allowed:
                continue
            stem = path.stem
            if label == "fake" and (m := _CELEB_FAKE.match(stem)):
                ids = (f"{name}:{m.group(1)}", f"{name}:{m.group(2)}")
            elif m := _CELEB_REAL.match(stem):
                ids = (f"{name}:{m.group(1)}",)
            else:
                ids = (f"{name}:yt{stem}",)
            method = "celebdf_synthesis" if label == "fake" else "real"
            items.append(
                SourceItem(path, rel, "video", label, name, method, f"{name}:{folder}/{stem}", ids,
                           "test")
            )
    return items


def enumerate_dataset(kind: str, root: Path, name: str, **options) -> list[SourceItem]:
    if kind == "image_folder":
        return image_folder(root, name, **options)
    if kind == "ffpp":
        if options.get("splits_dir"):
            options["splits_dir"] = Path(options["splits_dir"])
        return ffpp(root, name, **options)
    if kind == "celebdf":
        if options.get("test_list"):
            options["test_list"] = Path(options["test_list"])
        return celebdf(root, name, **options)
    raise ValueError(f"unknown dataset type: {kind!r}")
