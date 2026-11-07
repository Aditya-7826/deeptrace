"""Manifest-backed dataset that streams crops from disk (never loads a whole dataset)."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from deeptrace.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, decode_jpeg, to_model_input

LABELS = {"real": 0, "fake": 1}


def read_manifest(root: str | Path) -> list[dict[str, str]]:
    path = Path(root) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `deeptrace prepare` or check data.root")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    datasets: list[str] | None = None,
    max_items: int | None = None,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Rows of one split (optionally some datasets), stratified-subsampled to max_items."""
    out = [
        r for r in rows if r["split"] == split and (not datasets or r["source_dataset"] in datasets)
    ]
    if max_items and len(out) > max_items:
        rng = random.Random(seed)
        by_label: dict[str, list[dict[str, str]]] = {}
        for r in out:
            by_label.setdefault(r["label"], []).append(r)
        picked = []
        for label in sorted(by_label):
            share = round(max_items * len(by_label[label]) / len(out))
            picked.extend(rng.sample(by_label[label], min(share, len(by_label[label]))))
        out = sorted(picked, key=lambda r: r["path"])
    return out


class CropDataset(Dataset):
    """Reads one JPEG crop per item. Augmentation is reseeded per (epoch, index),
    so a resumed run sees exactly the same augmented images."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        root: str | Path,
        transform=None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ) -> None:
        self.rows = rows
        self.root = Path(root)
        self.transform = transform
        self.mean, self.std = tuple(mean), tuple(std)
        self.epoch_seed = 0
        self.labels = [LABELS[r["label"]] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rgb = decode_jpeg((self.root / self.rows[idx]["path"]).read_bytes())
        if self.transform is not None:
            self.transform.set_random_seed((self.epoch_seed * 1_000_003 + idx) % 2**32)
            rgb = self.transform(image=rgb)["image"]
        x = torch.from_numpy(to_model_input(rgb, self.mean, self.std))
        return x, torch.tensor(float(self.labels[idx]))


def class_counts(labels: list[int]) -> dict[int, int]:
    counts = Counter(labels)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def balanced_sampler(labels: list[int], generator: torch.Generator) -> WeightedRandomSampler:
    counts = class_counts(labels)
    weights = np.array([1.0 / max(counts[y], 1) for y in labels], dtype=np.float64)
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(labels), replacement=True, generator=generator
    )


def pos_weight(labels: list[int]) -> float:
    """BCE pos_weight = n_real / n_fake (class-weighted loss alternative to the sampler)."""
    counts = class_counts(labels)
    return counts[0] / max(counts[1], 1)
