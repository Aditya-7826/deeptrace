"""Write a small synthetic crop set + manifest so the training loop can be proven on CPU
without downloading any dataset.

    uv run python scripts/make_debug_data.py --out data/crops/debug --n 200

"real" crops are smooth colour fields with noise; "fake" crops add a faint periodic
grid (a stand-in for generator upsampling artifacts). The class ratio is 40/60 so
the balanced sampler is exercised. This data proves the loop runs and learns.
It says nothing about deepfake detection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from deeptrace.data.phash import phash, to_hex
from deeptrace.data.preprocess import decode_jpeg, encode_jpeg

FIELDS = ["path", "label", "source_dataset", "manipulation_method", "video_id",
          "identity_id", "split", "phash"]  # fmt: skip


def make_image(rng: np.random.Generator, fake: bool, size: int = 224) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([rng.uniform(40, 200) + rng.uniform(-60, 60) * (xx * c1 + yy * c2)
                     for c1, c2 in rng.uniform(-1, 1, (3, 2))], axis=-1)  # fmt: skip
    img = base + rng.normal(0, 8, base.shape)
    if fake:
        period = rng.integers(4, 9)
        grid = ((np.arange(size) % period) == 0).astype(np.float32)
        img += 14 * (grid[None, :, None] + grid[:, None, None])
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/crops/debug"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        fake = i % 5 >= 2  # 40% real, 60% fake
        label = "fake" if fake else "real"
        rel = f"crops/debug/{label}/img_{i:05d}.jpg"
        data = encode_jpeg(make_image(rng, fake))
        (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
        (args.out / rel).write_bytes(data)
        rows.append({
            "path": rel, "label": label, "source_dataset": "debug",
            "manipulation_method": "grid" if fake else "real", "video_id": "",
            "identity_id": f"debug:{i}", "split": "val" if i % 5 == 4 or i % 10 == 1 else "train",
            "phash": to_hex(phash(decode_jpeg(data))),
        })  # fmt: skip
    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_val = sum(r["split"] == "val" for r in rows)
    print(f"wrote {len(rows)} crops ({len(rows) - n_val} train / {n_val} val) to {args.out}")


if __name__ == "__main__":
    main()
