"""Leakage-proof split assignment and verification.

- Video datasets: all videos that share any identity/source video are joined
  (union-find) and the whole component goes to one split.
- Image datasets: near-duplicate images (pHash distance <= 4) are joined into
  one group, so near-duplicates never straddle splits.
- A final pass removes any remaining cross-split near-duplicates (keeping the
  test copy over val over train), and `find_leaks` verifies the result.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

from deeptrace.data.phash import from_hex, near_duplicate_pairs

SPLITS = ("train", "val", "test")
_PRIORITY = {"test": 2, "val": 1, "train": 0}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def find(self, x: Hashable) -> Hashable:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=str)] = min(ra, rb, key=str)


def assign_groups(
    group_sizes: dict[str, int], fractions: Sequence[float], seed: int
) -> dict[str, str]:
    """Shuffle groups deterministically and fill train, then val, then test by item count."""
    keys = sorted(group_sizes)
    random.Random(seed).shuffle(keys)
    total = sum(group_sizes.values()) or 1
    bounds = (fractions[0], fractions[0] + fractions[1])
    out, running = {}, 0
    for key in keys:
        position = running / total
        out[key] = "train" if position < bounds[0] else "val" if position < bounds[1] else "test"
        running += group_sizes[key]
    return out


def identity_groups(identity_ids: Sequence[str]) -> list[str]:
    """Group label per row: rows sharing any identity (split on '|') share a group."""
    uf = UnionFind()
    for ident in identity_ids:
        parts = ident.split("|")
        for p in parts[1:]:
            uf.union(parts[0], p)
    return [str(uf.find(ident.split("|")[0])) for ident in identity_ids]


def near_duplicate_groups(hashes: Sequence[str], max_distance: int = 4) -> list[str]:
    ints = [from_hex(h) for h in hashes]
    uf = UnionFind()
    for i in range(len(ints)):
        uf.find(i)
    for i, j in near_duplicate_pairs(ints, max_distance):
        uf.union(i, j)
    return [f"dup{uf.find(i)}" for i in range(len(ints))]


def remove_cross_split_near_duplicates(
    rows: list[dict], max_distance: int = 4
) -> tuple[list[dict], list[dict]]:
    """Drop the lower-priority member of any near-duplicate pair in different splits."""
    hashes = [from_hex(r["phash"]) for r in rows]
    dropped: set[int] = set()
    for i, j in near_duplicate_pairs(hashes, max_distance):
        si, sj = rows[i]["split"], rows[j]["split"]
        if si == sj or i in dropped or j in dropped:
            continue
        dropped.add(i if _PRIORITY[si] < _PRIORITY[sj] else j)
    kept = [r for k, r in enumerate(rows) if k not in dropped]
    removed = [r for k, r in enumerate(rows) if k in dropped]
    return kept, removed


def _multi_split(values: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for value, split in values:
        if value:
            seen[value].add(split)
    return {v: s for v, s in seen.items() if len(s) > 1}


def find_leaks(rows: list[dict], max_distance: int = 4) -> dict[str, list]:
    """Return every video_id, identity and near-duplicate pair seen in more than one split."""
    videos = _multi_split((r["video_id"], r["split"]) for r in rows)
    identities = _multi_split(
        (ident, r["split"]) for r in rows for ident in r["identity_id"].split("|")
    )
    hashes = [from_hex(r["phash"]) for r in rows]
    dup_pairs = [
        (rows[i]["path"], rows[j]["path"])
        for i, j in near_duplicate_pairs(hashes, max_distance)
        if rows[i]["split"] != rows[j]["split"]
    ]
    return {
        "video_id": sorted(videos),
        "identity_id": sorted(identities),
        "near_duplicates": dup_pairs,
    }
