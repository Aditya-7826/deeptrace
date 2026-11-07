"""64-bit perceptual hash (DCT pHash) and fast near-duplicate search."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import cv2
import numpy as np

# 64 bits split into 5 chunks: two hashes within Hamming distance 4 must agree
# exactly on at least one chunk (pigeonhole), so bucketing by chunk finds all pairs.
_CHUNKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))


def phash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    median = np.median(low[1:])  # exclude the DC term
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def to_hex(value: int) -> str:
    return f"{value:016x}"


def from_hex(text: str) -> int:
    return int(text, 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate_pairs(hashes: Sequence[int], max_distance: int = 4) -> Iterator[tuple[int, int]]:
    """Yield index pairs (i < j) whose hashes are within `max_distance` bits."""
    if max_distance > len(_CHUNKS) - 1:
        raise ValueError("max_distance > 4 needs more chunks")
    seen: set[tuple[int, int]] = set()
    for lo, hi in _CHUNKS:
        mask = (1 << (hi - lo)) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, h in enumerate(hashes):
            buckets[(h >> lo) & mask].append(idx)
        for members in buckets.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pair = (members[a], members[b])
                    if pair not in seen and hamming(hashes[pair[0]], hashes[pair[1]]) <= max_distance:
                        seen.add(pair)
                        yield pair
