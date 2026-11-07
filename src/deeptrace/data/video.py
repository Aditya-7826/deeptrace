"""Evenly spaced frame sampling, one frame in memory at a time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReadError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def frame_indices(total: int, n: int) -> list[int]:
    if total <= 0 or n <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, min(n, total))})


def sample_frames(path: str | Path, n: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, RGB frame) for up to n evenly spaced frames."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise VideoReadError("empty_file", str(path))
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("unreadable_video", str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = frame_indices(total, n)
        if not indices:
            raise VideoReadError("unreadable_video", f"{path} reports {total} frames")
        produced = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            produced += 1
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if produced == 0:
            raise VideoReadError("unreadable_video", f"no decodable frames in {path}")
    finally:
        cap.release()
