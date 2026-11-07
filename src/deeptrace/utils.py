"""Small shared helpers: logging, CPU thread limits, YAML loading."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("deeptrace")


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def set_cpu_threads(n: int = 2) -> None:
    """Cap CPU threads for PyTorch and OpenCV (8 GB RAM rule)."""
    import cv2
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        pass  # can only be set once per process
    cv2.setNumThreads(n)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
