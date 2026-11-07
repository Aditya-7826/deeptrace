"""Seeding and run provenance (config, git commit, package versions)."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGES = ("torch", "torchvision", "timm", "albumentations", "numpy", "scikit-learn",
            "opencv-python-headless", "deeptrace")  # fmt: skip


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    env = os.environ.get("DEEPTRACE_GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run_info(config: dict[str, Any], device: str) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    return {
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": package_versions(),
        "platform": platform.platform(),
        "device": device,
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count() if gpu else 0,
        "config": config,
    }
