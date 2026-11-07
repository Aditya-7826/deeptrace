"""API settings (environment variables override the defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DISCLAIMER = (
    "probabilistic estimate; may be wrong on unseen generators, heavy compression or screenshots"
)
# chrome-extension://<id>, http://localhost:<port>, http://127.0.0.1:<port>
CORS_ORIGIN_REGEX = (
    r"^(chrome-extension://[a-z]+|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$"
)


def _env(name: str, default, cast=str):
    value = os.environ.get(name)
    return default if value in (None, "") else cast(value)


@dataclass
class Settings:
    weights_dir: Path = field(
        default_factory=lambda: Path(_env("DEEPTRACE_WEIGHTS_DIR", "weights"))
    )
    threads: int = field(default_factory=lambda: _env("DEEPTRACE_THREADS", 2, int))
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_concurrent: int = 2
    request_timeout_s: float = field(
        default_factory=lambda: _env("DEEPTRACE_TIMEOUT_S", 30.0, float)
    )
    rate_limit: int = field(default_factory=lambda: _env("DEEPTRACE_RATE_LIMIT", 60, int))
    rate_window_s: float = 60.0
    cache_size: int = 128
    verify_checksums: bool = True
