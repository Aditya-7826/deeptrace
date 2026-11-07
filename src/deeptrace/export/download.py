"""`deeptrace download-weights`: fetch release assets and verify SHA-256 checksums.

The expected checksums live in deeptrace/weights.json, which ships with the code.
So a file is only accepted if it matches what this code version was released with.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any

USER_AGENT = "deeptrace-download/0.1"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


class ChecksumMismatch(DownloadError):
    pass


def default_dest() -> Path:
    return Path(os.environ.get("DEEPTRACE_WEIGHTS_DIR", "weights"))


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    text = resources.files("deeptrace").joinpath("weights.json").read_text(encoding="utf-8")
    return json.loads(text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def release_base_url(manifest: dict[str, Any]) -> str:
    return f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}/"


def missing_weights(dest: Path, manifest: dict[str, Any] | None = None) -> list[str]:
    """Asset names that are absent or fail their checksum in `dest`."""
    manifest = manifest or load_manifest()
    bad = []
    for asset in manifest.get("assets", []):
        p = Path(dest) / asset["name"]
        if not p.exists() or p.stat().st_size != asset["size"] or sha256_file(p) != asset["sha256"]:
            bad.append(asset["name"])
    return bad


def _fetch(url: str, dest: Path, asset: dict[str, Any], retries: int, timeout: float,
           backoff: float) -> None:  # fmt: skip
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:
                while chunk := resp.read(CHUNK):
                    h.update(chunk)
                    fh.write(chunk)
            digest = h.hexdigest()
            if digest != asset["sha256"]:
                part.unlink(missing_ok=True)
                raise ChecksumMismatch(
                    f"checksum mismatch for {asset['name']}: expected {asset['sha256']}, "
                    f"got {digest}. The file was corrupted in transit or does not match this "
                    f"release. Retry with `deeptrace download-weights --force`; if it keeps "
                    f"failing, the release asset differs from deeptrace/weights.json."
                )
            part.replace(dest)
            return
        except ChecksumMismatch:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise DownloadError(
        f"could not download {asset['name']} from {url} after {retries} attempts ({last}). "
        f"Check your internet connection (and that the release exists), then retry: "
        f"`deeptrace download-weights`"
    )


def download_weights(dest: Path | None = None, base_url: str | None = None,
                     manifest: dict[str, Any] | None = None, force: bool = False,
                     retries: int = 3, timeout: float = 60.0,
                     backoff: float = 2.0) -> list[dict[str, Any]]:  # fmt: skip
    manifest = manifest or load_manifest()
    if not manifest.get("tag") or not manifest.get("assets"):
        raise DownloadError("this code version has no released weights yet "
                            "(deeptrace/weights.json lists no assets)")  # fmt: skip
    dest = Path(dest or default_dest())
    dest.mkdir(parents=True, exist_ok=True)
    base = base_url or release_base_url(manifest)
    if not base.endswith("/"):
        base += "/"
    report = []
    for asset in manifest["assets"]:
        path = dest / asset["name"]
        if not force and path.exists() and sha256_file(path) == asset["sha256"]:
            report.append({"name": asset["name"], "status": "already present, checksum ok"})
            continue
        _fetch(base + asset["name"], path, asset, retries, timeout, backoff)
        report.append({"name": asset["name"], "status": "downloaded, checksum ok",
                       "bytes": asset["size"]})  # fmt: skip
    return report
