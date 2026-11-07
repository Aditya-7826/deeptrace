"""Upload validation: real type from magic bytes (never the extension or Content-Type)."""

from __future__ import annotations

ALLOWED = {"jpeg", "png", "webp", "gif"}
_PIL_FORMAT = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "gif": "GIF"}


def sniff_type(data: bytes) -> str | None:
    """Return jpeg/png/webp/gif from the file signature, or None for anything else."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def decoded_matches(kind: str, pil_format: str | None) -> bool:
    """Guard against polyglots: the decoder must agree with the signature."""
    return _PIL_FORMAT.get(kind) == pil_format
