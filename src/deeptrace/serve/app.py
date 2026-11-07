"""FastAPI service. Run with `deeptrace serve` (one uvicorn worker).

The API never fetches URLs: it only scores bytes the client uploads (no SSRF).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deeptrace import __version__
from deeptrace.data.io import ImageLoadError, load_image
from deeptrace.serve.engine import Engine
from deeptrace.serve.limits import LRUCache, RateLimiter
from deeptrace.serve.settings import CORS_ORIGIN_REGEX, DISCLAIMER, Settings
from deeptrace.serve.validation import ALLOWED, decoded_matches, sniff_type

LOG = logging.getLogger("deeptrace.serve")


class APIError(Exception):
    def __init__(self, status: int, error: str, message: str, headers: dict | None = None):
        self.status, self.error, self.message, self.headers = status, error, message, headers


def respond(body: dict[str, Any], status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({**body, "disclaimer": DISCLAIMER}, status_code=status, headers=headers)


class Gate:
    """At most `n` inferences at once. The slot is taken inside the worker thread, and
    a request that timed out while queued is skipped instead of running late."""

    def __init__(self, n: int) -> None:
        self.sem = threading.BoundedSemaphore(n)
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def run(self, fn, cancelled: threading.Event):
        with self.sem:
            if cancelled.is_set():
                return None
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                return fn()
            finally:
                with self._lock:
                    self.active -= 1


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()
    # Raises MissingWeightsError (with a download hint) before the app exists.
    engine = engine or Engine(settings.weights_dir, settings.threads, settings.verify_checksums)
    cache = LRUCache(settings.cache_size)
    limiter = RateLimiter(settings.rate_limit, settings.rate_window_s)
    gate = Gate(settings.max_concurrent)

    app = FastAPI(title="DeepTrace API", version=__version__)
    app.state.engine, app.state.cache, app.state.gate = engine, cache, gate
    app.state.settings, app.state.limiter = settings, limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        return respond({"error": exc.error, "message": exc.message}, exc.status, exc.headers)

    async def read_upload(request: Request) -> bytes:
        limit = settings.max_upload_bytes
        too_big = APIError(413, "file_too_large", f"maximum upload size is {limit // 2**20} MB")
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > limit + 64 * 1024:  # multipart overhead
            raise too_big
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form(max_files=1, max_fields=5, max_part_size=limit + 1)
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise APIError(400, "missing_file", "send the image as multipart field 'file'")
            data = await upload.read(limit + 1)
            await upload.close()
        else:
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise too_big
                chunks.append(chunk)
            data = b"".join(chunks)
        if len(data) > limit:
            raise too_big
        if not data:
            raise APIError(400, "empty_file", "the upload is empty")
        return data

    def decode(data: bytes):
        kind = sniff_type(data)
        if kind not in ALLOWED:
            msg = "only JPEG, PNG, WebP and GIF (first frame) images are accepted"
            raise APIError(415, "unsupported_type", msg)
        try:
            rgb, meta = load_image(data, settings.max_pixels)
        except ImageLoadError as exc:
            if exc.reason == "too_many_pixels":
                msg = f"image exceeds {settings.max_pixels:,} pixels"
                raise APIError(413, "too_many_pixels", msg) from exc
            raise APIError(400, "corrupt_image",
                           f"the image could not be decoded ({exc.reason})") from exc  # fmt: skip
        if not decoded_matches(kind, meta.format):
            raise APIError(415, "unsupported_type", "file signature and content disagree")
        return rgb

    def rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait is not None:
            raise APIError(429, "rate_limited", "too many requests; slow down",
                           {"Retry-After": str(max(1, round(wait)))})  # fmt: skip

    async def run_inference(fn):
        cancelled = threading.Event()
        try:
            with anyio.fail_after(settings.request_timeout_s):
                return await anyio.to_thread.run_sync(lambda: gate.run(fn, cancelled),
                                                      abandon_on_cancel=True)  # fmt: skip
        except TimeoutError as exc:
            cancelled.set()
            raise APIError(
                504, "timeout", f"inference took longer than {settings.request_timeout_s:g} s"
            ) from exc

    def versioned(body: dict[str, Any], digest: str, cached: bool) -> dict[str, Any]:
        return {**body, "image_sha256": digest, "cached": cached,
                "model_version": engine.meta["version"]}  # fmt: skip

    @app.get("/health")
    async def health():
        return respond({"status": "ok"})

    @app.get("/version")
    async def version():
        m = engine.meta
        return respond({
            "api_version": __version__, "model_version": m["version"],
            "model_name": m["model_name"], "serving_model": m["serving_model"],
            "threshold": m["threshold"], "calibration_temperature": m["temperature"],
            "uncertain_band": m["uncertain_band"], "calibrated": m["calibrated"],
        })  # fmt: skip

    @app.post("/predict")
    async def predict(request: Request):
        rate_limit(request)
        data = await read_upload(request)
        digest = hashlib.sha256(data).hexdigest()
        key = f"predict:{digest}"
        if (hit := cache.get(key)) is not None:
            return respond(versioned(hit, digest, True))
        rgb = decode(data)
        result = await run_inference(lambda: engine.predict(rgb))
        cache.put(key, result)
        return respond(versioned(result, digest, False))

    @app.post("/explain")
    async def explain(request: Request, face_index: int = 0):
        rate_limit(request)
        data = await read_upload(request)
        digest = hashlib.sha256(data).hexdigest()
        key = f"explain:{face_index}:{digest}"
        if (hit := cache.get(key)) is not None:
            return respond(versioned(hit, digest, True))
        rgb = decode(data)
        result = await run_inference(lambda: engine.explain(rgb, face_index))
        if result.get("status") == "bad_face_index":
            raise APIError(400, "bad_face_index",
                           f"face_index must be 0..{result['faces_found'] - 1}")  # fmt: skip
        cache.put(key, result)
        return respond(versioned(result, digest, False))

    return app
