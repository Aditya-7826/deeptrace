"""FastAPI service tests (TestClient) on the tiny exported model."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from deeptrace.cli import main as cli_main
from deeptrace.data.datasets import CropDataset
from deeptrace.data.faces import FaceDetector
from deeptrace.data.io import load_image
from deeptrace.data.preprocess import extract_crops
from deeptrace.serve.app import create_app
from deeptrace.serve.engine import Engine, MissingWeightsError
from deeptrace.serve.limits import LRUCache, RateLimiter
from deeptrace.serve.settings import DISCLAIMER, Settings
from deeptrace.serve.validation import sniff_type
from tests.conftest import IMAGES

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine(exported):
    return Engine(exported[0], threads=2)


def make_client(engine, **overrides) -> TestClient:
    s = Settings()
    s.weights_dir = engine.weights_dir
    for k, v in overrides.items():
        setattr(s, k, v)
    return TestClient(create_app(s, engine=engine))


@pytest.fixture
def client(engine):
    return make_client(engine)


def upload(client, data: bytes, name="photo.jpg", ctype="image/jpeg", path="/predict"):
    return client.post(path, files={"file": (name, data, ctype)})


def img_bytes(rel: str) -> bytes:
    return (IMAGES / rel).read_bytes()


def encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


# ------------------------------------------------------------------ basics


def test_health_version_and_disclaimer(client):
    assert client.get("/health").json() == {"status": "ok", "disclaimer": DISCLAIMER}
    v = client.get("/version").json()
    assert v["threshold"] == 0.42 and v["calibration_temperature"] == 1.7
    assert v["model_version"].startswith("tiny-") and v["disclaimer"] == DISCLAIMER


def test_happy_path_multipart_and_raw(client):
    r = upload(client, img_bytes("real/aldrin.jpg"))
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and len(body["faces"]) == 1
    face = body["faces"][0]
    assert 0.0 <= face["probability_fake"] <= 1.0
    assert face["label"] in ("real", "fake", "uncertain") and len(face["box"]) == 4
    assert body["disclaimer"] == DISCLAIMER
    raw = client.post("/predict", content=img_bytes("real/aldrin.jpg"),
                      headers={"Content-Type": "application/octet-stream"})  # fmt: skip
    assert raw.status_code == 200 and raw.json()["faces"] == body["faces"]


def test_no_face_and_tiny_face(client):
    assert upload(client, img_bytes("real/noface.jpg")).json()["status"] == "no_face"
    assert upload(client, img_bytes("fake/tinyface.jpg")).json()["status"] == "face_too_small"


def test_multiple_faces(client):
    a = Image.open(IMAGES / "real/aldrin.jpg").convert("RGB")
    b = Image.open(IMAGES / "fake/whitson.jpg").convert("RGB")
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (128, 128, 128))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    body = upload(client, encode(canvas, "JPEG", quality=92)).json()
    assert body["status"] == "ok" and len(body["faces"]) == 2
    assert [f["index"] for f in body["faces"]] == [0, 1]
    xs = sorted(f["box"][0] for f in body["faces"])
    assert xs[0] < a.width <= xs[1]  # one face in each half


def test_rgba_gif_webp_and_rotated_exif(client):
    assert upload(client, img_bytes("real/jemison_rgba.png"), "x.png", "image/png").json()[
        "status"] == "ok"  # fmt: skip
    portrait = Image.open(IMAGES / "real/aldrin.jpg").convert("RGB")
    assert upload(client, encode(portrait, "GIF"), "x.gif", "image/gif").json()["status"] == "ok"
    assert upload(client, encode(portrait, "WEBP"), "x.webp", "image/webp").json()[
        "status"] == "ok"  # fmt: skip
    rotated = upload(client, img_bytes("fake/armstrong_exif_rotated.jpg")).json()
    upright = upload(client, img_bytes("real/armstrong.jpg")).json()
    assert rotated["status"] == "ok"
    # boxes are in upright (EXIF-applied) coordinates: same place as the upright original
    assert np.allclose(rotated["faces"][0]["box"], upright["faces"][0]["box"], atol=15)


# ------------------------------------------------------------------ rejections


def test_corrupt_and_empty(client):
    r = upload(client, img_bytes("real/corrupt.jpg"))
    assert r.status_code == 400 and r.json()["error"] == "corrupt_image"
    assert "could not be decoded" in r.json()["message"] and r.json()["disclaimer"] == DISCLAIMER
    assert upload(client, b"").status_code == 400


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.7\n1 0 obj<<>>endobj\n", b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
     b"MZ\x90\x00 not an image", b"BM" + b"\x00" * 60],
)  # fmt: skip
def test_wrong_type_disguised_as_jpeg(client, data):
    r = upload(client, data, name="holiday.jpg", ctype="image/jpeg")
    assert r.status_code == 415 and r.json()["error"] == "unsupported_type"


def test_png_with_jpg_name_is_judged_by_content(client):
    png = encode(Image.open(IMAGES / "real/aldrin.jpg").convert("RGB"), "PNG")
    assert sniff_type(png) == "png"
    assert upload(client, png, name="really.jpg", ctype="image/jpeg").status_code == 200


def test_polyglot_signature_mismatch(client):
    fake = b"\xff\xd8\xff" + img_bytes("real/jemison_rgba.png")  # JPEG magic, PNG body
    assert upload(client, fake).status_code in (400, 415)


def test_oversized_upload(client):
    big = b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024)
    assert upload(client, big).status_code == 413
    r = client.post("/predict", content=big, headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 413 and r.json()["error"] == "file_too_large"


def test_pixel_cap(engine):
    c = make_client(engine, max_pixels=100_000)  # aldrin.jpg is 520x650 = 338k pixels
    r = upload(c, img_bytes("real/aldrin.jpg"))
    assert r.status_code == 413 and r.json()["error"] == "too_many_pixels"


# ------------------------------------------------------------------ explain


def test_explain_lazy_fp32_and_overlays(engine):
    c = make_client(engine)
    engine.gradcam = None
    assert upload(c, img_bytes("real/aldrin.jpg")).status_code == 200
    assert engine.gradcam is None  # /predict never loads the FP32 model
    r = upload(c, img_bytes("real/aldrin.jpg"), path="/explain?face_index=0")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and engine.gradcam is not None
    for key in ("crop_overlay_png", "image_overlay_png"):
        png = base64.b64decode(body[key])
        assert png.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(base64.b64decode(body["crop_overlay_png"]))) as im:
        assert im.size == (224, 224)
    bad = upload(c, img_bytes("real/aldrin.jpg"), path="/explain?face_index=3")
    assert bad.status_code == 400 and bad.json()["error"] == "bad_face_index"


# ------------------------------------------------------------------ runtime behaviour


def test_cache_hit(engine):
    c = make_client(engine)
    calls = []
    original = engine.predict
    engine.predict = lambda rgb: calls.append(1) or original(rgb)
    try:
        first = upload(c, img_bytes("fake/ochoa.jpg")).json()
        second = upload(c, img_bytes("fake/ochoa.jpg")).json()
    finally:
        engine.predict = original
    assert first["cached"] is False and second["cached"] is True and len(calls) == 1
    assert first["faces"] == second["faces"] and first["image_sha256"] == second["image_sha256"]


def test_concurrent_requests_are_capped_at_two(engine):
    c = make_client(engine)
    original = engine.predict

    def slow(rgb):
        time.sleep(0.2)
        return original(rgb)

    engine.predict = slow
    portrait = Image.open(IMAGES / "real/aldrin.jpg").convert("RGB")
    payloads = [encode(portrait, "JPEG", quality=70 + k) for k in range(6)]  # no cache hits
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            codes = list(pool.map(lambda d: upload(c, d).status_code, payloads))
    finally:
        engine.predict = original
    assert codes == [200] * 6
    gate = c.app.state.gate
    assert gate.max_active == 2 and gate.active == 0


def test_timeout_returns_504(engine):
    c = make_client(engine, request_timeout_s=0.2)
    original = engine.predict
    engine.predict = lambda rgb: (time.sleep(1.0), original(rgb))[1]
    try:
        r = upload(c, img_bytes("fake/whitson.jpg"))
    finally:
        engine.predict = original
    assert r.status_code == 504 and r.json()["error"] == "timeout"


def test_rate_limit(engine):
    c = make_client(engine, rate_limit=3)
    codes = [upload(c, img_bytes("real/aldrin.jpg")).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = upload(c, img_bytes("real/aldrin.jpg"))
    assert int(r.headers["Retry-After"]) >= 1 and r.json()["error"] == "rate_limited"
    assert c.get("/health").status_code == 200  # health is not rate limited


def test_rate_limiter_window_unit():
    now = [0.0]
    rl = RateLimiter(2, 10.0, clock=lambda: now[0])
    assert rl.check("a") is None and rl.check("a") is None and rl.check("a") == 10.0
    assert rl.check("b") is None  # per client
    now[0] = 10.5
    assert rl.check("a") is None


def test_lru_cache_unit():
    cache = LRUCache(2)
    cache.put("a", 1), cache.put("b", 2), cache.get("a"), cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and len(cache) == 2


def test_cors(client):
    ok = client.options("/predict", headers={"Origin": "chrome-extension://abcdefghijklmnop",
                                             "Access-Control-Request-Method": "POST"})  # fmt: skip
    assert ok.headers.get("access-control-allow-origin") == "chrome-extension://abcdefghijklmnop"
    local = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


# ------------------------------------------------------------------ weights + parity


def test_missing_weights(tmp_path, capsys):
    with pytest.raises(MissingWeightsError, match="deeptrace download-weights"):
        Engine(tmp_path)
    assert cli_main(["serve", "--weights-dir", str(tmp_path)]) == 1
    assert "deeptrace download-weights" in capsys.readouterr().err


def test_serving_preprocessing_matches_training(engine, tmp_path):
    """Same input -> identical tensor through the training dataset and the API engine."""
    rgb, _ = load_image(IMAGES / "real/aldrin.jpg")
    crops, _ = extract_crops(rgb, FaceDetector("cpu"), largest_only=True)  # what prepare writes
    (tmp_path / "c.jpg").write_bytes(crops[0].jpeg)
    row = {"path": "c.jpg", "label": "real"}
    x_train, _ = CropDataset([row], tmp_path, None, engine.mean, engine.std)[0]
    served, _ = engine.face_crops(load_image(img_bytes("real/aldrin.jpg"))[0])
    x_serve = torch.from_numpy(engine.face_tensors(served)[0])
    assert torch.equal(x_train, x_serve)


def test_memory_script_under_budget(exported, tmp_path):
    out = tmp_path / "api_memory.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "measure_api_memory.py"), "--weights-dir",
         str(exported[0]), "--requests", "6", "--explains", "2", "--out", str(out)],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    report = json.loads(out.read_text(encoding="utf-8"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert report["all_requests_ok"] and report["peak_rss_mb"] < 1500


def test_threading_gate_skips_cancelled():
    from deeptrace.serve.app import Gate

    gate, cancelled = Gate(1), threading.Event()
    cancelled.set()
    assert gate.run(lambda: "ran", cancelled) is None
    assert gate.run(lambda: "ran", threading.Event()) == "ran"
