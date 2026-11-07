"""Export, INT8 gate, benchmark helpers, release staging and checksum-verified downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deeptrace.export.benchmark import _worker, cpu_name, peak_rss_mb, percentiles
from deeptrace.export.download import (
    ChecksumMismatch,
    DownloadError,
    download_weights,
    missing_weights,
)
from deeptrace.export.onnx_export import export_onnx, onnx_logits, ort_session, parity
from deeptrace.export.quantize import Attempt, choose, sensitive_nodes
from deeptrace.models.factory import ModelConfig, create_model
from tests.conftest import IMAGES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["test_efficientnet", "test_resnet", "test_vit"])
def test_onnx_parity_and_dynamic_batch(name, tmp_path):
    torch.manual_seed(0)
    model = create_model(ModelConfig(name=name, pretrained=False)).eval()
    path = export_onnx(model, tmp_path / "m.onnx")
    x = np.random.default_rng(0).normal(size=(5, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).reshape(-1).numpy()
    out = onnx_logits(ort_session(path, threads=2), iter([x[:3], x[3:]]))  # batch 3, then 2
    assert parity(ref, out)["passed"]


def test_parity_reports_failure():
    r = parity(np.array([1.0, 2.0]), np.array([1.0, 2.001]), fp64=np.array([1.0, 2.0005]))
    assert not r["passed"] and r["max_abs_diff"] == pytest.approx(1e-3)
    assert r["onnx_fp32_vs_fp64_max"] == pytest.approx(5e-4)


def test_rgb_fft_export_is_refused(tmp_path):
    model = create_model(ModelConfig(name="test_efficientnet", arch="rgb_fft", pretrained=False))
    with pytest.raises(NotImplementedError):
        export_onnx(model, tmp_path / "m.onnx")


def test_sensitive_nodes(tmp_path):
    model = create_model(ModelConfig(name="test_efficientnet", pretrained=False))
    path = export_onnx(model, tmp_path / "m.onnx")
    edges = sensitive_nodes(path, "edges")
    assert len(edges) == 3 and sensitive_nodes(path, "none") == []
    assert len(sensitive_nodes(path, "edges+depthwise")) > len(edges)


def test_int8_gate_picks_first_within_budget():
    a = [Attempt("a", False, "none", auc_drop=0.05), Attempt("b", True, "none", auc_drop=0.008),
         Attempt("c", True, "edges", auc_drop=0.001)]  # fmt: skip
    assert choose(a).name == "b" and a[1].accepted and not a[2].accepted
    assert choose([Attempt("x", False, "none", auc_drop=0.02)]) is None
    assert choose([Attempt("y", False, "none", error="boom")]) is None


def test_export_outputs_and_gate(exported):
    out, results, report = exported
    assert report["parity"]["passed"] and report["parity"]["n"] > 0
    meta = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert meta["temperature"] == 1.7 and meta["threshold"] == 0.42 and meta["calibrated"]
    assert (out / meta["serving_model"]).exists() and (out / "model_fp32.pt").exists()
    assert json.loads((results / "tiny" / "export.json").read_text(encoding="utf-8"))
    if meta["serving_model"] == "model_int8.onnx":
        accepted = [a for a in report["int8_attempts"] if a["accepted"]]
        assert len(accepted) == 1 and accepted[0]["auc_drop"] <= 0.01
    else:  # rejected: no INT8 file may be left behind, and the reason is recorded
        assert not (out / "model_int8.onnx").exists()
        assert "FP32" in report["decision"]
    assert not list(out.glob("model_int8_*.onnx"))  # candidate files cleaned up


def test_benchmark_worker_and_helpers(exported, tmp_path):
    out, _, _ = exported
    from deeptrace.data.faces import FaceDetector
    from deeptrace.data.io import load_image
    from deeptrace.data.preprocess import extract_crops

    image = IMAGES / "real" / "aldrin.jpg"
    crops, _ = extract_crops(load_image(image)[0], FaceDetector("cpu"))
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(crops[0].jpeg)
    kind = "onnx_int8" if (out / "model_int8.onnx").exists() else "onnx_fp32"
    for k in (kind, "pytorch_fp32"):
        row = _worker(k, "model_only", out, image, crop, runs=3, warmup=1, threads=2)
        assert row["p50_ms"] > 0 and row["p95_ms"] >= row["p50_ms"] and row["peak_rss_mb"] > 0
    p = percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert p["p50_ms"] == 3.0 and p["p95_ms"] > 4.0
    assert peak_rss_mb() > 10 and cpu_name()


# ------------------------------------------------------------ release + download


def _release(tmp_path):
    spec = importlib.util.spec_from_file_location("rw", ROOT / "scripts" / "release_weights.py")
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model_int8.onnx").write_bytes(b"onnx-bytes" * 100)
    (export / "model_fp32.pt").write_bytes(b"torch-bytes" * 100)
    (export / "model.json").write_text(
        json.dumps({"serving_model": "model_int8.onnx", "gradcam_model": "model_fp32.pt"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "weights.json"
    manifest_path.write_text(json.dumps({"repo": "x/y", "tag": None, "assets": []}),
                             encoding="utf-8")  # fmt: skip
    manifest = rw.stage(export, "v9.9.9", tmp_path / "release", manifest_path)
    return manifest, tmp_path / "release" / "v9.9.9"


def test_release_staging_pins_checksums(tmp_path):
    manifest, folder = _release(tmp_path)
    assert manifest["tag"] == "v9.9.9" and len(manifest["assets"]) == 3
    for a in manifest["assets"]:
        assert hashlib.sha256((folder / a["name"]).read_bytes()).hexdigest() == a["sha256"]
    assert (folder / "SHA256SUMS").read_text(encoding="utf-8").count("\n") == 3


def test_download_verifies_and_skips(tmp_path):
    manifest, folder = _release(tmp_path)
    dest = tmp_path / "weights"
    base = folder.as_uri() + "/"
    assert missing_weights(dest, manifest) == [a["name"] for a in manifest["assets"]]
    first = download_weights(dest, base, manifest, backoff=0)
    assert all("downloaded" in r["status"] for r in first)
    assert missing_weights(dest, manifest) == []
    again = download_weights(dest, base, manifest, backoff=0)
    assert all("already present" in r["status"] for r in again)


def test_download_rejects_tampered_file(tmp_path):
    manifest, folder = _release(tmp_path)
    (folder / "model_fp32.pt").write_bytes(b"tampered")
    dest = tmp_path / "weights"
    with pytest.raises(ChecksumMismatch) as exc:
        download_weights(dest, folder.as_uri() + "/", manifest, backoff=0)
    assert "download-weights --force" in str(exc.value)
    assert not (dest / "model_fp32.pt").exists() and not list(dest.glob("*.part"))


def test_download_network_failure_is_clear(tmp_path):
    manifest, _ = _release(tmp_path)
    with pytest.raises(DownloadError) as exc:
        download_weights(tmp_path / "w", "http://127.0.0.1:9/", manifest, retries=2, backoff=0,
                         timeout=2)  # fmt: skip
    assert "retry" in str(exc.value) and "after 2 attempts" in str(exc.value)


def test_unreleased_manifest_is_explained(tmp_path):
    with pytest.raises(DownloadError, match="no released weights"):
        download_weights(tmp_path, manifest={"repo": "x/y", "tag": None, "assets": []})


def test_packaged_manifest_is_valid():
    from deeptrace.export.download import load_manifest

    m = load_manifest()
    assert m["repo"] == "Aditya-7826/deeptrace" and isinstance(m["assets"], list)
