"""`deeptrace export`: ONNX FP32 export + parity, INT8 static quantization with an AUC gate."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from deeptrace.data.datasets import LABELS, read_manifest, select_rows
from deeptrace.data.preprocess import decode_jpeg, to_model_input
from deeptrace.eval.evaluate import clean_json, training_datasets
from deeptrace.eval.metrics import auc, sigmoid
from deeptrace.export.onnx_export import (
    export_onnx,
    onnx_logits,
    ort_session,
    parity,
    torch_logits,
    torch_logits_fp64,
)
from deeptrace.export.quantize import (
    ATTEMPTS,
    MAX_AUC_DROP,
    ArrayReader,
    Attempt,
    choose,
    quantize,
    sensitive_nodes,
)
from deeptrace.models.load import load_trained
from deeptrace.utils import set_cpu_threads

LOG = logging.getLogger("deeptrace.export")


@dataclass
class ExportOptions:
    run_dir: Path
    data_root: Path
    out_dir: Path  # artifacts/export/<run>
    results_dir: Path = Path("results")
    threads: int = 4
    parity_n: int = 100
    calib_n: int = 300
    max_test: int | None = None  # test crops for the FP32 vs INT8 comparison; None = all
    batch: int = 16
    seed: int = 0
    parity_tol: float = 1e-4


def batches(rows: list[dict[str, str]], root: Path, mean, std, size: int) -> Iterator[np.ndarray]:
    """Stream preprocessed crops from disk in batches (never the whole set in memory)."""
    for i in range(0, len(rows), size):
        chunk = rows[i : i + size]
        yield np.stack([to_model_input(decode_jpeg((root / r["path"]).read_bytes()), mean, std)
                        for r in chunk])  # fmt: skip


def load_calibration(results_dir: Path, run: str) -> dict[str, Any]:
    path = results_dir / run / "metrics.json"
    if path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))["calibration"]
        return {"temperature": c["temperature"], "threshold": c["threshold"],
                "uncertain_band": c["uncertain_band"], "calibrated": True,
                "source": str(path)}  # fmt: skip
    LOG.warning("%s missing: run `deeptrace evaluate` first; using T=1, threshold 0.5", path)
    return {"temperature": 1.0, "threshold": 0.5, "uncertain_band": [0.4, 0.6],
            "calibrated": False, "source": None}  # fmt: skip


def run_export(opts: ExportOptions) -> dict[str, Any]:
    set_cpu_threads(opts.threads)
    lm = load_trained(opts.run_dir / "best.pt", "cpu")
    run = lm.meta.get("run_name") or opts.run_dir.name
    rows = read_manifest(opts.data_root)
    names = training_datasets(opts.run_dir, rows)
    test_rows = select_rows(rows, "test", names, opts.max_test, opts.seed)
    calib_rows = select_rows(rows, "val", names, opts.calib_n, opts.seed)
    parity_rows = select_rows(rows, "test", names, opts.parity_n, opts.seed) or calib_rows
    if not test_rows or not calib_rows:
        raise ValueError(f"need val and test rows for {names} in {opts.data_root}")
    out = opts.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def feed(rs, size=opts.batch):
        return batches(rs, opts.data_root, lm.mean, lm.std, size)

    # 1) FP32 export + parity on up to 100 images
    fp32_path = export_onnx(lm.model, out / "model_fp32.onnx", lm.config.img_size)
    fp32 = ort_session(fp32_path, opts.threads)
    par = parity(
        torch_logits(lm.model, feed(parity_rows)),
        onnx_logits(fp32, feed(parity_rows)),
        opts.parity_tol,
        torch_logits_fp64(lm.model, feed(parity_rows)),
    )
    LOG.info("parity on %d images: max |diff| %.2e (tol %.0e)", par["n"], par["max_abs_diff"],
             par["tolerance"])  # fmt: skip
    if not par["passed"]:
        fail = opts.results_dir / run / "export.json"
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(json.dumps(clean_json({"run": run, "parity": par, "serving_model": None,
                                               "decision": "parity failed: nothing exported"}),
                                   indent=2), encoding="utf-8")  # fmt: skip
        (out / "model_fp32.onnx").unlink(missing_ok=True)
        raise RuntimeError(f"ONNX FP32 parity failed (details in {fail}): {par}")

    # 2) INT8 attempts, compared with ONNX FP32 on the test set
    calib = load_calibration(opts.results_dir, run)
    y = np.array([LABELS[r["label"]] for r in test_rows])
    z32 = onnx_logits(fp32, feed(test_rows))
    auc32 = auc(y, z32)
    labels32 = sigmoid(z32 / calib["temperature"]) >= calib["threshold"]
    reader = ArrayReader(lambda: feed(calib_rows, 4))
    attempts: list[Attempt] = []
    for name, per_channel, exclude in ATTEMPTS:
        a = Attempt(name, per_channel, exclude)
        attempts.append(a)
        path = out / f"model_int8_{name}.onnx"
        try:
            a.nodes_excluded = sensitive_nodes(fp32_path, exclude)
            quantize(fp32_path, path, reader, per_channel, a.nodes_excluded)
            z8 = onnx_logits(ort_session(path, opts.threads), feed(test_rows))
            a.auc_int8 = auc(y, z8)
            a.auc_drop = auc32 - a.auc_int8
            labels8 = sigmoid(z8 / calib["temperature"]) >= calib["threshold"]
            a.agreement = float((labels8 == labels32).mean())
            a.mean_abs_logit_diff = float(np.abs(z8 - z32).mean())
        except Exception as exc:
            a.error = f"{type(exc).__name__}: {exc}"
            LOG.warning("INT8 attempt %s failed: %s", name, a.error)
        LOG.info("INT8 %s: AUC %s (drop %s)", name, a.auc_int8, a.auc_drop)
        if choose([a]) is not None:
            break
    chosen = next((a for a in attempts if a.accepted), None)

    # Keep only the shipped INT8 model.
    for a in attempts:
        p = out / f"model_int8_{a.name}.onnx"
        if chosen is not None and a is chosen:
            p.replace(out / "model_int8.onnx")
        elif p.exists():
            p.unlink()
    if chosen is None and (out / "model_int8.onnx").exists():
        (out / "model_int8.onnx").unlink()  # never leave a stale, rejected INT8 model around
    if chosen is not None:
        serving, reason = "model_int8.onnx", f"INT8 '{chosen.name}' within {MAX_AUC_DROP} AUC"
    else:
        serving = "model_fp32.onnx"
        reason = (f"every INT8 attempt lost more than {MAX_AUC_DROP} test AUC (or failed); "
                  "shipping FP32 ONNX instead")  # fmt: skip
    LOG.info("serving model: %s (%s)", serving, reason)

    # 3) FP32 PyTorch weights for on-demand Grad-CAM + serving metadata
    shutil.copy2(opts.run_dir / "best.pt", out / "model_fp32.pt")
    commit = lm.meta.get("git_commit") or "unknown"
    meta = {
        "run": run,
        "version": f"{run}-{commit[:7]}",
        "model_name": lm.config.name,
        "arch": lm.config.arch,
        "input_size": lm.config.img_size,
        "mean": list(lm.mean),
        "std": list(lm.std),
        **{k: calib[k] for k in ("temperature", "threshold", "uncertain_band", "calibrated")},
        "serving_model": serving,
        "gradcam_model": "model_fp32.pt",
        "git_commit": commit,
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = clean_json({
        "run": run,
        "exported_utc": meta["exported_utc"],
        "parity": par,
        "test_set": {"datasets": names, "n": len(test_rows), "auc_onnx_fp32": auc32},
        "calibration_images": len(calib_rows),
        "calibration": calib,
        "int8_attempts": [asdict(a) for a in attempts],
        "max_auc_drop": MAX_AUC_DROP,
        "serving_model": serving,
        "decision": reason,
        "files_mb": {p.name: round(p.stat().st_size / 1e6, 2)
                     for p in sorted(out.glob("model*")) if p.is_file()},
    })  # fmt: skip
    dest = opts.results_dir / run / "export.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
