"""`deeptrace benchmark`: batch-1 CPU latency (p50/p95) and peak RSS on this machine.

Every configuration runs in a fresh subprocess, so its peak RSS is its own.
Configurations: {PyTorch FP32, ONNX FP32, ONNX INT8} x {model only, end to end}.
End to end = JPEG bytes -> decode -> MTCNN face detection -> crop -> standardize ->
normalize -> model, exactly the serving path.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def peak_rss_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = ru if sys.platform == "darwin" else ru * 1024  # macOS: bytes, Linux: KiB
    return peak / 1e6


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")  # fmt: skip
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)  # fmt: skip
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def system_info() -> dict[str, Any]:
    import psutil

    return {
        "cpu": cpu_name(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def percentiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "min_ms": float(a.min())}  # fmt: skip


def _worker(kind: str, pipeline: str, export_dir: Path, image: Path, crop: Path, runs: int,
            warmup: int, threads: int) -> dict[str, Any]:  # fmt: skip
    """Runs inside the subprocess."""
    meta = json.loads((export_dir / "model.json").read_text(encoding="utf-8"))
    mean, std = tuple(meta["mean"]), tuple(meta["std"])
    t_load = time.perf_counter()
    if kind == "pytorch_fp32":
        import torch

        from deeptrace.models.load import load_trained

        torch.set_num_threads(threads)
        model = load_trained(export_dir / "model_fp32.pt", "cpu").model

        def infer(x: np.ndarray) -> float:
            with torch.inference_mode():
                return float(model(torch.from_numpy(x)).reshape(-1)[0])
    else:
        from deeptrace.export.onnx_export import INPUT_NAME, ort_session

        name = "model_fp32.onnx" if kind == "onnx_fp32" else "model_int8.onnx"
        sess = ort_session(export_dir / name, threads)

        def infer(x: np.ndarray) -> float:
            return float(sess.run(None, {INPUT_NAME: x})[0].reshape(-1)[0])

    load_ms = (time.perf_counter() - t_load) * 1000
    from deeptrace.data.preprocess import CropSettings, decode_jpeg, extract_crops, to_model_input

    data = image.read_bytes()
    detector = None
    if pipeline == "end_to_end":
        import torch

        from deeptrace.data.faces import FaceDetector
        from deeptrace.data.io import load_image

        torch.set_num_threads(threads)
        detector = FaceDetector("cpu")

        def step() -> float:
            rgb, _ = load_image(data)
            crops, reason = extract_crops(rgb, detector, CropSettings())
            if not crops:
                raise RuntimeError(f"benchmark image has no usable face: {reason}")
            x = to_model_input(decode_jpeg(crops[0].jpeg), mean, std)[None]
            return infer(x)
    else:
        # The crop was made once by the parent, so this process never imports MTCNN/torch
        # unless the model itself is PyTorch: its peak RSS is the model's own footprint.
        x = to_model_input(decode_jpeg(crop.read_bytes()), mean, std)[None]

        def step() -> float:
            return infer(x)

    for _ in range(warmup):
        step()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000)
    return {"model": kind, "pipeline": pipeline, "runs": runs, "warmup": warmup,
            "load_ms": round(load_ms, 1), **percentiles(times),
            "peak_rss_mb": round(peak_rss_mb(), 1)}  # fmt: skip


def run_benchmark(export_dir: Path, image: Path, out: Path, runs: int = 200, warmup: int = 20,
                  threads: int = 4) -> dict[str, Any]:  # fmt: skip
    import onnxruntime
    import torch

    kinds = ["pytorch_fp32", "onnx_fp32"]
    if (export_dir / "model_int8.onnx").exists():
        kinds.append("onnx_int8")
    results, notes = [], []
    if "onnx_int8" not in kinds:
        notes.append("no model_int8.onnx: INT8 was rejected by the accuracy gate (see export.json)")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    from deeptrace.data.faces import FaceDetector
    from deeptrace.data.io import load_image
    from deeptrace.data.preprocess import CropSettings, extract_crops

    crops, reason = extract_crops(load_image(image)[0], FaceDetector("cpu"), CropSettings())
    if not crops:
        raise ValueError(f"{image} has no usable face ({reason}); pick another image")
    crop_path = out.parent / ".benchmark_crop.jpg"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(crops[0].jpeg)
    for pipeline in ("model_only", "end_to_end"):
        for kind in kinds:
            cmd = [sys.executable, "-m", "deeptrace.export.benchmark", "--worker", kind, pipeline,
                   str(export_dir), str(image), str(crop_path), str(runs), str(warmup),
                   str(threads)]  # fmt: skip
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"benchmark worker {kind}/{pipeline} failed:\n{proc.stderr}")
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            results.append(row)
            print(
                f"{kind:13s} {pipeline:11s} p50 {row['p50_ms']:8.2f} ms  p95 "
                f"{row['p95_ms']:8.2f} ms  peak RSS {row['peak_rss_mb']:7.1f} MB",
                flush=True,
            )
    crop_path.unlink(missing_ok=True)
    meta = json.loads((export_dir / "model.json").read_text(encoding="utf-8"))
    from PIL import Image

    with Image.open(image) as im:
        size = list(im.size)
    report = {
        "measured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": system_info(),
        "settings": {"threads": threads, "batch_size": 1, "runs": runs, "warmup": warmup,
                     "image": image.name, "image_size_wh": size,
                     "torch": torch.__version__, "onnxruntime": onnxruntime.__version__},
        "model": {"run": meta["run"], "version": meta["version"], "name": meta["model_name"],
                  "serving_model": meta["serving_model"]},
        "results": results,
        "notes": notes,
    }  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    k, p, d, img, crop, n, w, t = sys.argv[2:10]
    print(json.dumps(_worker(k, p, Path(d), Path(img), Path(crop), int(n), int(w), int(t))))
