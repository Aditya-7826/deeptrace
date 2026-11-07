"""PyTorch -> ONNX export (dynamic batch axis), ONNX Runtime sessions and parity checks."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

INPUT_NAME, OUTPUT_NAME = "input", "logit"
OPSET = 17


def export_onnx(model, path: Path, img_size: int = 224, opset: int = OPSET) -> Path:
    """Export with the TorchScript-based exporter (no extra dependency).
    torch.fft (the optional rgb_fft model) is not exportable this way."""
    import torch

    from deeptrace.models.two_branch import RGBFFTNet

    if isinstance(model, RGBFFTNet):
        raise NotImplementedError("the rgb_fft model uses torch.fft, which this exporter "
                                  "does not support; serve a single-branch model")  # fmt: skip
    model = model.eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    dummy = torch.zeros(1, 3, img_size, img_size)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")  # legacy-exporter deprecation notice
        torch.onnx.export(
            model, dummy, str(tmp), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )  # fmt: skip
    import onnx

    onnx.checker.check_model(str(tmp))
    tmp.replace(path)
    return path


def ort_session(path: Path, threads: int = 4):
    """CPU session with explicit thread counts (8 GB laptop rule)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_logits(session, batches: Iterator[np.ndarray]) -> np.ndarray:
    out = [session.run([OUTPUT_NAME], {INPUT_NAME: b.astype(np.float32)})[0].reshape(-1)
           for b in batches]  # fmt: skip
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def torch_logits(model, batches: Iterator[np.ndarray]) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for b in batches:
            out.append(model(torch.from_numpy(b.astype(np.float32))).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def parity(
    torch_out: np.ndarray, onnx_out: np.ndarray, tol: float = 1e-4, fp64: np.ndarray | None = None
) -> dict:
    """Max |PyTorch FP32 - ONNX FP32| over the logits. With a float64 PyTorch reference,
    also report how far each FP32 output is from it: if both are about equally far,
    a failure is float32 rounding (large logits), not an export bug."""
    diff = np.abs(torch_out.astype(np.float64) - onnx_out.astype(np.float64))
    out = {
        "n": int(len(diff)),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "max_abs_logit": float(np.abs(torch_out).max()) if len(diff) else 0.0,
        "tolerance": tol,
        "passed": bool(len(diff) and diff.max() <= tol),
    }
    if fp64 is not None and len(diff):
        out["torch_fp32_vs_fp64_max"] = float(np.abs(torch_out - fp64).max())
        out["onnx_fp32_vs_fp64_max"] = float(np.abs(onnx_out - fp64).max())
    return out


def torch_logits_fp64(model, batches: Iterator[np.ndarray]) -> np.ndarray:
    """Float64 reference on a copy of the model (the original stays FP32)."""
    import copy

    import torch

    ref = copy.deepcopy(model).double().eval()
    out = []
    with torch.no_grad():
        for b in batches:
            out.append(ref(torch.from_numpy(b.astype(np.float64))).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float64)
