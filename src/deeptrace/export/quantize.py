"""INT8 static quantization with an accuracy gate.

Attempts go from simplest to most conservative. The first attempt whose test
AUC drop vs ONNX FP32 is <= MAX_AUC_DROP is shipped. If none passes, FP32 ONNX
is shipped and the reason is recorded. A degraded model is never shipped silently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("deeptrace.export")
MAX_AUC_DROP = 0.01


class ArrayReader:
    """onnxruntime CalibrationDataReader over a list of preprocessed batches (streamed)."""

    def __init__(self, batches: Callable[[], list[np.ndarray]] | list[np.ndarray]) -> None:
        self._batches = batches
        self._it = None

    def get_next(self):
        if self._it is None:
            data = self._batches() if callable(self._batches) else self._batches
            self._it = iter(data)
        batch = next(self._it, None)
        return None if batch is None else {"input": batch.astype(np.float32)}

    def rewind(self) -> None:
        self._it = None


@dataclass
class Attempt:
    name: str
    per_channel: bool
    exclude: str  # "none", "edges", "edges+depthwise"
    nodes_excluded: list[str] = field(default_factory=list)
    auc_int8: float | None = None
    auc_drop: float | None = None
    agreement: float | None = None  # same label as FP32 at the serving threshold
    mean_abs_logit_diff: float | None = None
    accepted: bool = False
    error: str | None = None


ATTEMPTS = (
    ("per_tensor", False, "none"),
    ("per_channel", True, "none"),
    ("per_channel_excl_edges", True, "edges"),
    ("per_channel_excl_edges_depthwise", True, "edges+depthwise"),
)


def sensitive_nodes(onnx_path: Path, mode: str) -> list[str]:
    """Nodes to keep in FP32: the stem conv, the classifier and the last conv before
    pooling ("edges"), optionally all depthwise convs (group > 1), which are the most
    quantization-sensitive layers in EfficientNet."""
    if mode == "none":
        return []
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    compute = [n for n in graph.node if n.op_type in ("Conv", "Gemm", "MatMul")]
    names = set()
    if compute:
        names.add(compute[0].name)
        names.update(n.name for n in compute[-2:])
    if "depthwise" in mode:
        for n in graph.node:
            if n.op_type == "Conv":
                group = next((a.i for a in n.attribute if a.name == "group"), 1)
                if group > 1:
                    names.add(n.name)
    return sorted(names)


def quantize(fp32_path: Path, out_path: Path, reader: ArrayReader, per_channel: bool,
             exclude: list[str]) -> Path:  # fmt: skip
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = out_path.with_name(out_path.stem + "_prep.onnx")
    try:
        quant_pre_process(str(fp32_path), str(prep))
        source = prep
    except Exception as exc:  # pre-processing is an optimisation, not required
        LOG.warning("quant_pre_process failed (%s); quantizing the raw graph", exc)
        source = fp32_path
    reader.rewind()
    quantize_static(
        str(source), str(out_path), reader, quant_format=QuantFormat.QDQ,
        per_channel=per_channel, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude or None, calibrate_method=CalibrationMethod.MinMax,
        # merge activation ranges every few batches instead of holding all 300 images'
        # intermediate tensors in RAM (8 GB laptop)
        extra_options={"CalibMaxIntermediateOutputs": 8},
    )  # fmt: skip
    if prep.exists():
        prep.unlink()
    return out_path


def choose(attempts: list[Attempt], max_drop: float = MAX_AUC_DROP) -> Attempt | None:
    """First attempt (in order) within the AUC budget, else None (ship FP32)."""
    for a in attempts:
        if a.error is None and a.auc_drop is not None and a.auc_drop <= max_drop:
            a.accepted = True
            return a
    return None
