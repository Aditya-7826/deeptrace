"""Inference engine: ONNX scoring (loaded once) + FP32 PyTorch Grad-CAM (loaded on first use).

Preprocessing is the training path, imported, not re-implemented:
extract_crops (decode->detect->crop->resize->JPEG q95) -> decode_jpeg -> to_model_input.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from deeptrace.data.faces import FaceDetector, crop_box
from deeptrace.data.preprocess import (
    CropSettings,
    FaceCrop,
    decode_jpeg,
    extract_crops,
    to_model_input,
)
from deeptrace.eval.metrics import label_predictions, sigmoid
from deeptrace.export.download import load_manifest, missing_weights
from deeptrace.export.onnx_export import INPUT_NAME, ort_session

LOG = logging.getLogger("deeptrace.serve")


class MissingWeightsError(RuntimeError):
    pass


def check_weights(weights_dir: Path, verify_checksums: bool = True) -> dict[str, Any]:
    """Return model.json if the serving files are present (and match weights.json)."""
    hint = ("Download them with `deeptrace download-weights` "
            "(or set DEEPTRACE_WEIGHTS_DIR to a folder made by `deeptrace export`).")  # fmt: skip
    meta_path = weights_dir / "model.json"
    if not meta_path.exists():
        raise MissingWeightsError(f"model weights not found in {weights_dir.resolve()}. {hint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in (meta["serving_model"], meta["gradcam_model"]):
        if not (weights_dir / name).exists():
            raise MissingWeightsError(f"{name} is missing from {weights_dir.resolve()}. {hint}")
    manifest = load_manifest()
    names = {a["name"] for a in manifest.get("assets", [])}
    if verify_checksums and meta["serving_model"] in names:  # released weights: verify them
        bad = missing_weights(weights_dir, manifest)
        if bad:
            raise MissingWeightsError(
                f"checksum mismatch for {bad} in {weights_dir.resolve()}. "
                "Re-download with `deeptrace download-weights --force`."
            )
    return meta


def png_base64(rgb: np.ndarray, max_side: int | None = None) -> str:
    img = Image.fromarray(rgb)
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Engine:
    def __init__(self, weights_dir: Path, threads: int = 2, verify_checksums: bool = True) -> None:
        import torch

        torch.set_num_threads(threads)
        self.weights_dir = Path(weights_dir)
        self.meta = check_weights(self.weights_dir, verify_checksums)
        self.mean, self.std = tuple(self.meta["mean"]), tuple(self.meta["std"])
        self.temperature = float(self.meta["temperature"])
        self.threshold = float(self.meta["threshold"])
        self.band = tuple(self.meta["uncertain_band"])
        self.session = ort_session(self.weights_dir / self.meta["serving_model"], threads)
        self.detector = FaceDetector("cpu")
        self.crop_settings = CropSettings()
        self.gradcam = None  # (model, explainer), loaded lazily on the first /explain
        self._gradcam_lock = threading.Lock()

    # ------------------------------------------------------------- shared steps
    def face_crops(self, rgb: np.ndarray) -> tuple[list[FaceCrop], str | None]:
        """All usable faces, largest first (same order for /predict and /explain)."""
        return extract_crops(rgb, self.detector, self.crop_settings, largest_only=False)

    def face_tensors(self, crops: list[FaceCrop]) -> np.ndarray:
        return np.stack([to_model_input(decode_jpeg(c.jpeg), self.mean, self.std) for c in crops])

    def _result(self, crop: FaceCrop, logit: float, index: int) -> dict[str, Any]:
        prob = float(sigmoid(np.array(logit / self.temperature)))
        return {
            "index": index,
            "box": [round(v, 1) for v in crop.face.box],
            "detection_confidence": round(crop.face.confidence, 4),
            "probability_fake": round(prob, 4),
            "label": label_predictions(np.array([prob]), self.threshold, self.band)[0],
        }

    # ------------------------------------------------------------- endpoints
    def predict(self, rgb: np.ndarray) -> dict[str, Any]:
        crops, reason = self.face_crops(rgb)
        if not crops:
            return {"status": reason if reason == "face_too_small" else "no_face"}
        logits = self.session.run(None, {INPUT_NAME: self.face_tensors(crops)})[0].reshape(-1)
        return {"status": "ok",
                "faces": [self._result(c, float(z), i) for i, (c, z) in
                          enumerate(zip(crops, logits, strict=True))]}  # fmt: skip

    def _load_gradcam(self):
        if self.gradcam is None:
            from deeptrace.explain.gradcam import Explainer
            from deeptrace.models.load import load_trained

            LOG.info("loading FP32 Grad-CAM model (first /explain call)")
            model = load_trained(self.weights_dir / self.meta["gradcam_model"], "cpu").model
            self.gradcam = (model, Explainer(model))
        return self.gradcam

    def explain(self, rgb: np.ndarray, face_index: int) -> dict[str, Any]:
        import torch

        from deeptrace.explain.gradcam import cam_to_original, overlay, overlay_on_original

        crops, reason = self.face_crops(rgb)
        if not crops:
            return {"status": reason if reason == "face_too_small" else "no_face"}
        if not 0 <= face_index < len(crops):
            return {"status": "bad_face_index", "faces_found": len(crops)}
        crop = crops[face_index]
        x = self.face_tensors([crop])
        logit = float(self.session.run(None, {INPUT_NAME: x})[0].reshape(-1)[0])
        with self._gradcam_lock:  # Grad-CAM hooks are not thread-safe
            _, explainer = self._load_gradcam()
            cam = explainer(torch.from_numpy(x), target="predicted")[0]
        face_rgb = decode_jpeg(crop.jpeg)
        box = crop_box(crop.face.box)
        full = overlay_on_original(rgb, cam_to_original(cam, box, rgb.shape), box)
        return {
            "status": "ok",
            "face": self._result(crop, logit, face_index),
            "heatmap_target": "evidence for the predicted class (Grad-CAM, FP32 model)",
            "crop_overlay_png": png_base64(overlay(face_rgb, cam)),
            "image_overlay_png": png_base64(full, max_side=800),
        }
