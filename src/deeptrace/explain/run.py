"""`deeptrace explain`: sanity checks, gallery, or a single-image heatmap."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deeptrace.data.datasets import read_manifest, select_rows  # noqa: E402
from deeptrace.data.faces import FaceDetector, crop_box, crop_face, filter_faces  # noqa: E402
from deeptrace.data.io import load_image  # noqa: E402
from deeptrace.data.preprocess import decode_jpeg, standardize, to_model_input  # noqa: E402
from deeptrace.eval.evaluate import clean_json, training_datasets  # noqa: E402
from deeptrace.eval.metrics import label_predictions, sigmoid  # noqa: E402
from deeptrace.explain.gradcam import (  # noqa: E402
    Explainer,
    cam_to_original,
    overlay,
    overlay_on_original,
)
from deeptrace.explain.sanity import energy_shares, randomization_test  # noqa: E402
from deeptrace.models.factory import create_model  # noqa: E402
from deeptrace.models.load import LoadedModel, load_trained  # noqa: E402
from deeptrace.utils import resolve_device, set_cpu_threads  # noqa: E402

LOG = logging.getLogger("deeptrace.explain")


def _tensor(rgbs: list[np.ndarray], lm: LoadedModel, device: str) -> torch.Tensor:
    return torch.from_numpy(np.stack([to_model_input(r, lm.mean, lm.std) for r in rgbs])).to(device)


def cams_for(explainer: Explainer, rgbs: list[np.ndarray], lm: LoadedModel, device: str,
             batch: int = 16, target: str = "predicted") -> np.ndarray:  # fmt: skip
    out = [explainer(_tensor(rgbs[i : i + batch], lm, device), target)
           for i in range(0, len(rgbs), batch)]  # fmt: skip
    return np.concatenate(out) if out else np.zeros((0, 224, 224), np.float32)


def run_sanity(run_dir: Path, data_root: Path, n: int, device: str, seed: int = 0) -> dict:
    lm = load_trained(run_dir / "best.pt", device)
    rows = read_manifest(data_root)
    names = training_datasets(run_dir, rows)
    sample = select_rows(rows, "test", names, n, seed)
    if not sample:
        raise ValueError(f"no test rows for {names} in {data_root}")
    rgbs = [decode_jpeg((data_root / r["path"]).read_bytes()) for r in sample]
    trained = cams_for(Explainer(lm.model), rgbs, lm, device, target="fake")
    torch.manual_seed(seed)
    random_model = create_model(lm.config).to(device).eval()  # same arch, random weights
    randomized = cams_for(Explainer(random_model), rgbs, lm, device, target="fake")
    return {
        "run": lm.meta.get("run_name") or run_dir.name,
        "model": lm.config.name,
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_images": len(sample),
        "target": "fake logit",
        "randomization_test": randomization_test(trained, randomized),
        "energy": energy_shares(trained),
        "note": "Energy shares assume face-crop inputs (1.3x margin); "
                "they are not meaningful for the full-frame ablation.",
    }  # fmt: skip


def merge_sanity(results_dir: Path) -> Path:
    """Rebuild results/explain_sanity.json from every results/<run>/explain_sanity.json."""
    runs = {}
    for f in sorted(results_dir.glob("*/explain_sanity.json")):
        runs[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    path = results_dir / "explain_sanity.json"
    path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    return path


def write_sanity(results_dir: Path, report: dict) -> Path:
    per_run = results_dir / report["run"] / "explain_sanity.json"
    per_run.parent.mkdir(parents=True, exist_ok=True)
    per_run.write_text(json.dumps(clean_json(report), indent=2), encoding="utf-8")
    return merge_sanity(results_dir)


def pick_gallery(preds: list[dict[str, str]], threshold: float, band, k: int = 4) -> dict:
    probs = np.array([float(p["prob"]) for p in preds])
    labels = label_predictions(probs, threshold, band)
    truth = [p["label"] for p in preds]
    idx = np.arange(len(preds))
    conf = np.abs(probs - threshold)
    correct = [i for i in idx[np.argsort(-conf)] if labels[i] == truth[i]]
    real_ok = [i for i in correct if truth[i] == "real"][: k // 2]
    fake_ok = [i for i in correct if truth[i] == "fake"][: k - len(real_ok)]
    wrong = [i for i in idx[np.argsort(-conf)] if labels[i] not in ("uncertain", truth[i])][:k]
    uncertain = [i for i in idx[np.argsort(np.abs(probs - 0.5))] if labels[i] == "uncertain"][:k]
    return {"correct": real_ok + fake_ok, "wrong": wrong, "uncertain": uncertain}


def make_gallery(run_dir: Path, data_root: Path, metrics_path: Path, out_png: Path,
                 device: str) -> Path:  # fmt: skip
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found: run `deeptrace evaluate` first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    calib = metrics["calibration"]
    in_set = next(iter(metrics["in_dataset"]))
    pred_path = run_dir / "eval" / f"predictions_{in_set}.csv"
    with pred_path.open(encoding="utf-8", newline="") as fh:
        preds = list(csv.DictReader(fh))
    picks = pick_gallery(preds, calib["threshold"], calib["uncertain_band"])
    lm = load_trained(run_dir / "best.pt", device)
    explainer = Explainer(lm.model)
    groups = [("correct", "Correct"), ("wrong", "Wrong"), ("uncertain", "Uncertain")]
    cols = 4
    fig, axes = plt.subplots(len(groups), cols, figsize=(cols * 2.6, len(groups) * 2.9))
    for row, (key, title) in enumerate(groups):
        chosen = picks[key]
        rgbs = [decode_jpeg((data_root / preds[i]["path"]).read_bytes()) for i in chosen]
        cams = cams_for(explainer, rgbs, lm, device) if rgbs else []
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                p = preds[chosen[col]]
                ax.imshow(overlay(rgbs[col], cams[col]))
                ax.set_title(f"true {p['label']}, P(fake)={float(p['prob']):.2f}", fontsize=8)
            elif col == 0:
                ax.text(0.5, 0.5, f"no {key} examples", ha="center", va="center", fontsize=9)
        axes[row, 0].text(-0.12, 0.5, title, transform=axes[row, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=11, weight="bold")  # fmt: skip
    fig.suptitle(f"Grad-CAM ({lm.config.name}, {in_set} test): heat = evidence for the "
                 "predicted class", fontsize=10)  # fmt: skip
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def explain_image(run_dir: Path, image: Path, out_png: Path, device: str,
                  temperature: float = 1.0) -> list[dict[str, Any]]:  # fmt: skip
    """Detect faces in one image, explain each, and draw heatmaps on the original."""
    rgb, _ = load_image(image)
    faces, reason = filter_faces(FaceDetector(device).detect(rgb))
    if not faces:
        raise ValueError(f"no usable face in {image}: {reason}")
    lm = load_trained(run_dir / "best.pt", device)
    explainer = Explainer(lm.model)
    canvas, results = rgb.copy(), []
    for face in faces:
        crop = standardize(crop_face(rgb, face)[0])  # same pixels the model sees in serving
        x = _tensor([crop], lm, device)
        with torch.no_grad():
            logit = float(lm.model(x).squeeze())
        cam = explainer(x)[0]
        box = crop_box(face.box)
        canvas = overlay_on_original(canvas, cam_to_original(cam, box, rgb.shape), box)
        results.append({"box": [round(v, 1) for v in face.box],
                        "prob_fake": float(sigmoid(np.array(logit / temperature)))})  # fmt: skip
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png, canvas)
    return results


def run_explain(run_dir: Path, data_root: Path, results_dir: Path, n: int, gallery: Path | None,
                device: str = "auto", threads: int = 4) -> dict:  # fmt: skip
    set_cpu_threads(threads)
    device = resolve_device(device)
    report = run_sanity(run_dir, data_root, n, device)
    path = write_sanity(results_dir, report)
    LOG.info(
        "sanity: randomization passed=%s, face energy share %.3f -> %s",
        report["randomization_test"]["passed"],
        report["energy"]["mean_face_share"],
        path,
    )
    if gallery is not None:
        metrics = results_dir / report["run"] / "metrics.json"
        LOG.info("gallery -> %s", make_gallery(run_dir, data_root, metrics, gallery, device))
    return report
