"""`deeptrace evaluate`: in-dataset, cross-dataset, per-method, robustness and calibration.

Order matters: validation predictions are computed first, the temperature and
threshold are fitted on them and frozen, and only then is any test row loaded.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from deeptrace.data.datasets import LABELS, CropDataset, read_manifest, select_rows
from deeptrace.eval import plots
from deeptrace.eval.calibration import Calibration, fit_calibration
from deeptrace.eval.metrics import (
    auc,
    binary_metrics,
    bootstrap_auc_ci,
    ece,
    sigmoid,
    uncertain_rate,
)
from deeptrace.eval.robustness import PERTURBATIONS, Perturb
from deeptrace.models.load import LoadedModel, load_trained
from deeptrace.train.loop import predict_logits
from deeptrace.utils import resolve_device, set_cpu_threads

LOG = logging.getLogger("deeptrace.eval")


@dataclass
class EvalOptions:
    run_dir: Path
    data_roots: list[Path]  # first = the training manifest; others add cross-dataset tests
    out_dir: Path
    batch_size: int = 64
    num_workers: int = 2
    device: str = "auto"
    threads: int = 4
    max_test: int | None = None  # cap per test set (stratified); None = all
    robustness_max: int | None = 2000  # cap for robustness curves; None = all
    bootstrap: int = 1000
    band: tuple[float, float] = (0.4, 0.6)
    seed: int = 0
    predictions_dir: Path | None = None  # defaults to <run_dir>/eval
    _timings: dict[str, float] = field(default_factory=dict)


def clean_json(obj: Any) -> Any:
    """NaN/inf -> None so metrics.json is valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    return obj


def training_datasets(run_dir: Path, train_rows: list[dict[str, str]]) -> list[str]:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        names = (cfg.get("data") or {}).get("datasets")
        if names:
            return list(names)
    return sorted({r["source_dataset"] for r in train_rows if r["split"] == "train"})


class Evaluator:
    def __init__(self, opts: EvalOptions) -> None:
        self.opts = opts
        set_cpu_threads(opts.threads)
        self.device = resolve_device(opts.device)
        self.amp = self.device.startswith("cuda")
        self.lm: LoadedModel = load_trained(opts.run_dir / "best.pt", self.device)
        self.pred_dir = opts.predictions_dir or (opts.run_dir / "eval")

    def predict(self, rows: list[dict[str, str]], root: Path, transform=None) -> np.ndarray:
        ds = CropDataset(rows, root, transform, self.lm.mean, self.lm.std)
        logits, _ = predict_logits(self.lm.model, ds, self.opts.batch_size, self.device, self.amp,
                                   self.opts.num_workers)  # fmt: skip
        return logits

    @staticmethod
    def labels(rows: list[dict[str, str]]) -> np.ndarray:
        return np.array([LABELS[r["label"]] for r in rows], dtype=np.int64)

    def calibrate(self, val_rows: list[dict[str, str]], root: Path) -> Calibration:
        logits = self.predict(val_rows, root)
        calib = fit_calibration(logits, self.labels(val_rows), self.opts.band)
        LOG.info("calibration on %d val crops: T=%.3f threshold=%.3f", len(val_rows),
                 calib.temperature, calib.threshold)  # fmt: skip
        self.val_logits = logits
        return calib

    def score_set(
        self, name: str, rows: list[dict[str, str]], logits: np.ndarray, calib: Calibration
    ) -> dict[str, Any]:
        y = self.labels(rows)
        raw = sigmoid(logits)
        probs = calib.probs(logits)
        m = binary_metrics(y, probs, calib.threshold)
        lo, hi = bootstrap_auc_ci(y, probs, self.opts.bootstrap, seed=self.opts.seed)
        sure = ~((probs >= calib.uncertain_band[0]) & (probs <= calib.uncertain_band[1]))
        sure_acc = float("nan")
        if sure.any():
            sure_acc = float(((probs[sure] >= calib.threshold) == y[sure]).mean())
        m.update({
            "auc_ci95": [lo, hi],
            "ece_before": ece(y, raw),
            "ece_after": ece(y, probs),
            "uncertain_rate": uncertain_rate(probs, calib.uncertain_band),
            "accuracy_excluding_uncertain": sure_acc,
        })  # fmt: skip
        per_method: dict[str, Any] = {}
        reals = y == 0
        methods = sorted({r["manipulation_method"] for r in rows if r["label"] == "fake"})
        for method in methods:
            fakes = np.array([r["label"] == "fake" and r["manipulation_method"] == method
                              for r in rows])  # fmt: skip
            mask = reals | fakes
            per_method[method] = {
                "n_fake": int(fakes.sum()),
                "auc_vs_all_real": auc(y[mask], probs[mask]),
                "recall": float((probs[fakes] >= calib.threshold).mean()),
            }
        m["per_manipulation"] = per_method
        self._save_predictions(name, rows, logits, probs)
        return m

    def _save_predictions(self, name, rows, logits, probs) -> None:
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        with (self.pred_dir / f"predictions_{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label", "manipulation_method", "logit", "prob"])
            for r, z, p in zip(rows, logits, probs, strict=True):
                w.writerow(
                    [r["path"], r["label"], r["manipulation_method"], f"{z:.6f}", f"{p:.6f}"]
                )

    def robustness(
        self, rows: list[dict[str, str]], root: Path, calib: Calibration
    ) -> dict[str, list[dict[str, Any]]]:
        y = self.labels(rows)
        out: dict[str, list[dict[str, Any]]] = {}
        for family, levels in PERTURBATIONS.items():
            out[family] = []
            for label, value, fn in levels:
                probs = calib.probs(self.predict(rows, root, Perturb(fn)))
                m = binary_metrics(y, probs, calib.threshold)
                out[family].append({
                    "level": label, "value": value, "auc": m["auc"], "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"], "recall": m["recall"],
                    "ece_after": ece(y, probs),
                    "uncertain_rate": uncertain_rate(probs, calib.uncertain_band),
                })  # fmt: skip
                LOG.info("robustness %s %s: AUC %.4f", family, label, m["auc"])
        return out


def run_evaluate(opts: EvalOptions) -> dict[str, Any]:
    t0 = time.time()
    ev = Evaluator(opts)
    roots = [Path(r) for r in opts.data_roots]
    manifests = {root: read_manifest(root) for root in roots}
    train_root = roots[0]
    in_names = training_datasets(opts.run_dir, manifests[train_root])

    # 1) Validation only: fit and freeze calibration.
    val_rows = select_rows(manifests[train_root], "val", in_names)
    if not val_rows:
        raise ValueError(f"no validation rows for {in_names} in {train_root}")
    calib = ev.calibrate(val_rows, train_root)

    # 2) Test sets, loaded only after calibration is fixed.
    test_sets: dict[str, tuple[list[dict[str, str]], Path]] = {}
    for root in roots:
        for name in sorted({r["source_dataset"] for r in manifests[root] if r["split"] == "test"}):
            if name in test_sets:
                continue
            rows = select_rows(manifests[root], "test", [name], opts.max_test, opts.seed)
            if rows:
                test_sets[name] = (rows, root)
    if not test_sets:
        raise ValueError("no test rows found in any data root")

    results: dict[str, Any] = {"in_dataset": {}, "cross_dataset": {}}
    curves, set_logits = {}, {}
    for name, (rows, root) in test_sets.items():
        logits = ev.predict(rows, root)
        set_logits[name] = logits
        m = ev.score_set(name, rows, logits, calib)
        group = "in_dataset" if name in in_names else "cross_dataset"
        results[group][name] = m
        curves[name] = (ev.labels(rows), calib.probs(logits), m["auc"])
        LOG.info("%s [%s] n=%d AUC %.4f [%.4f, %.4f]", name, group, m["n"], m["auc"],
                 *m["auc_ci95"])  # fmt: skip

    out = opts.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plots.plot_roc(curves, out / "roc.png")

    robust, robust_set = {}, None
    in_sets = [n for n in in_names if n in test_sets]
    if in_sets:
        robust_set = in_sets[0]
        rows, root = test_sets[robust_set]
        sub = select_rows(rows, "test", None, opts.robustness_max, opts.seed)
        robust = ev.robustness(sub, root, calib)
        clean = calib.probs(ev.predict(sub, root))
        plots.plot_robustness(robust, auc(ev.labels(sub), clean), out / "robustness.png")
        labels, logits = curves[robust_set][0], set_logits[robust_set]
        m = results["in_dataset"][robust_set]
        eces = (m["ece_before"], m["ece_after"])
        plots.plot_reliability(
            labels, sigmoid(logits), calib.probs(logits), eces, out / "reliability.png"
        )
        plots.plot_confusion(m["confusion_matrix"], f"{robust_set} test", out / "confusion.png")

    gaps = {}
    for in_name, in_m in results["in_dataset"].items():
        for x_name, x_m in results["cross_dataset"].items():
            if in_m["auc"] is not None and x_m["auc"] is not None:
                gaps[f"{in_name}->{x_name}"] = in_m["auc"] - x_m["auc"]

    val_y = ev.labels(val_rows)
    report = {
        "run": ev.lm.meta.get("run_name") or opts.run_dir.name,
        "checkpoint": {**ev.lm.meta, "model": ev.lm.config.name, "arch": ev.lm.config.arch},
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": ev.device,
        "data_roots": [str(r) for r in roots],
        "training_datasets": in_names,
        "calibration": {
            **calib.to_dict(),
            "val_ece_before": ece(val_y, sigmoid(ev.val_logits)),
            "val_ece_after": ece(val_y, calib.probs(ev.val_logits)),
            "val_auc": auc(val_y, ev.val_logits),
        },
        **results,
        "generalization_gap_auc": gaps,
        "robustness": {
            "test_set": robust_set,
            "n": len(sub) if robust_set else 0,
            "families": robust,
        },  # fmt: skip
        "options": {
            "max_test": opts.max_test,
            "robustness_max": opts.robustness_max,
            "bootstrap": opts.bootstrap,
            "seed": opts.seed,
        },  # fmt: skip
        "seconds": round(time.time() - t0, 1),
    }
    report = clean_json(report)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s", out / "metrics.json")
    return report
