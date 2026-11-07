"""Evaluation: metric functions on known inputs, calibration, and the full evaluate run."""

from __future__ import annotations

import csv
import json
import pickle
import shutil

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deeptrace.eval.calibration import choose_threshold, fit_calibration, fit_temperature
from deeptrace.eval.compare import NYM, build_table
from deeptrace.eval.evaluate import EvalOptions, Evaluator, run_evaluate
from deeptrace.eval.metrics import (
    auc,
    binary_metrics,
    bootstrap_auc_ci,
    ece,
    label_predictions,
    sigmoid,
    uncertain_rate,
)
from deeptrace.eval.robustness import PERTURBATIONS, Perturb
from deeptrace.train.loop import Trainer
from tests.test_train import tiny_cfg

# --------------------------------------------------------------- metric functions


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) + 0.3 * y, 1)  # rounding creates many ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert np.isnan(auc([1, 1], [0.2, 0.3]))


def test_binary_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    m = binary_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]  # [[TN, FP], [FN, TP]]
    assert m["accuracy"] == 0.75 and m["precision"] == 0.75 and m["recall"] == 0.75
    assert m["f1"] == pytest.approx(0.75)
    assert m["auc"] == pytest.approx(14 / 16)


def test_ece_known_values():
    assert ece(np.array([1, 0, 1, 0]), np.full(4, 0.8)) == pytest.approx(0.3)
    assert ece(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_bootstrap_ci_brackets_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y
    lo, hi = bootstrap_auc_ci(y, s, n_boot=300, seed=3)
    assert lo < auc(y, s) < hi and hi - lo < 0.2
    assert (lo, hi) == bootstrap_auc_ci(y, s, n_boot=300, seed=3)


def test_uncertain_band_and_labels():
    p = np.array([0.1, 0.45, 0.55, 0.9])
    assert uncertain_rate(p, (0.4, 0.6)) == 0.5
    assert label_predictions(p, 0.5, (0.4, 0.6)) == ["real", "uncertain", "uncertain", "fake"]


def test_temperature_recovers_true_value():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 4, 20000)  # model logits
    y = (rng.random(20000) < sigmoid(z / 2.5)).astype(int)  # true temperature 2.5
    assert fit_temperature(z, y) == pytest.approx(2.5, rel=0.05)


def test_threshold_maximises_balanced_accuracy():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    assert choose_threshold(y, p) == pytest.approx(0.35)


def test_perturbations_keep_shape_and_pickle():
    rgb = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    for levels in PERTURBATIONS.values():
        for _, _, fn in levels:
            out = fn(rgb)
            assert out.shape == rgb.shape and out.dtype == np.uint8
            pickle.loads(pickle.dumps(Perturb(fn)))  # Windows DataLoader workers need this


def test_compare_marks_missing_runs():
    metrics = {
        "in_dataset": {"ffpp": {"auc": 0.95, "auc_ci95": [0.94, 0.96], "ece_after": 0.02,
                                "uncertain_rate": 0.05}},
        "cross_dataset": {"celebdf": {"auc": 0.7}},
        "robustness": {"families": {"jpeg_quality": [{"level": "q50", "auc": 0.9}]}},
    }  # fmt: skip
    table = build_table("T", {"a": "A", "b": "B"}, {"a": metrics, "b": None}, baseline="a")
    assert "0.950 [0.940, 0.960] (ffpp)" in table and "0.700" in table and "0.900" in table
    row_b = next(line for line in table.splitlines() if "(`b`)" in line)
    assert row_b.count(NYM) == 6


# --------------------------------------------------------------- full evaluate


@pytest.fixture(scope="module")
def trained_run(debug_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("evalrun")
    Trainer(tiny_cfg(debug_data, out, epochs=1, freeze_epochs=0)).fit()
    return out / "tiny"


def _opts(run_dir, roots, out):
    return EvalOptions(run_dir=run_dir, data_roots=roots, out_dir=out, batch_size=16,
                       num_workers=0, device="cpu", threads=2, robustness_max=10, bootstrap=50,
                       predictions_dir=out / "preds")  # fmt: skip


def test_evaluate_writes_full_report(trained_run, debug_data, tmp_path):
    report = run_evaluate(_opts(trained_run, [debug_data], tmp_path))
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["run"] == "tiny"
    assert set(saved["in_dataset"]) == {"debug"} and set(saved["cross_dataset"]) == {"debugx"}
    m = saved["in_dataset"]["debug"]
    for key in (
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "auc_ci95",
        "ece_before",
        "ece_after",
        "uncertain_rate",
        "per_manipulation",
    ):
        assert key in m
    assert "grid" in m["per_manipulation"]
    assert set(saved["robustness"]["families"]) == set(PERTURBATIONS)
    assert saved["calibration"]["fitted_on"] == "val"
    assert "debug->debugx" in saved["generalization_gap_auc"]
    for png in ("roc.png", "robustness.png", "reliability.png", "confusion.png"):
        assert (tmp_path / png).exists()
    assert report["calibration"]["temperature"] > 0


def test_calibration_uses_validation_only(trained_run, debug_data, tmp_path, monkeypatch):
    """1) Calibration is fitted before any test row is predicted, from val rows only.
    2) Changing every test label leaves temperature and threshold unchanged."""
    calls = []
    original = Evaluator.predict

    def spy(self, rows, root, transform=None):
        calls.append({r["split"] for r in rows})
        return original(self, rows, root, transform)

    monkeypatch.setattr(Evaluator, "predict", spy)
    first = run_evaluate(_opts(trained_run, [debug_data], tmp_path / "a"))
    assert calls[0] == {"val"}
    assert all("val" not in c for c in calls[1:])  # val is never mixed with test

    flipped = tmp_path / "flipped"
    shutil.copytree(debug_data, flipped)
    path = flipped / "manifest.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["split"] == "test":
            r["label"] = "real" if r["label"] == "fake" else "fake"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    second = run_evaluate(_opts(trained_run, [flipped], tmp_path / "b"))
    for key in ("temperature", "threshold", "n_val"):
        assert first["calibration"][key] == second["calibration"][key]
    a, b = first["in_dataset"]["debug"]["auc"], second["in_dataset"]["debug"]["auc"]
    assert a == pytest.approx(1 - b)  # the test labels really did change


def test_fit_calibration_signature_takes_no_test_data():
    import inspect

    assert list(inspect.signature(fit_calibration).parameters) == [
        "val_logits", "val_labels", "band"
    ]  # fmt: skip
