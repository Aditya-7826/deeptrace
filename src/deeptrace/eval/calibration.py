"""Temperature scaling and threshold selection, fitted on VALIDATION data only."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from deeptrace.eval.metrics import sigmoid


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    y = np.asarray(labels, dtype=np.float64)
    # numerically stable binary cross-entropy with logits
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0, iters: int = 80
) -> float:
    """Minimise validation NLL over T with a golden-section search on log T."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, labels, math.exp(c)), nll(logits, labels, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, labels, math.exp(d))
    return float(math.exp((a + b) / 2))


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on validation probabilities."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return 0.5
    candidates = np.unique(probs)
    order = np.argsort(probs)
    p_sorted, y_sorted = probs[order], labels[order]
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    # predicted fake = p >= t. For each candidate t, count positives/negatives below t.
    pos_below = np.searchsorted(p_sorted, candidates, side="left")
    cum_pos = np.r_[0, np.cumsum(y_sorted)]
    fn = cum_pos[pos_below]
    tn = pos_below - fn
    tpr = (n_pos - fn) / n_pos
    tnr = tn / n_neg
    best = int(np.argmax(tpr + tnr))
    return float(candidates[best])


@dataclass(frozen=True)
class Calibration:
    temperature: float
    threshold: float
    uncertain_band: tuple[float, float]
    fitted_on: str = "val"
    n_val: int = 0

    def probs(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_calibration(
    val_logits: np.ndarray, val_labels: np.ndarray, band: tuple[float, float] = (0.4, 0.6)
) -> Calibration:
    """The only place temperature and threshold are chosen. Takes validation arrays only."""
    t = fit_temperature(val_logits, val_labels)
    threshold = choose_threshold(val_labels, sigmoid(np.asarray(val_logits) / t))
    return Calibration(t, threshold, (float(band[0]), float(band[1])), "val", len(val_labels))
