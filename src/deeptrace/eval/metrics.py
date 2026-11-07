"""Metric functions. Label 1 = fake, scores/probabilities are P(fake)."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks for ties
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def confusion(labels: np.ndarray, preds: np.ndarray) -> list[list[int]]:
    """[[TN, FP], [FN, TP]] with rows = true class (real, fake)."""
    labels, preds = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    return [
        [int(((labels == 0) & (preds == 0)).sum()), int(((labels == 0) & (preds == 1)).sum())],
        [int(((labels == 1) & (preds == 0)).sum()), int(((labels == 1) & (preds == 1)).sum())],
    ]


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(int)
    (tn, fp), (fn, tp) = confusion(labels, preds)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auc": auc(labels, probs),
        "accuracy": float((preds == labels).mean()) if len(labels) else float("nan"),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def bootstrap_auc_ci(
    labels: np.ndarray, scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Stratified percentile bootstrap CI for AUC (resamples reals and fakes separately)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    y = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    for b in range(n_boot):
        s = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        stats[b] = auc(y, s)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def reliability_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> list[dict]:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": count,
            "confidence": float(probs[mask].mean()) if count else None,
            "accuracy": float(labels[mask].mean()) if count else None,
        })  # fmt: skip
    return bins


def ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of P(fake) (binary reliability, equal-width bins)."""
    n = len(labels)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in reliability_bins(labels, probs, n_bins):
        if b["count"]:
            total += b["count"] / n * abs(b["confidence"] - b["accuracy"])
    return float(total)


def uncertain_rate(probs: np.ndarray, band: tuple[float, float]) -> float:
    probs = np.asarray(probs)
    if len(probs) == 0:
        return float("nan")
    return float(((probs >= band[0]) & (probs <= band[1])).mean())


def label_predictions(probs: np.ndarray, threshold: float, band: tuple[float, float]) -> list[str]:
    out = []
    for p in np.asarray(probs):
        if band[0] <= p <= band[1]:
            out.append("uncertain")
        else:
            out.append("fake" if p >= threshold else "real")
    return out
