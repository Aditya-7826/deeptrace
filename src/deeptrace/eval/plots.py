"""Evaluation plots (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from deeptrace.eval.metrics import reliability_bins  # noqa: E402


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (labels, scores, auc_value) in curves.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(
    labels: np.ndarray, before: np.ndarray, after: np.ndarray, eces: tuple[float, float], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, probs, title, e in zip(axes, (before, after), ("Before", "After temperature scaling"),
                                   eces, strict=True):  # fmt: skip
        bins = [b for b in reliability_bins(labels, probs) if b["count"]]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        ax.bar(centers, [b["accuracy"] for b in bins], width=1 / 15, edgecolor="k", alpha=0.7,
               label="fraction fake")  # fmt: skip
        sizes = [20 + 180 * b["count"] / max(len(labels), 1) for b in bins]
        ax.scatter([b["confidence"] for b in bins], [b["accuracy"] for b in bins], s=sizes,
                   color="C3", zorder=3, label="bin (size = count)")  # fmt: skip
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="predicted P(fake)", title=f"{title} (ECE {e:.3f})", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("observed fraction fake")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_robustness(robust: dict, clean_auc: float, path: Path) -> None:
    families = list(robust)
    fig, axes = plt.subplots(1, len(families), figsize=(3.4 * len(families), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families, strict=True):
        levels = robust[fam]
        xs = list(range(len(levels)))
        ax.plot(xs, [lv["auc"] for lv in levels], "o-", label="AUC")
        ax.plot(xs, [lv["balanced_accuracy"] for lv in levels], "s--", label="balanced acc.")
        ax.axhline(clean_auc, color="gray", lw=1.0, ls=":", label="clean AUC")
        ax.axhline(0.5, color="k", lw=0.6, ls="-", alpha=0.3)
        ax.set_xticks(xs, [lv["level"] for lv in levels], rotation=30, fontsize=8)
        ax.set(title=fam.replace("_", " "), ylim=(0.0, 1.02))
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], title: str, path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > arr.max() / 2 else "black")  # fmt: skip
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
