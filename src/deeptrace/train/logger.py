"""Local CSV + matplotlib logging; Weights & Biases only if WANDB_API_KEY is set."""

from __future__ import annotations

import csv
import logging
import math
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger("deeptrace.train")


class RunLogger:
    def __init__(self, run_dir: Path, run_name: str, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.epochs_csv = run_dir / "metrics.csv"
        self.steps_csv = run_dir / "steps.csv"
        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # optional, not a project dependency

                self._wandb = wandb.init(
                    project="deeptrace", name=run_name, config=config, resume="allow", id=run_name
                )
            except ImportError:
                LOG.warning("WANDB_API_KEY is set but wandb is not installed; CSV logging only")

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def log_steps(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._append(self.steps_csv, row)

    def log_epoch(self, row: dict[str, Any]) -> None:
        self._append(self.epochs_csv, row)
        if self._wandb is not None:
            self._wandb.log(row)
        self.plot()

    def truncate_after(self, epoch: int) -> None:
        """On resume, drop rows logged after the checkpointed epoch (a cut mid-epoch)."""
        for path in (self.epochs_csv, self.steps_csv):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            keep = [r for r in rows if int(r["epoch"]) <= epoch]
            if len(keep) != len(rows):
                with path.open("w", encoding="utf-8", newline="") as fh:
                    if rows:
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(keep)

    def plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with self.epochs_csv.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return
        epochs = [int(r["epoch"]) for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", label="train")
        ax1.plot(epochs, [float(r["val_loss"]) for r in rows], marker="o", label="val")
        ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
        ax1.legend()
        aucs = [float(r["val_auc"]) for r in rows]
        ax2.plot(epochs, [a if not math.isnan(a) else None for a in aucs], marker="o")
        ax2.set(xlabel="epoch", ylabel="AUC", title="Validation AUC")
        fig.tight_layout()
        fig.savefig(self.run_dir / "curves.png", dpi=110)
        plt.close(fig)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
