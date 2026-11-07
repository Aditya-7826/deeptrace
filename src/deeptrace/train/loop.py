"""`deeptrace train`: two-stage fine-tuning with checkpoint/resume and OOM fallback."""

from __future__ import annotations

import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from deeptrace.data.augment import build_train_augment
from deeptrace.data.datasets import (
    CropDataset,
    balanced_sampler,
    class_counts,
    pos_weight,
    read_manifest,
    select_rows,
)
from deeptrace.models.factory import (
    create_model,
    normalization,
    set_backbone_trainable,
    split_parameters,
)
from deeptrace.train.checkpoint import atomic_save, load, rng_state, set_rng_state, unwrap
from deeptrace.train.config import TrainConfig
from deeptrace.train.logger import RunLogger
from deeptrace.train.oom import run_with_oom_fallback
from deeptrace.train.repro import git_commit, run_info, seed_everything
from deeptrace.utils import resolve_device, set_cpu_threads

LOG = logging.getLogger("deeptrace.train")


class WarmupCosine:
    """LR = base * factor(progress), progress measured in (fractional) epochs.

    Using epoch progress rather than a step counter keeps the schedule correct
    even if the batch size (and so steps per epoch) changes after an OOM."""

    def __init__(self, optimizer, base_lrs, epochs, warmup_epochs, min_ratio) -> None:
        self.optimizer, self.base_lrs = optimizer, list(base_lrs)
        self.epochs, self.warmup, self.min_ratio = epochs, warmup_epochs, min_ratio
        self.progress = 0.0

    def factor(self, p: float) -> float:
        if self.warmup > 0 and p < self.warmup:
            return max(p / self.warmup, 1e-3)
        span = max(self.epochs - self.warmup, 1e-8)
        t = min(max((p - self.warmup) / span, 0.0), 1.0)
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    def set(self, progress: float) -> None:
        self.progress = progress
        f = self.factor(progress)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * f

    def state_dict(self) -> dict[str, Any]:
        return {"progress": self.progress, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = state["base_lrs"]
        self.set(state["progress"])


def smooth_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    return y * (1 - eps) + 0.5 * eps


@torch.no_grad()
def predict_logits(
    model: nn.Module, dataset, batch_size: int, device: str, amp: bool, num_workers: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    logits, labels = [], []
    dev = torch.device(device)
    for x, y in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = model(x.to(dev, non_blocking=True)).squeeze(1)
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: bool = False) -> None:
        self.cfg = cfg
        t = cfg.train
        set_cpu_threads(t.threads)
        self.device = resolve_device(t.device)
        self.use_amp = t.amp and self.device.startswith("cuda")
        self.run_dir = Path(cfg.output_root) / cfg.run_name
        self.last_path, self.best_path = self.run_dir / "last.pt", self.run_dir / "best.pt"
        if not resume and self.last_path.exists():
            raise FileExistsError(
                f"{self.last_path} exists: pass --resume to continue, or change run_name"
            )
        if resume and not self.last_path.exists():
            raise FileNotFoundError(f"--resume given but {self.last_path} does not exist")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(t.seed)

        root = Path(cfg.data.root)
        rows = read_manifest(root)
        d = cfg.data
        train_rows = select_rows(rows, "train", d.datasets, d.max_train, t.seed)
        val_rows = select_rows(rows, "val", d.datasets, d.max_val, t.seed)
        if not train_rows or not val_rows:
            raise ValueError(f"empty split in {root}: train={len(train_rows)} val={len(val_rows)}")

        self.model = create_model(cfg.model)
        self.mean, self.std = normalization(self.model)
        self.train_ds = CropDataset(train_rows, root, build_train_augment(cfg.augment),
                                    self.mean, self.std)  # fmt: skip
        self.val_ds = CropDataset(val_rows, root, None, self.mean, self.std)
        LOG.info("train %s, val %s (real=0, fake=1)", class_counts(self.train_ds.labels),
                 class_counts(self.val_ds.labels))  # fmt: skip

        self.model.to(self.device)
        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            LOG.info("using DataParallel over %d GPUs", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        backbone, head = split_parameters(unwrap(self.model))
        base_lrs = [t.lr * t.backbone_lr_mult, t.lr]
        self.optimizer = torch.optim.AdamW(
            [{"params": backbone, "lr": base_lrs[0]}, {"params": head, "lr": base_lrs[1]}],
            weight_decay=t.weight_decay,
        )
        self.sched = WarmupCosine(self.optimizer, base_lrs, t.epochs, t.warmup_epochs,
                                  t.min_lr_ratio)  # fmt: skip
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.pos_weight = (
            torch.tensor(pos_weight(self.train_ds.labels), device=self.device)
            if d.balance == "loss"
            else None
        )
        self.state: dict[str, Any] = {
            "epoch": -1, "best_auc": -math.inf, "best_epoch": -1, "bad_epochs": 0,
            "batch_size": t.batch_size, "oom_retried": False, "finished": False,
            "stopped_early": False,
        }  # fmt: skip
        self.logger = RunLogger(self.run_dir, cfg.run_name, cfg.to_dict())
        if resume:
            self._load_last()
        else:
            info = run_info(cfg.to_dict(), self.device)
            (self.run_dir / "run_info.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8"
            )
            (self.run_dir / "config.yaml").write_text(
                yaml.safe_dump(cfg.to_dict()), encoding="utf-8"
            )

    # ----------------------------------------------------------------- state
    def _load_last(self) -> None:
        ck = load(self.last_path, map_location=self.device)
        unwrap(self.model).load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state.update(ck["state"])
        set_rng_state(ck["rng"])
        self.logger.truncate_after(self.state["epoch"])
        LOG.info("resumed from %s after epoch %d (best AUC %.4f)", self.last_path,
                 self.state["epoch"], self.state["best_auc"])  # fmt: skip

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "model": unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": dict(self.state),
            "rng": rng_state(),
            "config": self.cfg.to_dict(),
        }

    def _snapshot(self) -> bytes | None:
        """In-memory copy of the epoch-start state, used to restart an epoch after OOM."""
        if not self.device.startswith("cuda"):
            return None
        buf = io.BytesIO()
        torch.save(self._checkpoint(), buf)
        return buf.getvalue()

    def _restore(self, snap: bytes | None, new_bs: int) -> None:
        self.state["batch_size"] = new_bs
        if snap is None:
            return
        ck = torch.load(io.BytesIO(snap), map_location=self.device, weights_only=False)
        unwrap(self.model).load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])

    # ----------------------------------------------------------------- loops
    def _loader(self, batch_size: int, epoch: int) -> DataLoader:
        seed = self.cfg.train.seed * 1000 + epoch
        self.train_ds.epoch_seed = seed
        gen = torch.Generator().manual_seed(seed)
        sampler = None
        if self.cfg.data.balance == "sampler":
            sampler = balanced_sampler(
                self.train_ds.labels, torch.Generator().manual_seed(seed + 1)
            )
        return DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            generator=gen,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.device.startswith("cuda"),
            drop_last=len(self.train_ds) > batch_size,
            persistent_workers=False,
        )

    def train_one_epoch(self, epoch: int, batch_size: int) -> tuple[float, list[dict[str, Any]]]:
        t = self.cfg.train
        frozen = epoch < t.freeze_epochs
        self.model.train()
        set_backbone_trainable(unwrap(self.model), not frozen)
        loader = self._loader(batch_size, epoch)
        dev = torch.device(self.device)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        steps, losses = len(loader), []
        for i, (x, y) in enumerate(loader):
            self.sched.set(epoch + i / steps)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), smooth_targets(y, t.label_smoothing), pos_weight=self.pos_weight
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if t.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.detach()))
        self.sched.set(epoch + 1.0)
        rows = [{"epoch": epoch, "step": i, "loss": f"{v:.6f}"} for i, v in enumerate(losses)]
        return float(np.mean(losses)) if losses else float("nan"), rows

    def validate(self) -> dict[str, float]:
        bs = max(self.state["batch_size"], 1)
        logits, labels = predict_logits(self.model, self.val_ds, bs, self.device, self.use_amp,
                                        self.cfg.data.num_workers)  # fmt: skip
        loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits),
                                                  torch.from_numpy(labels)).item()  # fmt: skip
        acc = float(((logits > 0).astype(np.float32) == labels).mean())
        return {"val_loss": loss, "val_auc": safe_auc(labels, logits), "val_acc": acc}

    def _save_best(self, epoch: int, auc: float) -> None:
        atomic_save(
            {
                "model": unwrap(self.model).state_dict(),
                "model_config": self.cfg.to_dict()["model"],
                "mean": list(self.mean),
                "std": list(self.std),
                "epoch": epoch,
                "val_auc": auc,
                "run_name": self.cfg.run_name,
                "git_commit": git_commit(),
            },
            self.best_path,
        )

    def fit(self, stop_after_epoch: int | None = None) -> dict[str, Any]:
        t = self.cfg.train
        if self.state["finished"]:
            LOG.info("run already finished; nothing to do")
            return self.summary()
        start = time.time()
        epoch_times: list[float] = []
        for epoch in range(self.state["epoch"] + 1, t.epochs):
            t0 = time.time()
            snap = self._snapshot()
            (train_loss, step_rows), bs, retried = run_with_oom_fallback(
                lambda b, e=epoch: self.train_one_epoch(e, b),
                self.state["batch_size"],
                lambda b, s=snap: self._restore(s, b),
                allow_retry=not self.state["oom_retried"],
            )
            if retried:
                self.state["oom_retried"] = True
            self.state["batch_size"] = bs
            metrics = self.validate()
            auc = metrics["val_auc"]
            improved = not math.isnan(auc) and auc > self.state["best_auc"] + 1e-4
            if improved or self.state["best_epoch"] < 0:
                self.state.update(best_auc=auc if not math.isnan(auc) else -math.inf,
                                  best_epoch=epoch, bad_epochs=0)  # fmt: skip
                self._save_best(epoch, auc)
            else:
                self.state["bad_epochs"] += 1
            self.state["epoch"] = epoch
            stop = self.state["bad_epochs"] >= t.early_stopping_patience
            self.state["stopped_early"] = stop
            self.state["finished"] = stop or epoch == t.epochs - 1
            seconds = time.time() - t0
            epoch_times.append(seconds)
            lrs = [g["lr"] for g in self.optimizer.param_groups]
            self.logger.log_steps(step_rows)
            self.logger.log_epoch({
                "epoch": epoch, "stage": "head" if epoch < t.freeze_epochs else "full",
                "train_loss": f"{train_loss:.6f}", "val_loss": f"{metrics['val_loss']:.6f}",
                "val_auc": f"{auc:.6f}", "val_acc": f"{metrics['val_acc']:.6f}",
                "lr_backbone": f"{lrs[0]:.3e}", "lr_head": f"{lrs[1]:.3e}",
                "batch_size": bs, "epoch_seconds": f"{seconds:.1f}",
            })  # fmt: skip
            atomic_save(self._checkpoint(), self.last_path)
            LOG.info("epoch %d: train_loss %.4f val_loss %.4f val_auc %.4f (%.0fs)%s", epoch,
                     train_loss, metrics["val_loss"], auc, seconds,
                     " *best*" if improved else "")  # fmt: skip
            if stop:
                LOG.info("early stopping: no val AUC gain for %d epochs", t.early_stopping_patience)
                break
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
            if t.time_budget_hours:
                elapsed = time.time() - start
                if elapsed + max(epoch_times) > t.time_budget_hours * 3600:
                    LOG.warning("time budget reached after epoch %d: stopping cleanly; "
                                "re-run with --resume to continue", epoch)  # fmt: skip
                    break
        summary = self.summary()
        (self.run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                         encoding="utf-8")  # fmt: skip
        self.logger.close()
        return summary

    def summary(self) -> dict[str, Any]:
        s = self.state
        return {
            "run_name": self.cfg.run_name,
            "epochs_completed": s["epoch"] + 1,
            "best_val_auc": s["best_auc"],
            "best_epoch": s["best_epoch"],
            "finished": s["finished"],
            "stopped_early": s["stopped_early"],
            "final_batch_size": s["batch_size"],
            "oom_fallback_used": s["oom_retried"],
        }
