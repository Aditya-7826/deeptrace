"""Training loop tests on CPU with a tiny timm model and synthetic crops."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from deeptrace.data.datasets import balanced_sampler, read_manifest, select_rows
from deeptrace.models.factory import (
    ModelConfig,
    create_model,
    head_modules,
    set_backbone_trainable,
    split_parameters,
)
from deeptrace.train.checkpoint import atomic_save
from deeptrace.train.config import TrainConfig
from deeptrace.train.loop import Trainer, WarmupCosine
from deeptrace.train.oom import is_cuda_oom, run_with_oom_fallback

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(data_root: Path, out_root: Path, **train) -> TrainConfig:
    return TrainConfig.from_dict({
        "run_name": "tiny",
        "output_root": str(out_root),
        "data": {"root": str(data_root), "num_workers": 0, "balance": "sampler"},
        "model": {"name": "test_efficientnet", "pretrained": False},
        "augment": True,
        "train": {"epochs": 2, "batch_size": 8, "freeze_epochs": 1, "warmup_epochs": 0.5,
                  "amp": False, "threads": 2, "device": "cpu", **train},
    })  # fmt: skip


def step_losses(run_dir: Path, epoch: int) -> list[str]:
    with (run_dir / "steps.csv").open(encoding="utf-8", newline="") as fh:
        return [r["loss"] for r in csv.DictReader(fh) if int(r["epoch"]) == epoch]


def test_forward_backward_cpu():
    model = create_model(ModelConfig(name="test_efficientnet", pretrained=False))
    x = torch.randn(4, 3, 224, 224)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(x).squeeze(1), torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    assert model(x).shape == (4, 1)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_head_backbone_split_and_freeze():
    model = create_model(ModelConfig(name="test_efficientnet", pretrained=False))
    backbone, head = split_parameters(model)
    assert len(head) == 2 and len(backbone) > 10  # classifier weight + bias
    model.train()
    set_backbone_trainable(model, False)
    assert not any(p.requires_grad for p in backbone) and all(p.requires_grad for p in head)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)


def test_resume_gives_identical_next_step_loss(debug_data, tmp_path):
    full = Trainer(tiny_cfg(debug_data, tmp_path / "a"))
    full.fit()
    cut = Trainer(tiny_cfg(debug_data, tmp_path / "b"))
    cut.fit(stop_after_epoch=0)  # simulated Kaggle session cut after epoch 0
    resumed = Trainer(tiny_cfg(debug_data, tmp_path / "b"), resume=True)
    assert resumed.state["epoch"] == 0
    resumed.fit()
    a, b = step_losses(tmp_path / "a" / "tiny", 1), step_losses(tmp_path / "b" / "tiny", 1)
    assert a and a[0] == b[0]  # identical next-step loss
    assert a == b  # and the whole resumed epoch matches
    assert (tmp_path / "b" / "tiny" / "best.pt").exists()
    for name in ("run_info.json", "config.yaml", "metrics.csv", "curves.png"):
        assert (tmp_path / "a" / "tiny" / name).exists()


def test_frozen_stage_leaves_backbone_unchanged(debug_data, tmp_path):
    trainer = Trainer(tiny_cfg(debug_data, tmp_path, epochs=1))
    backbone, head = split_parameters(trainer.model)
    before_b = [p.detach().clone() for p in backbone]
    before_h = [p.detach().clone() for p in head]
    bn = next(m for m in trainer.model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    stats = bn.running_mean.clone()
    trainer.train_one_epoch(0, 8)
    assert all(torch.equal(a, b) for a, b in zip(before_b, backbone, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(before_h, head, strict=True))
    assert torch.equal(stats, bn.running_mean)


def test_existing_run_needs_resume(debug_data, tmp_path):
    Trainer(tiny_cfg(debug_data, tmp_path, epochs=1)).fit()
    with pytest.raises(FileExistsError):
        Trainer(tiny_cfg(debug_data, tmp_path, epochs=1))
    done = Trainer(tiny_cfg(debug_data, tmp_path, epochs=1), resume=True)
    assert done.fit()["finished"] is True  # nothing left to do


def test_validation_has_no_augmentation(debug_data, tmp_path):
    trainer = Trainer(tiny_cfg(debug_data, tmp_path))
    assert trainer.val_ds.transform is None and trainer.train_ds.transform is not None


def test_oom_fallback_halves_once():
    calls, restored = [], []

    def fn(bs):
        calls.append(bs)
        if bs > 8:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return "ok"

    assert run_with_oom_fallback(fn, 16, restored.append) == ("ok", 8, True)
    assert calls == [16, 8] and restored == [8]
    with pytest.raises(torch.cuda.OutOfMemoryError):  # second OOM is not retried again
        run_with_oom_fallback(fn, 32, restored.append)
    with pytest.raises(torch.cuda.OutOfMemoryError):  # retry already used up
        run_with_oom_fallback(fn, 16, restored.append, allow_retry=False)
    with pytest.raises(ValueError):  # other errors propagate untouched
        run_with_oom_fallback(lambda bs: (_ for _ in ()).throw(ValueError("x")), 16, print)
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_atomic_save_leaves_no_temp(tmp_path):
    path = tmp_path / "ck" / "last.pt"
    atomic_save({"a": 1}, path)
    atomic_save({"a": 2}, path)
    assert torch.load(path)["a"] == 2
    assert [p.name for p in path.parent.iterdir()] == ["last.pt"]


def test_warmup_cosine_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = WarmupCosine(opt, [1.0], epochs=10, warmup_epochs=1, min_ratio=0.01)
    assert s.factor(0.5) == pytest.approx(0.5)
    assert s.factor(1.0) == pytest.approx(1.0)
    assert s.factor(10.0) == pytest.approx(0.01)
    assert s.factor(3.0) > s.factor(6.0)


def test_select_rows_and_balanced_sampler(debug_data):
    rows = read_manifest(debug_data)
    train = select_rows(rows, "train")
    sub = select_rows(rows, "train", max_items=10, seed=1)
    assert len(sub) == 10 and sub == select_rows(rows, "train", max_items=10, seed=1)
    labels = [0] * 20 + [1] * 80
    picks = list(balanced_sampler(labels, torch.Generator().manual_seed(0)))
    reals = sum(labels[i] == 0 for i in picks)
    assert 35 <= reals <= 65 and len(train) > 0  # ~50% real despite 20% in data


def test_two_branch_rgb_fft():
    model = create_model(ModelConfig(name="test_efficientnet", arch="rgb_fft", pretrained=False))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)
    backbone, head = split_parameters(model)
    assert any(p is model.head.weight for p in head)
    assert all(p is not model.head.weight for p in backbone)


@pytest.mark.parametrize(("name", "arch"), [("efficientnet_b0", "timm"), ("resnet18", "timm"),
                                            ("efficientnet_b0", "rgb_fft")])  # fmt: skip
def test_logit_head_starts_near_zero(name, arch):
    model = create_model(ModelConfig(name=name, arch=arch, pretrained=False))
    heads = [m for h in head_modules(model) for m in h.modules()
             if isinstance(m, torch.nn.Linear) and m.out_features == 1]  # fmt: skip
    assert heads
    for m in heads:
        assert m.weight.std().item() < 0.02 and torch.all(m.bias == 0)
