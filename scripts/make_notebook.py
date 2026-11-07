# ruff: noqa: E501  (long lines are notebook markdown)
"""Generate notebooks/train_kaggle.ipynb (the notebook's source of truth).

uv run python scripts/make_notebook.py notebooks/train_kaggle.ipynb
uv run ruff format notebooks
"""

import json
import sys
from pathlib import Path


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("""
# DeepTrace: training on Kaggle

Before you run this notebook:
1. **Settings → Accelerator → GPU T4 x2** (preferred; see the P100 note in cell 3).
2. **Settings → Internet → On** (pretrained ImageNet weights are downloaded from the Hugging Face Hub).
3. **Add Input:** your private `deeptrace-bundle` dataset, plus either the public *140k Real and Fake Faces* dataset (MODE = "prepare") or your private crops dataset (MODE = "train").
4. Edit cell 1, then **Save Version → Save & Run All (Commit)**, so it runs in the background and the outputs are saved.

Everything this notebook writes goes to `/kaggle/working/outputs/` (training) or `/kaggle/working/crops/` (prepare).
See `docs/kaggle_guide.md` for the full step-by-step guide.
"""),
    md("## 1. Settings: edit this cell"),
    code("""
MODE = "train"   # "prepare": crops once (Kaggle 140k) | "train" | "evaluate": metrics + Grad-CAM
CONFIG = "configs/kaggle140k_effb0.yaml"   # see the run table in docs/kaggle_guide.md

BUNDLE = "/kaggle/input/deeptrace-bundle"            # private dataset: wheel + configs + COMMIT
REPO = "https://github.com/Aditya-7826/deeptrace.git"  # used only if BUNDLE is not attached
CROPS = "/kaggle/input/deeptrace-crops-kaggle140k"   # private crops dataset (MODE = "train")
RAW_ROOT = "/kaggle/input"  # parent of 140k-real-and-fake-faces (MODE = "prepare")

# Resuming after a session cut: attach the previous version's output as an input
# (Add Input -> Your Work -> Notebook Output), set RESUME = True and its path here.
RESUME = False
PREVIOUS_OUTPUT = "/kaggle/input/deeptrace-train-kaggle/outputs"

# MODE = "evaluate": attach the finished training version's output and point at it.
TRAINED_OUTPUT = "/kaggle/input/deeptrace-train-kaggle/outputs"
EXTRA_TEST_CROPS = []  # more private crops datasets used as cross-dataset tests

OUT = "/kaggle/working/outputs"
TIME_BUDGET_HOURS = 11.0  # stop cleanly before Kaggle's 12 h session limit
"""),
    md("## 2. Install DeepTrace (keeps Kaggle's own PyTorch)"),
    code("""
import glob
import os
import shutil
import subprocess
import sys


def sh(*cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


# Kaggle mounts inputs at /kaggle/input/<name> or deeper
# (e.g. /kaggle/input/datasets/<owner>/<name>). If a path from cell 1 doesn't exist,
# look for a folder with the same name up to 3 levels down.
INPUT_ROOT = "/kaggle/input"


def find_input(path, what, required=True):
    if os.path.exists(path):
        return path
    name = os.path.basename(path.rstrip("/"))
    for depth in range(4):
        hits = sorted(glob.glob(INPUT_ROOT + "/" + "*/" * depth + name))
        if hits:
            print(f"{what}: {path} not found, using {hits[0]}")
            return hits[0]
    if required:
        raise FileNotFoundError(
            f"{what} not found: attach it with Add Input "
            f"(no folder named {name!r} under {INPUT_ROOT})"
        )
    return None


def find_run_output(path, run_name, filename, what):
    # Notebook outputs are mounted under the notebook's own slug: search for <run_name>/<filename>.
    if os.path.exists(os.path.join(path, run_name, filename)):
        return path
    for depth in range(5):
        hits = sorted(glob.glob(INPUT_ROOT + "/" + "*/" * depth + f"{run_name}/{filename}"))
        if hits:
            found = os.path.dirname(os.path.dirname(hits[0]))
            print(f"{what}: {path} not found, using {found}")
            return found
    raise FileNotFoundError(
        f"{what}: no {run_name}/{filename} under {INPUT_ROOT}; attach the earlier version's "
        "output (Add Input -> Your Work -> Notebook Output)"
    )


WORK = "/kaggle/working/deeptrace_src"
bundle = find_input(BUNDLE, "code bundle", required=False)
wheels = glob.glob(f"{bundle}/**/*.whl", recursive=True) if bundle else []
if wheels:
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", wheels[0])
    bundle_root = os.path.dirname(wheels[0])
    shutil.copytree(os.path.join(bundle_root, "configs"), f"{WORK}/configs", dirs_exist_ok=True)
    commit_file = os.path.join(bundle_root, "COMMIT")
    os.environ["DEEPTRACE_GIT_COMMIT"] = (
        open(commit_file).read().strip() if os.path.exists(commit_file) else "unknown")
else:
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    sh("git", "clone", "--depth", "1", REPO, WORK)
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", WORK)
    os.environ["DEEPTRACE_GIT_COMMIT"] = subprocess.run(
        ["git", "-C", WORK, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

# Only packages Kaggle may lack. torch/torchvision/numpy stay as Kaggle installed them.
sh(sys.executable, "-m", "pip", "install", "-q", "timm>=1.0", "albumentations>=1.4", "pyyaml")
if MODE == "prepare":
    # facenet-pytorch pins old torch; --no-deps keeps Kaggle's torch (MTCNN only needs torch + PIL)
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch>=2.6")
os.chdir(WORK)
print("code version:", os.environ["DEEPTRACE_GIT_COMMIT"])
"""),
    md("## 3. Detect the GPU and pick a batch size"),
    code("""
import torch
import yaml

n_gpu = torch.cuda.device_count()
assert n_gpu > 0, "No GPU found: Settings -> Accelerator -> GPU T4 x2"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"{n_gpu} x {name}, compute capability {cap}")
if cap < (7, 0):
    print("NOTE: P100 (sm_60). Recent PyTorch builds may drop sm_60; if you see "
          "'no kernel image is available', switch the accelerator to GPU T4 x2.")

cfg = yaml.safe_load(open(CONFIG))
model_name = cfg["model"]["name"] + ("+fft" if cfg["model"].get("arch") == "rgb_fft" else "")
# Per-GPU batch sizes for 16 GB cards with mixed precision at 224 px.
PER_GPU = {"efficientnet_b0": 96, "resnet50": 64, "vit_small_patch16_224": 96,
           "efficientnet_b0+fft": 64}
per_gpu = PER_GPU.get(model_name, 48)
if "P100" in name:
    per_gpu = int(per_gpu * 0.75)  # no tensor cores: leave more memory headroom
BATCH = per_gpu * n_gpu            # DataParallel splits each batch across GPUs
print("model:", model_name, "| batch size:", BATCH, "(halved automatically once on CUDA OOM)")
"""),
    md(
        '## 4. (MODE = "prepare") Make face crops once\n'
        "Run this once. Then: **Output → New Dataset**, keep it **Private**, "
        "and name it `deeptrace-crops-kaggle140k`."
    ),
    code("""
if MODE == "prepare":
    raw = find_input(os.path.join(RAW_ROOT, "140k-real-and-fake-faces"), "140k Real and Fake Faces dataset")
    sh(sys.executable, "-m", "deeptrace", "prepare", "--config", "configs/prepare_kaggle140k.yaml",
       "--raw-root", os.path.dirname(raw), "--output-dir", "/kaggle/working/crops", "--device", "cuda")
    print(open("/kaggle/working/crops/prepare_report.json").read()[:3000])
else:
    print("MODE is", MODE, "- skipping prepare")
"""),
    md("## 5. (Resume only) Copy the previous run's checkpoints back"),
    code("""
run_name = cfg["run_name"]
if MODE == "train" and RESUME:
    src = os.path.join(find_run_output(PREVIOUS_OUTPUT, run_name, "last.pt", "previous output"), run_name)
    shutil.copytree(src, os.path.join(OUT, run_name), dirs_exist_ok=True)
    print("restored", src, "->", os.path.join(OUT, run_name))
else:
    print("fresh run" if MODE == "train" else "not training")
"""),
    md("## 6. Train"),
    code("""
if MODE == "train":
    manifests = glob.glob(f"{find_input(CROPS, 'crops dataset')}/**/manifest.csv", recursive=True)
    assert manifests, f"manifest.csv not found under {CROPS}: attach your private crops dataset"
    data_root = os.path.dirname(manifests[0])
    cmd = [sys.executable, "-m", "deeptrace", "train", "--config", CONFIG,
           "--data-root", data_root, "--output-root", OUT,
           "--batch-size", str(BATCH), "--num-workers", "4",
           "--device", "cuda", "--time-budget-hours", str(TIME_BUDGET_HOURS)]
    if RESUME:
        cmd.append("--resume")
    sh(*cmd)
"""),
    md("## 7. Results of this session"),
    code("""
if MODE == "train":
    import pandas as pd
    from IPython.display import Image, display
    run_dir = os.path.join(OUT, run_name)
    display(pd.read_csv(os.path.join(run_dir, "metrics.csv")))
    print(open(os.path.join(run_dir, "train_summary.json")).read())
    display(Image(os.path.join(run_dir, "curves.png")))
    # last.pt holds optimizer state for resuming; best.pt is what evaluation/export use.
    for f in sorted(os.listdir(run_dir)):
        print(f"{f:28s} {os.path.getsize(os.path.join(run_dir, f)) / 1e6:8.1f} MB")
"""),
    md(
        '## 8. (MODE = "evaluate") Metrics, calibration, robustness and Grad-CAM checks\n'
        "Calibration (temperature + threshold) is fitted on the validation split only. "
        "Bring back `outputs/results/<run_name>/` into the repo's `results/<run_name>/`."
    ),
    code("""
if MODE == "evaluate":
    def manifest_root(folder):
        found = glob.glob(f"{find_input(folder, 'crops dataset')}/**/manifest.csv", recursive=True)
        assert found, f"manifest.csv not found under {folder}"
        return os.path.dirname(found[0])

    data_root = manifest_root(CROPS)
    run_src = os.path.join(find_run_output(TRAINED_OUTPUT, run_name, "best.pt", "trained output"), run_name)
    run_dir = os.path.join("/kaggle/working/runs", run_name)
    shutil.copytree(run_src, run_dir, dirs_exist_ok=True)
    results = os.path.join(OUT, "results")
    cmd = [sys.executable, "-m", "deeptrace", "evaluate", "--run", run_dir,
           "--data-root", data_root, "--results", results, "--device", "cuda",
           "--batch-size", "128", "--num-workers", "4", "--robustness-max", "0"]
    for extra in EXTRA_TEST_CROPS:
        cmd += ["--extra-data-root", manifest_root(extra)]
    sh(*cmd)
    sh(sys.executable, "-m", "deeptrace", "explain", "--run", run_dir, "--data-root", data_root,
       "--results", results, "--n", "200", "--device", "cuda",
       "--gallery", os.path.join(results, run_name, "gradcam_gallery.png"))
    from IPython.display import Image, display
    for png in ("roc.png", "reliability.png", "robustness.png", "gradcam_gallery.png"):
        display(Image(os.path.join(results, run_name, png)))
"""),
    md("""
## 9. Next
- If `train_summary.json` says `"finished": false`, the time budget ran out: re-run with `RESUME = True` (cell 1).
- Otherwise download **Output → outputs/<run_name>/** and put it in `artifacts/runs/<run_name>/` on your laptop.
- After `MODE = "evaluate"`: download **outputs/results/<run_name>/** into `results/<run_name>/` in the repo.
"""),
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("ok")
