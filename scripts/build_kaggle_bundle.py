"""Build the folder you upload to Kaggle as the private dataset `deeptrace-bundle`.

    uv run python scripts/build_kaggle_bundle.py

It contains the deeptrace wheel, the configs/ folder and a COMMIT file (so every
Kaggle run records which code version produced it). Once the GitHub repo is
public, the notebook can install from git instead and this bundle is optional.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "kaggle_bundle"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(OUT)], cwd=ROOT, check=True)
    shutil.copytree(ROOT / "configs", OUT / "configs")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()  # fmt: skip
    (OUT / "COMMIT").write_text(commit + ("-dirty" if dirty else "") + "\n", encoding="utf-8")
    print(f"bundle ready: {OUT}")
    print("upload: kaggle datasets init -p artifacts/kaggle_bundle   (then set the id)")
    print("        kaggle datasets create -p artifacts/kaggle_bundle (private by default)")
    print("    or: kaggle.com -> Datasets -> New Dataset -> drag the folder in, keep it Private")


if __name__ == "__main__":
    main()
