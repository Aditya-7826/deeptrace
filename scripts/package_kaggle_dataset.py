"""Zip prepared crops + manifest for upload as a PRIVATE Kaggle dataset.

PowerShell / bash:
    uv run python scripts/package_kaggle_dataset.py --crops-dir data/crops/kaggle140k
        --slug deeptrace-crops-kaggle140k --username <your-kaggle-username>
    kaggle datasets create -p artifacts/kaggle_upload/deeptrace-crops-kaggle140k

`kaggle datasets create` makes a private dataset unless you pass --public.
Never pass --public: FF++ and Celeb-DF crops must not be redistributed.
Only crops listed in manifest.csv are packaged (dropped near-duplicates are left out).
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

INCLUDE_FILES = ("manifest.csv", "prepare_report.json", "prepare_journal.csv")


def package(crops_dir: Path, out_root: Path, slug: str, username: str) -> Path:
    manifest = crops_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found: run `deeptrace prepare` first")
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{slug}.zip"
    with manifest.open(encoding="utf-8", newline="") as fh:
        crop_paths = [row["path"] for row in csv.DictReader(fh)]
    # ZIP_STORED: JPEGs don't compress further
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in INCLUDE_FILES:
            if (crops_dir / name).exists():
                zf.write(crops_dir / name, name)
        for rel in crop_paths:
            zf.write(crops_dir / rel, rel)
    metadata = {
        "title": slug,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
        "subtitle": "Private DeepTrace face crops. Not for redistribution.",
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--crops-dir", type=Path, required=True)
    ap.add_argument("--slug", required=True, help="dataset slug, e.g. deeptrace-crops-kaggle140k")
    ap.add_argument("--username", required=True, help="your Kaggle username")
    ap.add_argument("--out", type=Path, default=Path("artifacts/kaggle_upload"))
    args = ap.parse_args()
    zip_path = package(args.crops_dir, args.out, args.slug, args.username)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"wrote {zip_path} ({size_mb:.1f} MB)")
    print(f"upload as PRIVATE:  kaggle datasets create -p {zip_path.parent}")


if __name__ == "__main__":
    main()
