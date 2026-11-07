"""Stage model files for a GitHub Release and pin their SHA-256 checksums.

PowerShell / bash:
    uv run python scripts/release_weights.py --export-dir artifacts/export/effb0_main --tag v1.0.0

- copies the serving files to artifacts/release/<tag>/ (gitignored)
- writes SHA256SUMS there
- writes the tag + checksums into src/deeptrace/weights.json (committed), which
  `deeptrace download-weights` verifies against
Model files are never committed to git; they are uploaded as release assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "deeptrace" / "weights.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def release_files(export_dir: Path) -> list[Path]:
    meta = json.loads((export_dir / "model.json").read_text(encoding="utf-8"))
    names = [meta["serving_model"], meta["gradcam_model"], "model.json"]
    files = [export_dir / n for n in names]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}; run `deeptrace export` first")
    return files


def stage(export_dir: Path, tag: str, out_root: Path, manifest_path: Path) -> dict:
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    for f in release_files(export_dir):
        dest = out / f.name
        shutil.copy2(f, dest)
        assets.append({"name": f.name, "sha256": sha256(dest), "size": dest.stat().st_size})
    (out / "SHA256SUMS").write_text(
        "".join(f"{a['sha256']}  {a['name']}\n" for a in assets), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(tag=tag, assets=assets)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "release")
    args = ap.parse_args()
    manifest = stage(args.export_dir, args.tag, args.out, MANIFEST)
    for a in manifest["assets"]:
        print(f"{a['sha256']}  {a['size'] / 1e6:8.1f} MB  {a['name']}")
    folder = (args.out / args.tag).as_posix()
    print(f"\nupdated {MANIFEST.relative_to(ROOT).as_posix()} (commit it)")
    print(f'release: gh release create {args.tag} {folder}/* --title "DeepTrace {args.tag}"')


if __name__ == "__main__":
    main()
