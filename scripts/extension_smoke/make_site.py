"""Build the test web page for the extension smoke test (scripts/extension_smoke/smoke.mjs).

    uv run python scripts/extension_smoke/make_site.py <site_dir>          # full page
    uv run python scripts/extension_smoke/make_site.py <site_dir> --many   # fresh noisy copies

page.html covers every filter case: duplicates, too small, 1x1 pixel, SVG, a corrupt JPEG,
a missing file, a data: URL, a blob: URL and a lazy-loaded image far below the fold.
many.html holds 16 never-seen images (random noise, so the API cache cannot answer them
instantly) for the service-worker restart check.
"""

from __future__ import annotations

import argparse
import base64
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "images"


def many(site: Path, n: int = 16) -> None:
    rng = np.random.default_rng()
    base = np.asarray(Image.open(FIXTURES / "real" / "aldrin.jpg").convert("RGB"), np.int16)
    for i in range(n):
        noisy = np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(site / f"n{i}.jpg", quality=92)
    imgs = "".join(f'<img src="n{i}.jpg" style="width:120px">' for i in range(n))
    (site / "many.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>many</title></head>'
        f"<body>{imgs}</body></html>",
        encoding="utf-8",
    )


def page(site: Path) -> None:
    for name in ("real/aldrin.jpg", "real/multiface.jpg", "real/noface.jpg",
                 "real/armstrong.jpg", "real/hadfield.jpg", "fake/whitson.jpg",
                 "real/corrupt.jpg"):  # fmt: skip
        shutil.copy(FIXTURES / name, site / Path(name).name)
    Image.new("RGB", (32, 32), "red").save(site / "small.jpg")
    Image.new("RGB", (1, 1), "white").save(site / "pixel.gif")
    (site / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
        '<circle cx="100" cy="100" r="90" fill="teal"/></svg>',
        encoding="utf-8",
    )
    buf = io.BytesIO()
    Image.open(FIXTURES / "fake" / "ochoa.jpg").convert("RGB").save(buf, "JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DeepTrace smoke</title>
<style>img{{width:260px;margin:6px;vertical-align:top}} .s{{width:32px}} .p{{width:1px}}</style>
</head><body><h1>Smoke page</h1>
<img id="aldrin" src="aldrin.jpg"><img src="multiface.jpg"><img src="noface.jpg">
<img src="armstrong.jpg"><img src="armstrong.jpg"><img src="armstrong.jpg#again">
<img class="s" src="small.jpg"><img class="p" src="pixel.gif"><img src="logo.svg">
<img src="corrupt.jpg"><img src="missing.jpg">
<img id="data" src="{data_url}"><img id="blob">
<div style="height:3000px"></div><img id="lazy" loading="lazy" src="hadfield.jpg">
<script>fetch("whitson.jpg").then(r => r.blob())
  .then(b => {{ document.getElementById("blob").src = URL.createObjectURL(b); }});</script>
</body></html>"""
    (site / "page.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site", type=Path)
    ap.add_argument("--many", action="store_true", help="only regenerate many.html")
    args = ap.parse_args()
    args.site.mkdir(parents=True, exist_ok=True)
    if not args.many:
        page(args.site)
    many(args.site)


if __name__ == "__main__":
    main()
