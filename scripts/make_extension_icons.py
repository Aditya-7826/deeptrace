"""Draw the extension's toolbar icons (a magnifier over a face outline).

uv run python scripts/make_extension_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "extension" / "icons"
SIZES = (16, 32, 48, 128)
BLUE = (59, 91, 219, 255)
WHITE = (255, 255, 255, 255)


def draw(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 512
    d.rounded_rectangle((16 * s, 16 * s, 496 * s, 496 * s), radius=110 * s, fill=BLUE)
    # face outline inside the lens
    d.ellipse((140 * s, 110 * s, 320 * s, 330 * s), outline=WHITE, width=round(34 * s))
    d.ellipse((190 * s, 180 * s, 214 * s, 204 * s), fill=WHITE)
    d.ellipse((246 * s, 180 * s, 270 * s, 204 * s), fill=WHITE)
    # magnifier handle
    d.line((300 * s, 310 * s, 420 * s, 430 * s), fill=WHITE, width=round(56 * s))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    big = draw()
    for n in SIZES:
        big.resize((n, n), Image.Resampling.LANCZOS).save(OUT / f"icon{n}.png", optimize=True)
        print(OUT / f"icon{n}.png")


if __name__ == "__main__":
    main()
