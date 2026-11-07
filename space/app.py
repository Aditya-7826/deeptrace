"""DeepTrace demo for Hugging Face Spaces (Gradio, free CPU tier).

Upload an image -> per-face verdicts + a Grad-CAM heatmap. Scoring uses the ONNX serving
model (INT8 when it passed the accuracy gate); the FP32 PyTorch model is loaded only when
the first heatmap is drawn. Weights come from the GitHub Release at startup and are checked
against the SHA-256 sums in deeptrace/weights.json.

Run locally (no Docker needed):
    uv sync --extra space
    uv run python -m deeptrace download-weights
    uv run python space/app.py              # http://127.0.0.1:7860
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deeptrace.data.io import ImageLoadError, load_image
from deeptrace.export.download import DownloadError, default_dest, download_weights
from deeptrace.serve.engine import Engine, MissingWeightsError
from deeptrace.serve.settings import DISCLAIMER, Settings
from deeptrace.serve.validation import ALLOWED, decoded_matches, sniff_type

LOG = logging.getLogger("deeptrace.space")
HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Aditya-7826/deeptrace"
LIMITS = Settings()  # same upload and pixel limits as the API
COLORS = {"real": (47, 158, 68), "fake": (224, 49, 49), "uncertain": (230, 119, 0)}
NAMES = {"real": "Real", "fake": "Fake", "uncertain": "Uncertain"}
TABLE_HEADERS = ["Face", "Verdict", "P(fake)", "Detector confidence", "Box (x1, y1, x2, y2)"]


class DemoError(ValueError):
    """A problem the user should see (bad upload, no weights)."""


def weights_dir() -> Path:
    """Same folder as the CLI and API: $DEEPTRACE_WEIGHTS_DIR, else ./weights."""
    return default_dest()


def ensure_weights(dest: Path) -> str | None:
    """Download released weights if model.json is missing. Returns an error message or None."""
    if (dest / "model.json").exists():
        return None
    try:
        for row in download_weights(dest):
            LOG.info("%s: %s", row["name"], row["status"])
    except DownloadError as exc:  # includes ChecksumMismatch
        return f"Model weights could not be downloaded: {exc}"
    return None


class Demo:
    """Holds the engine (created once, lazily) and the analysis functions behind the UI."""

    def __init__(self, engine_factory: Callable[[], Engine], startup_error: str | None = None):
        self._factory = engine_factory
        self._engine: Engine | None = None
        self._error = startup_error
        self._lock = threading.Lock()

    def engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                if self._error:
                    raise DemoError(self._error)
                try:
                    self._engine = self._factory()
                except MissingWeightsError as exc:
                    self._error = str(exc)
                    raise DemoError(self._error) from exc
            return self._engine

    def model_info(self) -> str:
        try:
            m = self.engine().meta
        except DemoError as exc:
            return f"**Model not loaded.** {exc}"
        cal = "calibrated (temperature scaling)" if m.get("calibrated") else "not calibrated"
        return (
            f"Model `{m['model_name']}` · version `{m['run']}-{m['git_commit'][:7]}` · "
            f"scoring with `{m['serving_model']}` · {cal} · decision threshold "
            f"{m['threshold']:.2f}, uncertain band {m['uncertain_band'][0]:.2f}"
            f"–{m['uncertain_band'][1]:.2f}"
        )

    # ---- analysis (no Gradio types: unit-tested directly) ------------------------------

    def analyze(self, path: str | None) -> dict[str, Any]:
        """Detect faces, score them, and draw the heatmap for the face most likely fake."""
        rgb = read_image(path)
        engine = self.engine()
        result = engine.predict(rgb)
        if result["status"] != "ok":
            msg = ("No face found." if result["status"] == "no_face"
                   else "The face is too small to analyse (needs at least 64 px).")  # fmt: skip
            return {"summary": f"**{msg}**", "annotated": rgb, "rows": [], "choices": [],
                    "selected": None, "crop": None, "overlay": None}  # fmt: skip
        faces = result["faces"]
        top = max(faces, key=lambda f: f["probability_fake"])["index"]
        heat = self.explain(rgb, top)
        return {
            "summary": summarize(faces),
            "annotated": annotate(rgb, faces),
            "rows": table_rows(faces),
            "choices": [face_choice(f) for f in faces],
            "selected": face_choice(faces[top]),
            **heat,
        }

    def explain(self, rgb: np.ndarray, face_index: int) -> dict[str, Any]:
        out = self.engine().explain(rgb, face_index)
        if out.get("status") != "ok":
            raise DemoError(f"no heatmap for face {face_index + 1} ({out.get('status')})")
        return {"crop": decode_png(out["crop_overlay_png"]),
                "overlay": decode_png(out["image_overlay_png"])}  # fmt: skip


# ---- helpers ------------------------------------------------------------------------------


def read_image(path: str | None) -> np.ndarray:
    """The API's checks: size cap, real file type from magic bytes, decoder agreement."""
    if not path:
        raise DemoError("Upload an image first.")
    p = Path(path)
    size = p.stat().st_size
    if size == 0:
        raise DemoError("The file is empty.")
    if size > LIMITS.max_upload_bytes:
        raise DemoError(f"The file is larger than {LIMITS.max_upload_bytes // 2**20} MB.")
    data = p.read_bytes()
    kind = sniff_type(data)
    if kind not in ALLOWED:
        raise DemoError("Unsupported file type: upload a JPEG, PNG, WebP or GIF image.")
    try:
        rgb, meta = load_image(data, LIMITS.max_pixels)
    except ImageLoadError as exc:
        if exc.reason == "too_many_pixels":
            raise DemoError("The image has too many pixels (limit 40 megapixels).") from exc
        raise DemoError(f"The image could not be decoded ({exc.reason}).") from exc
    if not decoded_matches(kind, meta.format):
        raise DemoError("The file's signature and content disagree.")
    return rgb


def decode_png(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def face_choice(face: dict[str, Any]) -> str:
    return f"Face {face['index'] + 1}: {NAMES[face['label']]} ({face['probability_fake']:.0%} fake)"


def choice_index(choice: str | None) -> int:
    try:
        return int(str(choice).split(":")[0].removeprefix("Face ").strip()) - 1
    except ValueError as exc:
        raise DemoError("Pick a face from the list.") from exc


def table_rows(faces: list[dict[str, Any]]) -> list[list[Any]]:
    return [[f["index"] + 1, NAMES[f["label"]], f"{f['probability_fake']:.1%}",
             f"{f['detection_confidence']:.2f}", ", ".join(f"{v:.0f}" for v in f["box"])]
            for f in faces]  # fmt: skip


def summarize(faces: list[dict[str, Any]]) -> str:
    counts = {k: sum(f["label"] == k for f in faces) for k in NAMES}
    parts = [f"{n} {NAMES[k].lower()}" for k, n in counts.items() if n]
    lead = f"**{len(faces)} face{'s' if len(faces) != 1 else ''} found:** " + ", ".join(parts)
    return lead + ". The heatmap below shows the face most likely to be fake."


def annotate(rgb: np.ndarray, faces: list[dict[str, Any]]) -> np.ndarray:
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    width = max(2, round(max(img.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(max(img.size) / 45)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for f in faces:
        color = COLORS[f["label"]]
        x1, y1, x2, y2 = f["box"]
        d.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f"{f['index'] + 1}: {NAMES[f['label']]} {f['probability_fake']:.0%}"
        tb = d.textbbox((x1, y1), text, font=font)
        ty = y1 - (tb[3] - tb[1]) - 2 * width if y1 > tb[3] - tb[1] + 2 * width else y2
        tb = d.textbbox((x1, ty), text, font=font)
        d.rectangle((tb[0] - width, tb[1] - width, tb[2] + width, tb[3] + width), fill=color)
        d.text((x1, ty), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


# ---- UI -----------------------------------------------------------------------------------

INTRO = f"""# DeepTrace: is this face real?
Upload a photo. Each face is detected, scored by a deepfake classifier, and explained with a
Grad-CAM heatmap (red = regions that drove the verdict).
**Note:** {DISCLAIMER}. Code, metrics and limitations: [{REPO_URL}]({REPO_URL})."""

PRIVACY = (
    "Uploads are processed in memory for this request and are not stored by the app. "
    "Please do not upload images of people without their consent."
)


def examples() -> list[list[str]]:
    folder = HERE / "examples"
    return [[str(p)] for p in sorted(folder.glob("*.jpg"))] if folder.is_dir() else []


def build_ui(demo: Demo):
    # No usage analytics from a demo that receives people's photos (must precede the import).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    def run(path):
        try:
            r = demo.analyze(path)
        except DemoError as exc:
            raise gr.Error(str(exc)) from exc
        return (r["summary"], r["annotated"], r["rows"],
                gr.update(choices=r["choices"], value=r["selected"],
                          interactive=len(r["choices"]) > 1),
                r["crop"], r["overlay"])  # fmt: skip

    def rerun_heatmap(path, choice):
        if not choice:
            return None, None
        try:
            h = demo.explain(read_image(path), choice_index(choice))
        except DemoError as exc:
            raise gr.Error(str(exc)) from exc
        return h["crop"], h["overlay"]

    with gr.Blocks(title="DeepTrace") as ui:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Image (JPEG, PNG, WebP or GIF, up to 10 MB)",
                                 file_types=[".jpg", ".jpeg", ".png", ".webp", ".gif"],
                                 type="filepath")  # fmt: skip
                button = gr.Button("Analyze", variant="primary")
                if ex := examples():
                    gr.Examples(ex, inputs=upload, label="Examples (real, public-domain photos)")
                gr.Markdown(PRIVACY)
            with gr.Column(scale=1):
                summary = gr.Markdown()
                boxes = gr.Image(label="Detected faces", interactive=False, format="png")
                table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True)
        face = gr.Dropdown(label="Heatmap for", choices=[], interactive=False)
        with gr.Row():
            crop = gr.Image(label="Grad-CAM on the face crop (what the model saw)",
                            interactive=False, format="png")  # fmt: skip
            overlay = gr.Image(label="Grad-CAM on the whole image", interactive=False,
                               format="png")  # fmt: skip
        gr.Markdown(demo.model_info())

        button.click(run, upload, [summary, boxes, table, face, crop, overlay])
        upload.upload(run, upload, [summary, boxes, table, face, crop, overlay])
        face.input(rerun_heatmap, [upload, face], [crop, overlay])
    return ui


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    threads = int(os.environ.get("DEEPTRACE_THREADS", "2"))
    dest = weights_dir()
    error = ensure_weights(dest)
    demo = Demo(lambda: Engine(dest, threads=threads), startup_error=error)
    try:
        demo.engine()  # load at startup so the first visitor does not wait
    except DemoError as exc:
        LOG.error("%s", exc)
    ui = build_ui(demo)
    ui.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        max_file_size="10mb",
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
