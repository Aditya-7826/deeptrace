"""Hugging Face Space app (space/app.py): analysis logic, validation and UI construction."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deeptrace.serve.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "tests" / "fixtures" / "images"


def _load_app():
    spec = importlib.util.spec_from_file_location("deeptrace_space_app", ROOT / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()


class CountingFactory:
    def __init__(self, weights_dir: Path):
        self.weights_dir, self.calls = weights_dir, 0

    def __call__(self) -> Engine:
        self.calls += 1
        return Engine(self.weights_dir, threads=2)


@pytest.fixture(scope="module")
def demo(exported):
    factory = CountingFactory(exported[0])
    d = app.Demo(factory)
    d.factory = factory
    return d


def test_analyze_returns_faces_table_and_heatmaps(demo):
    r = demo.analyze(str(IMAGES / "real" / "aldrin.jpg"))
    assert len(r["rows"]) == 1
    face, verdict, prob, conf, box = r["rows"][0]
    assert face == 1 and verdict in {"Real", "Fake", "Uncertain"}
    assert prob.endswith("%") and 0 <= float(prob[:-1]) <= 100
    assert len(box.split(", ")) == 4
    assert r["choices"] == [r["selected"]]
    rgb = np.asarray(Image.open(IMAGES / "real" / "aldrin.jpg").convert("RGB"))
    assert r["annotated"].shape == rgb.shape
    assert not np.array_equal(r["annotated"], rgb), "boxes are drawn"
    assert r["crop"].shape == (224, 224, 3)
    assert r["overlay"].ndim == 3 and max(r["overlay"].shape[:2]) <= 800
    assert r["summary"].startswith("**1 face found:**")


def test_engine_is_created_once(demo):
    demo.analyze(str(IMAGES / "real" / "hadfield.jpg"))
    demo.analyze(str(IMAGES / "real" / "hadfield.jpg"))
    assert demo.factory.calls == 1


def test_heatmap_for_another_face(demo):
    rgb = app.read_image(str(IMAGES / "real" / "aldrin.jpg"))
    h = demo.explain(rgb, 0)
    assert h["crop"].shape == (224, 224, 3)
    with pytest.raises(app.DemoError, match="no heatmap"):
        demo.explain(rgb, 5)


def test_no_face_and_tiny_face(demo):
    r = demo.analyze(str(IMAGES / "real" / "noface.jpg"))
    assert r["rows"] == [] and r["crop"] is None and "No face" in r["summary"]
    r = demo.analyze(str(IMAGES / "fake" / "tinyface.jpg"))
    assert r["rows"] == [] and "too small" in r["summary"]


@pytest.mark.parametrize(
    ("name", "data", "match"),
    [
        ("empty.jpg", b"", "empty"),
        ("doc.pdf", b"%PDF-1.7 fake", "Unsupported file type"),
        ("image.svg", b"<svg xmlns='http://www.w3.org/2000/svg'/>", "Unsupported file type"),
        ("corrupt.jpg", (IMAGES / "real" / "corrupt.jpg").read_bytes(), "could not be decoded"),
    ],
)
def test_bad_uploads_give_clear_errors(tmp_path, demo, name, data, match):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(app.DemoError, match=match):
        demo.analyze(str(path))


def test_no_upload(demo):
    with pytest.raises(app.DemoError, match="Upload an image"):
        demo.analyze(None)


def test_upload_size_cap(tmp_path, monkeypatch):
    path = tmp_path / "big.jpg"
    path.write_bytes(b"\xff\xd8\xff" + b"\0" * 2000)
    monkeypatch.setattr(app.LIMITS, "max_upload_bytes", 1000)
    with pytest.raises(app.DemoError, match="larger than"):
        app.read_image(str(path))


def test_pixel_cap(tmp_path, monkeypatch):
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 300)).save(path)
    monkeypatch.setattr(app.LIMITS, "max_pixels", 100_000)
    with pytest.raises(app.DemoError, match="too many pixels"):
        app.read_image(str(path))


def test_file_type_comes_from_content_not_name(tmp_path):
    buf = io.BytesIO()
    Image.open(IMAGES / "real" / "aldrin.jpg").save(buf, "PNG")
    path = tmp_path / "actually_png.jpg"
    path.write_bytes(buf.getvalue())
    assert app.read_image(str(path)).ndim == 3


def test_missing_weights_message_instead_of_crash(tmp_path):
    error = app.ensure_weights(tmp_path / "weights")  # this code version has no release yet
    assert error and "could not be downloaded" in error
    d = app.Demo(lambda: Engine(tmp_path / "weights"), startup_error=error)
    with pytest.raises(app.DemoError, match="could not be downloaded"):
        d.analyze(str(IMAGES / "real" / "aldrin.jpg"))
    assert d.model_info().startswith("**Model not loaded.**")


def test_engine_error_is_shown_not_raised(tmp_path):
    d = app.Demo(lambda: Engine(tmp_path))  # no model.json
    with pytest.raises(app.DemoError, match="weights not found"):
        d.engine()


def test_existing_weights_skip_download(exported):
    assert app.ensure_weights(exported[0]) is None


def test_choice_round_trip():
    face = {"index": 2, "label": "fake", "probability_fake": 0.91}
    assert app.choice_index(app.face_choice(face)) == 2
    with pytest.raises(app.DemoError):
        app.choice_index("nonsense")


def test_summary_counts_labels():
    faces = [{"label": "real"}, {"label": "fake"}, {"label": "fake"}]
    assert app.summarize(faces).startswith("**3 faces found:** 1 real, 2 fake.")


def test_examples_are_bundled_with_sources():
    ex = app.examples()
    assert len(ex) == 3  # aldrin, hadfield, two_people and all(Path(p[0]).exists() for p in ex)
    assert (ROOT / "space" / "examples" / "SOURCES.md").exists()


def test_ui_builds(demo):
    gr = pytest.importorskip("gradio")
    ui = app.build_ui(demo)
    assert isinstance(ui, gr.Blocks)
    assert "Model `" in app.Demo.model_info(demo)


def test_space_front_matter():
    import yaml

    text = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    meta = yaml.safe_load(text.split("---\n")[1])
    assert meta["sdk"] == "docker" and meta["app_port"] == 7860
    assert meta["license"] == "mit"
    assert len(meta["short_description"]) <= 60  # Hugging Face limit


def test_space_dockerfile_uses_the_lock_and_a_release_tag():
    text = (ROOT / "space" / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev --extra space" in text
    assert "ARG DEEPTRACE_REF=v" in text and "archive/refs/tags/${DEEPTRACE_REF}" in text
    assert "--uid 1000" in text and "USER user" in text
    assert "GRADIO_SERVER_PORT=7860" in text and "GRADIO_ANALYTICS_ENABLED=False" in text
    assert 'CMD ["python", "space/app.py"]' in text


def test_weights_dir_matches_the_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPTRACE_WEIGHTS_DIR", str(tmp_path))
    assert app.weights_dir() == tmp_path
    monkeypatch.delenv("DEEPTRACE_WEIGHTS_DIR")
    assert app.weights_dir() == Path("weights")
