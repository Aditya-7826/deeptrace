from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = FIXTURES / "images"


@pytest.fixture(scope="session")
def detector():
    from deeptrace.data.faces import FaceDetector
    from deeptrace.utils import set_cpu_threads

    set_cpu_threads(2)
    return FaceDetector("cpu")


def fixture_prepare_dict(output_dir: Path) -> dict:
    return {
        "raw_root": str(FIXTURES),
        "output_dir": str(output_dir),
        "datasets": [
            {
                "name": "imgs",
                "type": "image_folder",
                "root": "images",
                "options": {"fake_method": "synthetic"},
            },
            {
                "name": "ffpp",
                "type": "ffpp",
                "root": "ffpp",
                "options": {"compression": "c23", "methods": ["Deepfakes"]},
            },
        ],
        "frames_per_video": 5,
        "split_fractions": [0.5, 0.25, 0.25],
        "device": "cpu",
        "threads": 2,
    }


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Run the full prepare pipeline once on the fixtures; shared by several tests."""
    from deeptrace.data.prepare import PrepareConfig, run_prepare

    out = tmp_path_factory.mktemp("prepared")
    cfg = PrepareConfig.from_dict(fixture_prepare_dict(out))
    report = run_prepare(cfg)
    yield out, cfg, report
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture(scope="session")
def debug_data(tmp_path_factory):
    """Synthetic crops: 40 train/val + 20 test + 20 cross-dataset ("debugx") test."""
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_debug_data.py"
    spec = importlib.util.spec_from_file_location("make_debug_data", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("debugdata")
    argv = sys.argv
    sys.argv = ["x", "--out", str(out), "--n", "40", "--n-test", "20", "--n-cross", "20"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return out
