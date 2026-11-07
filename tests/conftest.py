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


@pytest.fixture(scope="session")
def trained_run(debug_data, tmp_path_factory):
    """A tiny model trained for one epoch on the synthetic crops (run folder with best.pt)."""
    from deeptrace.train.loop import Trainer
    from tests.test_train import tiny_cfg

    out = tmp_path_factory.mktemp("evalrun")
    Trainer(tiny_cfg(debug_data, out, epochs=1, freeze_epochs=0)).fit()
    return out / "tiny"


@pytest.fixture(scope="session")
def exported(trained_run, debug_data, tmp_path_factory):
    """run_export on the tiny model; returns (export_dir, results_dir, report)."""
    import json

    from deeptrace.export.run import ExportOptions, run_export

    results = tmp_path_factory.mktemp("results")
    calib = {"calibration": {"temperature": 1.7, "threshold": 0.42, "uncertain_band": [0.4, 0.6]}}
    (results / "tiny").mkdir()
    (results / "tiny" / "metrics.json").write_text(json.dumps(calib), encoding="utf-8")
    out = tmp_path_factory.mktemp("export")
    report = run_export(ExportOptions(run_dir=trained_run, data_root=debug_data, out_dir=out,
                                      results_dir=results, threads=2, calib_n=12))  # fmt: skip
    return out, results, report
