"""scripts/update_readme_results.py: README results come only from results files."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "update_readme_results.py"
    spec = importlib.util.spec_from_file_location("update_readme_results", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


u = _load()


def _set(auc: float, n: int = 100) -> dict:
    return {"n": n, "auc": auc, "auc_ci95": [auc - 0.01, auc + 0.005], "accuracy": 0.9, "f1": 0.91,
            "ece_before": 0.08, "ece_after": 0.02, "uncertain_rate": 0.05}  # fmt: skip


def _metrics(run: str, cross: dict | None = None, gap: dict | None = None) -> dict:
    return {
        "run": run,
        "training_datasets": ["kaggle140k"],
        "calibration": {"temperature": 1.4, "threshold": 0.47, "uncertain_band": [0.4, 0.6],
                        "n_val": 4000, "val_ece_before": 0.07, "val_ece_after": 0.01},  # fmt: skip
        "in_dataset": {"kaggle140k": _set(0.987)},
        "cross_dataset": cross or {},
        "generalization_gap_auc": gap or {},
        "robustness": {"test_set": "kaggle140k", "n": 2000, "families": {
            "jpeg_quality": [{"level": "q90", "auc": 0.98}, {"level": "q30", "auc": 0.81}]}},
    }  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "README.md").write_text(f"# T\n\nintro\n\n{u.START}\nold\n{u.END}\n\nafter\n",
                                        encoding="utf-8")  # fmt: skip
    return tmp_path


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_nothing_measured_says_so_everywhere(repo):
    block = u.render(repo / "results", repo)
    assert "In-dataset AUC: not yet measured. Cross-dataset AUC: not yet measured." in block
    assert "Latency: not yet measured." in block
    assert "**API memory:** not yet measured" in block
    assert "Gallery: not yet measured." in block
    assert not any(ch.isdigit() for ch in block.replace("INT8", ""))


def test_numbers_come_from_the_files(repo):
    res = repo / "results"
    _write(res / "effb0_kaggle140k" / "metrics.json", _metrics("effb0_kaggle140k"))
    (res / "effb0_kaggle140k" / "robustness.png").write_bytes(b"png")
    _write(res / "benchmark.json", {
        "system": {"cpu": "Test CPU", "physical_cores": 6, "ram_gb": 7.3, "os": "Windows-11"},
        "settings": {"threads": 4, "runs": 200, "warmup": 20},
        "model": {"version": "effb0_kaggle140k-abc1234"},
        "results": [{"model": "onnx_int8", "pipeline": "model_only", "p50_ms": 4.24,
                     "p95_ms": 5.01, "peak_rss_mb": 99.6}]})  # fmt: skip
    _write(res / "api_memory.json", {"peak_rss_mb": 912.4, "limit_mb": 1500.0, "passed": True})
    _write(res / "effb0_kaggle140k" / "export.json", {
        "parity": {"n": 100, "max_abs_diff": 3e-5, "tolerance": 1e-4, "passed": True},
        "int8_attempts": [{"name": "per_channel", "accepted": True, "auc_drop": 0.0021,
                           "agreement": 0.991}]})  # fmt: skip
    block = u.render(res, repo)
    assert "0.987 [0.977, 0.992]" in block
    assert "Cross-dataset AUC (Celeb-DF v2): not yet measured" in block
    assert "Test CPU" in block and "| ONNX INT8 |" in block and "| 4.2 | 5.0 | 100 |" in block
    assert "peak RSS 912 MB" in block and "within the 1500 MB budget" in block
    assert "INT8 (per_channel) accepted: AUC drop 0.0021" in block
    assert 'src="results/effb0_kaggle140k/robustness.png"' in block
    assert "q30: 0.810" in block
    assert "temperature 1.400" in block and "5.0% of in-dataset test images" in block


def test_cross_dataset_gap_is_shown(repo):
    res = repo / "results"
    m = _metrics("effb0_main", cross={"celebdf": _set(0.712)},
                 gap={"kaggle140k->celebdf": 0.275})  # fmt: skip
    _write(res / "effb0_main" / "metrics.json", m)
    block = u.render(res, repo)
    assert "| Celeb-DF v2 | cross-dataset |" in block
    assert "AUC drops by 0.275" in block
    assert "Cross-dataset AUC (Celeb-DF v2): not yet measured" not in block


def test_rejected_int8_is_reported(repo):
    res = repo / "results"
    _write(res / "effb0_main" / "metrics.json", _metrics("effb0_main"))
    _write(res / "effb0_main" / "export.json", {
        "parity": {"n": 100, "max_abs_diff": 3e-5, "tolerance": 1e-4, "passed": True},
        "int8_attempts": [{"name": "per_tensor", "accepted": False}]})  # fmt: skip
    assert "INT8 rejected by the accuracy gate" in u.render(res, repo)


def test_compare_tables_are_embedded(repo):
    res = repo / "results"
    (res / "comparison.md").write_text("## Architecture comparison (FF++)\n\n| a |\n", "utf-8")
    block = u.render(res, repo)
    assert "### Architecture comparison (FF++)" in block
    assert "### Ablations\n\nNot yet measured" in block


def test_main_rewrites_only_the_block_and_check_detects_staleness(repo):
    args = ["--readme", str(repo / "README.md"), "--results", str(repo / "results")]
    assert u.main([*args, "--check"]) == 1
    assert u.main(args) == 0
    text = (repo / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# T\n\nintro\n\n") and text.endswith(f"{u.END}\n\nafter\n")
    assert "old" not in text
    assert u.main([*args, "--check"]) == 0


def test_markers_required(tmp_path):
    with pytest.raises(SystemExit):
        u.replace_block("no markers", "x")


def test_repo_readme_matches_results():
    assert u.main(["--check"]) == 0, "run: uv run python scripts/update_readme_results.py"
