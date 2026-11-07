"""End-to-end prepare on the fixture set (MTCNN on CPU)."""

import csv
import zipfile

import pytest

from deeptrace.data.prepare import MANIFEST_FIELDS, PrepareConfig, run_prepare
from deeptrace.data.splits import find_leaks
from deeptrace.data.video import frame_indices
from tests.conftest import fixture_prepare_dict

pytestmark = pytest.mark.slow


def _manifest(out):
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _journal(out):
    with (out / "prepare_journal.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_manifest_columns_and_files(prepared):
    out, _, report = prepared
    rows = _manifest(out)
    assert list(rows[0].keys()) == MANIFEST_FIELDS
    assert report["crops"] == len(rows) > 20
    for r in rows:
        assert (out / r["path"]).exists() and "\\" not in r["path"]
        assert r["split"] in ("train", "val", "test")
        assert len(r["phash"]) == 16


def test_no_leakage_across_splits(prepared):
    out, _, _ = prepared
    rows = _manifest(out)
    assert find_leaks(rows) == {"video_id": [], "identity_id": [], "near_duplicates": []}
    # Videos 000, 001 and 000_001 share identities, so they share one split
    splits = {r["video_id"]: r["split"] for r in rows if r["source_dataset"] == "ffpp"}
    assert (
        splits["ffpp:original/000"]
        == splits["ffpp:original/001"]
        == splits["ffpp:Deepfakes/000_001"]
    )
    assert splits["ffpp:original/002"] == splits["ffpp:Deepfakes/002_003"]


def test_near_duplicate_kept_together(prepared):
    out, _, _ = prepared
    ride = [r for r in _manifest(out) if r["path"].rsplit("/", 1)[-1].startswith("ride_")]
    assert len(ride) == 2 and ride[0]["split"] == ride[1]["split"]


def test_bad_inputs_skipped_and_logged(prepared):
    out, _, report = prepared
    reasons = {
        r["source_key"].split(":", 1)[1]: r["reason"]
        for r in _journal(out)
        if r["status"] == "skipped" and not r["frame_index"]
    }
    assert reasons["real/corrupt.jpg"] == "corrupt_image"
    assert reasons["fake/empty.jpg"] == "empty_file"
    assert reasons["real/noface.jpg"] == "no_face"
    assert reasons["fake/tinyface.jpg"] == "face_too_small"
    assert reasons["original_sequences/youtube/c23/videos/999.mp4"] == "unreadable_video"
    assert report["skipped"]["corrupt_image"] == 1


def test_videos_give_evenly_spaced_frames(prepared):
    out, cfg, _ = prepared
    frames = sorted(
        int(r["frame_index"])
        for r in _journal(out)
        if r["source_key"].endswith("videos/000.mp4") and r["status"] == "ok"
    )
    assert frames == frame_indices(12, cfg.frames_per_video)  # fixture videos have 12 frames


def test_resume_skips_processed_files(prepared):
    out, cfg, report = prepared
    before = _journal(out)
    again = run_prepare(cfg)
    assert _journal(out) == before  # nothing re-processed, nothing appended
    assert again["crops"] == report["crops"]


def test_interrupted_run_resumes(tmp_path):
    cfg = PrepareConfig.from_dict(fixture_prepare_dict(tmp_path))
    partial = run_prepare(cfg, limit=5)
    assert partial["pending_after_run"] > 0
    full = run_prepare(cfg)
    assert full["pending_after_run"] == 0
    first_rows = {}
    for r in _journal(tmp_path):  # a source's rows are written together, exactly once
        first_rows.setdefault(r["source_key"], []).append(r["frame_index"])
    for frames in first_rows.values():
        assert len(frames) == len(set(frames))
    assert len(first_rows) == full["source_files"]


def test_package_kaggle_dataset(prepared, tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "package_kaggle_dataset.py"
    spec = importlib.util.spec_from_file_location("pkg", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, _, report = prepared
    zip_path = mod.package(out, tmp_path, "deeptrace-crops-test", "someone")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "manifest.csv" in names
    assert sum(n.endswith(".jpg") for n in names) == report["crops"]
    assert (zip_path.parent / "dataset-metadata.json").exists()
