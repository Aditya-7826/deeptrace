import json

from deeptrace.data.shortcuts import class_stats, shortcut_warnings
from deeptrace.data.sources import celebdf, ffpp, image_folder
from tests.conftest import FIXTURES


def test_image_folder_labels():
    items = image_folder(FIXTURES / "images", "imgs")
    labels = {i.rel: i.label for i in items}
    assert labels["real/aldrin.jpg"] == "real" and labels["fake/ride.jpg"] == "fake"
    assert all(i.video_id == "" and i.kind == "image" for i in items)


def test_ffpp_ids():
    items = {i.rel.rsplit("/", 1)[-1]: i for i in ffpp(FIXTURES / "ffpp", methods=["Deepfakes"])}
    fake = items["000_001.mp4"]
    assert fake.label == "fake" and fake.method == "Deepfakes"
    assert fake.identities == ("ffpp:000", "ffpp:001")
    assert items["000.mp4"].identities == ("ffpp:000",)


def test_ffpp_official_split_files(tmp_path):
    (tmp_path / "train.json").write_text(json.dumps([["000", "001"]]), encoding="utf-8")
    (tmp_path / "val.json").write_text(json.dumps([["002", "003"]]), encoding="utf-8")
    (tmp_path / "test.json").write_text("[]", encoding="utf-8")
    items = {
        i.rel.rsplit("/", 1)[-1]: i
        for i in ffpp(FIXTURES / "ffpp", methods=["Deepfakes"], splits_dir=tmp_path)
    }
    assert items["000_001.mp4"].fixed_split == "train"
    assert items["002_003.mp4"].fixed_split == "val"
    assert items["999.mp4"].fixed_split == "unassigned"


def test_celebdf_is_test_only(tmp_path):
    for rel in (
        "Celeb-real/id0_0001.mp4",
        "Celeb-synthesis/id0_id5_0001.mp4",
        "YouTube-real/00010.mp4",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    items = {i.rel: i for i in celebdf(tmp_path)}
    assert {i.fixed_split for i in items.values()} == {"test"}
    assert items["Celeb-synthesis/id0_id5_0001.mp4"].identities == ("celebdf:id0", "celebdf:id5")
    (tmp_path / "list.txt").write_text("1 YouTube-real/00010.mp4\n", encoding="utf-8")
    assert [i.rel for i in celebdf(tmp_path, test_list=tmp_path / "list.txt")] == [
        "YouTube-real/00010.mp4"
    ]


def _r(label, w, q, pad=0.0):
    return {"label": label, "orig_width": w, "orig_height": w, "jpeg_quality": q, "pad_frac": pad}


def test_shortcut_warnings():
    same = [_r("real", 256, 90), _r("fake", 256, 90)]
    assert shortcut_warnings(class_stats(same), "d") == []
    skewed = [_r("real", 1024, 95, 0.0), _r("fake", 256, "", 0.3)]
    warns = shortcut_warnings(class_stats(skewed), "d")
    assert any("resolution" in w for w in warns)
    assert any("share of JPEG" in w for w in warns)
    assert any("padding" in w for w in warns)
