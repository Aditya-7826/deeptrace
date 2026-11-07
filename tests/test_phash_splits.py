import random

from deeptrace.data.io import load_image
from deeptrace.data.phash import from_hex, hamming, near_duplicate_pairs, phash, to_hex
from deeptrace.data.splits import (
    assign_groups,
    find_leaks,
    identity_groups,
    near_duplicate_groups,
    remove_cross_split_near_duplicates,
)
from tests.conftest import IMAGES


def test_phash_near_duplicate_vs_different():
    a, _ = load_image(IMAGES / "fake/ride.jpg")
    b, _ = load_image(IMAGES / "fake/ride_recompressed.jpg")
    c, _ = load_image(IMAGES / "fake/whitson.jpg")
    assert hamming(phash(a), phash(b)) <= 4
    assert hamming(phash(a), phash(c)) > 10
    assert from_hex(to_hex(phash(a))) == phash(a)


def test_bucketed_search_matches_brute_force():
    rng = random.Random(0)
    base = [rng.getrandbits(64) for _ in range(60)]
    hashes = base + [h ^ (1 << rng.randrange(64)) ^ (1 << rng.randrange(64)) for h in base[:20]]
    fast = set(near_duplicate_pairs(hashes, 4))
    brute = {
        (i, j)
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if hamming(hashes[i], hashes[j]) <= 4
    }
    assert fast == brute and len(brute) >= 20


def test_ffpp_pairs_share_a_group():
    ids = ["ffpp:000", "ffpp:003", "ffpp:000|ffpp:003", "ffpp:005", "ffpp:003|ffpp:007", "ffpp:007"]
    groups = identity_groups(ids)
    assert groups[0] == groups[1] == groups[2] == groups[4] == groups[5]
    assert groups[3] != groups[0]


def test_assign_groups_deterministic_and_complete():
    sizes = {f"g{i}": 1 + i % 3 for i in range(50)}
    a = assign_groups(sizes, (0.8, 0.1, 0.1), seed=1)
    assert a == assign_groups(sizes, (0.8, 0.1, 0.1), seed=1)
    assert set(a) == set(sizes) and set(a.values()) == {"train", "val", "test"}


def _row(path, split, phash_hex, video="", ident=None):
    return {
        "path": path,
        "split": split,
        "phash": phash_hex,
        "video_id": video,
        "identity_id": ident or path,
    }


def test_near_duplicate_groups_and_removal():
    h = 0x0F0F0F0F0F0F0F0F
    hashes = [to_hex(h), to_hex(h ^ 0b111), to_hex(~h & (2**64 - 1))]
    g = near_duplicate_groups(hashes)
    assert g[0] == g[1] != g[2]
    rows = [
        _row("a", "train", hashes[0]),
        _row("b", "test", hashes[1]),
        _row("c", "val", hashes[2]),
    ]
    kept, removed = remove_cross_split_near_duplicates(rows)
    assert [r["path"] for r in removed] == ["a"]  # the test copy wins
    assert find_leaks(kept) == {"video_id": [], "identity_id": [], "near_duplicates": []}


def test_find_leaks_reports_each_kind():
    rows = [
        _row("a", "train", "0000000000000000", video="v1", ident="p1|p2"),
        _row("b", "test", "ffffffffffffffff", video="v1", ident="p2"),
        _row("c", "val", "0000000000000001"),
    ]
    leaks = find_leaks(rows)
    assert leaks["video_id"] == ["v1"]
    assert leaks["identity_id"] == ["p2"]
    assert leaks["near_duplicates"] == [("a", "c")]
