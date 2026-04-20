from __future__ import annotations

import copy

from purrai_core.stitch import stitch_tracks
from purrai_core.types import BBox, Track


def _track(tid: int, f: int) -> tuple[int, list[Track]]:
    return (
        f,
        [Track(track_id=tid, bbox=BBox(0, 0, 10, 10), score=1.0, class_id=15, class_name="cat")],
    )


def _similar_vec(offset: float) -> tuple[float, ...]:
    # Unit vector [1,0,...] slightly perturbed — near-1 cosine to [1,0,...]
    return (1.0 - offset, offset, 0.0, 0.0)


def test_simple_death_birth_within_gap_stitched():
    # Track 1 ends at frame 4, track 2 is born at frame 6 with similar embedding.
    tracks = [_track(1, f) for f in range(5)] + [_track(2, f) for f in range(6, 11)]
    reids = [(f, {1: _similar_vec(0.01)}) for f in range(5)] + [
        (f, {2: _similar_vec(0.02)}) for f in range(6, 11)
    ]

    out = stitch_tracks(tracks, reids, cosine_threshold=0.9, max_gap_frames=5, embedding_window=3)
    # id 2 rewritten to 1 across its entire life
    assert {t.track_id for _, trs in out for t in trs} == {1}


def test_gap_exceeds_max_frames_no_stitch():
    tracks = [_track(1, f) for f in range(5)] + [_track(2, f) for f in range(100, 105)]
    reids = [(f, {1: _similar_vec(0.01)}) for f in range(5)] + [
        (f, {2: _similar_vec(0.01)}) for f in range(100, 105)
    ]

    out = stitch_tracks(tracks, reids, cosine_threshold=0.7, max_gap_frames=10, embedding_window=3)
    assert {t.track_id for _, trs in out for t in trs} == {1, 2}


def test_below_cosine_threshold_no_stitch():
    tracks = [_track(1, f) for f in range(5)] + [_track(2, f) for f in range(6, 11)]
    # Orthogonal vectors -> cosine ~= 0
    reids = [(f, {1: (1.0, 0.0, 0.0)}) for f in range(5)] + [
        (f, {2: (0.0, 1.0, 0.0)}) for f in range(6, 11)
    ]

    out = stitch_tracks(tracks, reids, cosine_threshold=0.9, max_gap_frames=5, embedding_window=3)
    assert {t.track_id for _, trs in out for t in trs} == {1, 2}


def test_empty_reids_is_noop_deep_copy():
    tracks = [_track(1, f) for f in range(5)]
    original = copy.deepcopy(tracks)
    out = stitch_tracks(tracks, [], cosine_threshold=0.7, max_gap_frames=5, embedding_window=3)
    # Same ids, input unchanged
    assert out == original
    assert tracks == original  # no mutation


def test_hungarian_picks_best_pair_among_multiple_candidates():
    # Two deaths (1, 2) and two births (3, 4). Only optimal pairing:
    # 1->3 (very similar), 2->4 (very similar). A greedy algorithm that picks
    # 1->4 first would lock 2 into the worse match.
    vec_a = (1.0, 0.0, 0.0)
    vec_b = (0.0, 1.0, 0.0)
    tracks = (
        [_track(1, 0), _track(1, 1), _track(1, 2)]
        + [_track(2, 0), _track(2, 1), _track(2, 2)]
        + [_track(3, 5), _track(3, 6)]
        + [_track(4, 5), _track(4, 6)]
    )
    # Flatten to per-frame tuples the function expects
    by_frame: dict[int, list[Track]] = {}
    for f, trs in tracks:
        by_frame.setdefault(f, []).extend(trs)
    tracks = sorted(by_frame.items())

    reids = [
        (0, {1: vec_a, 2: vec_b}),
        (1, {1: vec_a, 2: vec_b}),
        (2, {1: vec_a, 2: vec_b}),
        (5, {3: vec_a, 4: vec_b}),
        (6, {3: vec_a, 4: vec_b}),
    ]

    out = stitch_tracks(tracks, reids, cosine_threshold=0.9, max_gap_frames=5, embedding_window=2)
    # 3 rewritten to 1, 4 rewritten to 2
    final_ids = {t.track_id for _, trs in out for t in trs}
    assert final_ids == {1, 2}
