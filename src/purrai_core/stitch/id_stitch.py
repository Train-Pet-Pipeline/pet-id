"""Render-time id stitching via cosine similarity + Hungarian matching.

Pure function over already-tracked frames and aligned reid embeddings.
Rewrites birth ids back to deceased ids when they likely represent the
same individual (cosine similarity above threshold, frame gap within
max_gap_frames). Never mutates inputs.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from purrai_core.types import Track

log = logging.getLogger(__name__)

_INF = float("inf")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pool_window(
    embeddings_by_frame: dict[int, dict[int, list[float]]],
    track_id: int,
    center_frame: int,
    direction: str,
    window: int,
) -> np.ndarray | None:
    """Pool up to `window` embeddings for `track_id` centered around a frame.

    direction='backward' pools [center_frame - window + 1 .. center_frame].
    direction='forward' pools  [center_frame .. center_frame + window - 1].
    Missing frames (no reid for that id in that frame) are skipped, not padded.
    Returns mean vector or None when no embedding found.
    """
    if direction == "backward":
        frames = range(center_frame - window + 1, center_frame + 1)
    else:
        frames = range(center_frame, center_frame + window)

    vecs: list[list[float]] = []
    for f in frames:
        emb_map = embeddings_by_frame.get(f, {})
        if track_id in emb_map:
            vecs.append(emb_map[track_id])
    if not vecs:
        return None
    result: np.ndarray = np.mean(np.asarray(vecs, dtype=np.float64), axis=0)
    return result


def stitch_tracks(
    tracks: Sequence[tuple[int, list[Track]]],
    reids: Sequence[tuple[int, dict[int, Sequence[float]]]],
    *,
    cosine_threshold: float,
    max_gap_frames: int,
    embedding_window: int = 5,
) -> list[tuple[int, list[Track]]]:
    """Rewrite birth track ids to deceased ids when reid embeddings match.

    Args:
        tracks: Sequence of (frame_idx, list[Track]) as produced by running
            the pipeline over the video.
        reids: Sequence of (frame_idx, {track_id: embedding_vec}) aligned with
            the same frame indices. Missing frames or missing ids inside a
            frame are allowed.
        cosine_threshold: Minimum cosine similarity to accept a stitch (e.g. 0.7).
        max_gap_frames: Maximum frame gap between death and birth for candidacy.
        embedding_window: Frames pooled around death-last / birth-first for
            robust similarity.

    Returns:
        New list of (frame_idx, list[Track]) with stitched ids. Unmatched
        births keep original ids. Inputs are never mutated.
    """
    if not reids:
        log.warning("stitch_tracks: no reids provided, returning deep copy")
        return copy.deepcopy(list(tracks))

    # ----- 1. Index frame_idx -> embedding map (strip Sequence to dict[int,list]) -----
    emb_by_frame: dict[int, dict[int, list[float]]] = {}
    for f, emb_map in reids:
        emb_by_frame[int(f)] = {int(tid): list(v) for tid, v in emb_map.items()}

    # ----- 2. First/last frame per track_id -----
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    for f, trs in tracks:
        for tr in trs:
            first_frame.setdefault(tr.track_id, f)
            last_frame[tr.track_id] = f

    ids = sorted(first_frame.keys())
    if len(ids) < 2:
        return copy.deepcopy(list(tracks))

    # ----- 3. Pooled embedding for each track -----
    pool_death: dict[int, np.ndarray] = {}
    pool_birth: dict[int, np.ndarray] = {}
    for tid in ids:
        emb_death = _pool_window(emb_by_frame, tid, last_frame[tid], "backward", embedding_window)
        emb_birth = _pool_window(emb_by_frame, tid, first_frame[tid], "forward", embedding_window)
        if emb_death is not None:
            pool_death[tid] = emb_death
        if emb_birth is not None:
            pool_birth[tid] = emb_birth

    # ----- 4. Candidate pairs (death d, birth b) where b starts after d ends -----
    deaths = [tid for tid in ids if tid in pool_death]
    births = [tid for tid in ids if tid in pool_birth]
    if not deaths or not births:
        return copy.deepcopy(list(tracks))

    cost = np.full((len(deaths), len(births)), _INF, dtype=np.float64)
    for i, d in enumerate(deaths):
        for j, b in enumerate(births):
            if b == d:
                continue  # same track — never pair to self
            gap = first_frame[b] - last_frame[d]
            if gap <= 0 or gap > max_gap_frames:
                continue
            sim = _cosine(pool_death[d], pool_birth[b])
            if sim < cosine_threshold:
                continue
            cost[i, j] = 1.0 - sim

    # If every entry is +inf, nothing to do
    if not np.any(np.isfinite(cost)):
        log.info("stitch_tracks: no candidate pairs above threshold")
        return copy.deepcopy(list(tracks))

    # ----- 5. Hungarian assignment -----
    # linear_sum_assignment over a matrix with +inf allowed — we mask afterward.
    safe_cost = np.where(np.isfinite(cost), cost, 1e9)
    row_ind, col_ind = linear_sum_assignment(safe_cost)

    rewrite: dict[int, int] = {}  # birth_id -> death_id
    for i, j in zip(row_ind, col_ind, strict=False):
        i, j = int(i), int(j)
        if not math.isfinite(cost[i, j]):
            continue
        d = deaths[i]
        b = births[j]
        # Avoid cycles / re-binding ids already claimed
        if b in rewrite or d in rewrite.values():
            continue
        rewrite[b] = d

    if not rewrite:
        return copy.deepcopy(list(tracks))

    # ----- 6. Apply rewrites to a deep copy -----
    out: list[tuple[int, list[Track]]] = []
    for f, trs in tracks:
        new_trs: list[Track] = []
        for tr in trs:
            if tr.track_id in rewrite:
                new_trs.append(
                    Track(
                        track_id=rewrite[tr.track_id],
                        bbox=tr.bbox,
                        score=tr.score,
                        class_id=tr.class_id,
                        class_name=tr.class_name,
                    )
                )
            else:
                new_trs.append(tr)
        out.append((f, new_trs))
    return out
