"""Tests for parallel Reid ∥ Pose branch (M1.5 Task 7)."""

from __future__ import annotations

import time

import numpy as np

from purrai_core.pipelines.full_pipeline import FullPipeline
from purrai_core.types import (
    BBox,
    Detection,
    NarrativeOutput,
    PoseResult,
    ReidEmbedding,
    Track,
)


class _FakeDet:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(BBox(0, 0, 10, 10), 0.9, 15, "cat")]


class _FakeTr:
    def update(
        self,
        dets: list[Detection],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        return [Track(1, d.bbox, d.score, d.class_id, d.class_name) for d in dets]

    def reset(self) -> None:
        return None


class _FakeReid:
    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]:
        return [ReidEmbedding(t.track_id, (0.2,) * 512) for t in tracks]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        return None


class _FakePose:
    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        return [PoseResult(t.track_id, []) for t in tracks]


class _FakeNarr:
    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput:
        return NarrativeOutput(text="x", confidence=0.5)


class _SleepyReid:
    """Reid that sleeps for fixed duration to make parallelism observable."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s

    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]:
        time.sleep(self.sleep_s)
        return [ReidEmbedding(t.track_id, (0.2,) * 512) for t in tracks]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        return None


class _SleepyPose:
    """Pose that sleeps for fixed duration to make parallelism observable."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        time.sleep(self.sleep_s)
        return [PoseResult(t.track_id, []) for t in tracks]


def _make(reid: object, pose: object, parallel: bool) -> FullPipeline:
    return FullPipeline(
        detector=_FakeDet(),
        tracker=_FakeTr(),
        reid=reid,  # type: ignore[arg-type]
        pose=pose,  # type: ignore[arg-type]
        narrative=_FakeNarr(),
        vlm_trigger_interval_frames=60,
        parallel_reid_pose=parallel,
    )


def test_parallel_and_serial_produce_identical_outputs() -> None:
    """Given deterministic fake backends, parallel vs serial must produce equal lists."""
    p_par = _make(_FakeReid(), _FakePose(), parallel=True)
    p_ser = _make(_FakeReid(), _FakePose(), parallel=False)
    try:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        r_par = p_par.process_frame(frame, frame_idx=0)
        r_ser = p_ser.process_frame(frame, frame_idx=0)
        assert r_par.embeddings == r_ser.embeddings
        assert r_par.poses == r_ser.poses
    finally:
        p_par.shutdown()
        p_ser.shutdown()


def test_parallel_branch_is_faster_than_serial_for_slow_backends() -> None:
    """With 200ms reid + 200ms pose, parallel is noticeably faster than serial."""
    reid = _SleepyReid(sleep_s=0.2)
    pose = _SleepyPose(sleep_s=0.2)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    p_ser = _make(reid, pose, parallel=False)
    try:
        t0 = time.perf_counter()
        p_ser.process_frame(frame, frame_idx=0)
        serial_ms = (time.perf_counter() - t0) * 1000
    finally:
        p_ser.shutdown()

    p_par = _make(_SleepyReid(0.2), _SleepyPose(0.2), parallel=True)
    try:
        t0 = time.perf_counter()
        p_par.process_frame(frame, frame_idx=0)
        parallel_ms = (time.perf_counter() - t0) * 1000
    finally:
        p_par.shutdown()

    # Serial should be ~400ms; parallel should be ~200ms. Generous slack for CI scheduler jitter.
    assert serial_ms > 350, f"serial={serial_ms:.1f}ms unexpectedly fast"
    assert parallel_ms < 350, f"parallel={parallel_ms:.1f}ms not faster than serial"


def test_parallel_shutdown_releases_executor() -> None:
    p = _make(_FakeReid(), _FakePose(), parallel=True)
    p.shutdown()
    # Second call must be safe.
    p.shutdown()
