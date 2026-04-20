"""Tests for FullPipeline async narrative worker (M1.5 Task 2)."""

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
        return [ReidEmbedding(t.track_id, (0.1,) * 512) for t in tracks]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        return None


class _FakePose:
    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        return [PoseResult(t.track_id, []) for t in tracks]


class _SlowNarr:
    """Narrative that sleeps a configurable amount each call and counts invocations."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.call_count = 0

    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput:
        self.call_count += 1
        time.sleep(self.sleep_s)
        return NarrativeOutput(text=f"call {self.call_count}", confidence=0.9)


def _make_pipeline(narr: object, interval: int = 3) -> FullPipeline:
    return FullPipeline(
        detector=_FakeDet(),
        tracker=_FakeTr(),
        reid=_FakeReid(),
        pose=_FakePose(),
        narrative=narr,  # type: ignore[arg-type]
        vlm_trigger_interval_frames=interval,
    )


def test_fast_path_not_blocked_by_slow_narrative() -> None:
    """Each process_frame returns in <50ms even when narrative takes 300ms."""
    narr = _SlowNarr(sleep_s=0.3)
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for idx in range(10):
            t0 = time.perf_counter()
            p.process_frame(frame, frame_idx=idx)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, f"frame {idx} took {elapsed:.3f}s, expected <0.05s"
    finally:
        p.shutdown()


def test_latest_narrative_backfilled_into_subsequent_results() -> None:
    """After VLM completes, next process_frame attaches it to the returned result."""
    narr = _SlowNarr(sleep_s=0.05)
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for idx in range(4):
            p.process_frame(frame, frame_idx=idx)
        time.sleep(0.2)  # let worker finish
        result = p.process_frame(frame, frame_idx=5)
        assert result.narrative is not None
        assert result.narrative.text == "call 1"
        assert result.narrative_frame_idx == 3
    finally:
        p.shutdown()


def test_skip_when_worker_busy() -> None:
    """If VLM is still running when next interval hits, we skip (don't queue)."""
    narr = _SlowNarr(sleep_s=1.0)  # longer than 3 intervals
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        # Trigger interval at 3, 6, 9. First submit runs; next two should skip.
        for idx in range(10):
            p.process_frame(frame, frame_idx=idx)
        time.sleep(1.2)  # let worker finish
        assert narr.call_count == 1, f"expected 1 submit, got {narr.call_count}"
    finally:
        p.shutdown()


def test_reset_clears_narrative_state() -> None:
    """reset() drops latest_narrative but does not hang on in-flight worker."""
    narr = _SlowNarr(sleep_s=0.05)
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for idx in range(4):
            p.process_frame(frame, frame_idx=idx)
        time.sleep(0.2)
        p.reset()
        # After reset, latest_narrative should be None.
        result = p.process_frame(frame, frame_idx=0)
        assert result.narrative is None
        assert result.narrative_frame_idx is None
    finally:
        p.shutdown()


def test_reset_during_in_flight_worker_discards_stale_narrative() -> None:
    """If reset() is called while worker is still computing, worker's result is discarded."""
    narr = _SlowNarr(sleep_s=0.3)
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for idx in range(4):
            p.process_frame(frame, frame_idx=idx)
        # Worker is now running for frame 3 (still in its 300ms sleep). Reset immediately.
        p.reset()
        # Wait for the stale worker to finish.
        time.sleep(0.5)
        # Even though the stale worker completed, its result must not leak into state.
        result = p.process_frame(frame, frame_idx=0)
        assert result.narrative is None, "stale narrative leaked past reset()"
        assert result.narrative_frame_idx is None
    finally:
        p.shutdown()


def test_narrative_worker_exception_does_not_break_pipeline() -> None:
    """Worker-side exception is logged but does not crash process_frame or lock the slot."""

    class _BoomNarr:
        def __init__(self) -> None:
            self.call_count = 0

        def describe(
            self,
            frames: list[np.ndarray],
            tracks_history: list[list[Track]],
        ) -> NarrativeOutput:
            self.call_count += 1
            raise RuntimeError("boom")

    narr = _BoomNarr()
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for idx in range(10):
            p.process_frame(frame, frame_idx=idx)
        time.sleep(0.1)
        # Worker should have been submitted at idx 3, 6, 9 (all should run because each errors quickly).
        assert narr.call_count >= 1
    finally:
        p.shutdown()


def test_shutdown_returns_quickly() -> None:
    """shutdown() joins the narrative executor without hanging."""
    narr = _SlowNarr(sleep_s=0.05)
    p = _make_pipeline(narr, interval=3)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    for idx in range(4):
        p.process_frame(frame, frame_idx=idx)
    t0 = time.perf_counter()
    p.shutdown()
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"shutdown took {elapsed:.2f}s, expected <3s"


def test_shutdown_is_idempotent() -> None:
    """Calling shutdown twice is safe."""
    p = _make_pipeline(_SlowNarr(sleep_s=0.01), interval=3)
    p.shutdown()
    p.shutdown()  # must not raise
