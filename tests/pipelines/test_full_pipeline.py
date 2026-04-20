"""Full pipeline integration tests with fake backends."""

import numpy as np

from purrai_core.pipelines.full_pipeline import FullPipeline, PipelineResult
from purrai_core.types import (
    BBox,
    Detection,
    NarrativeOutput,
    PoseResult,
    ReidEmbedding,
    Track,
)


class FakeDet:
    """Fake detector returning a single cat detection."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return one fixed detection."""
        return [Detection(BBox(0, 0, 10, 10), 0.9, 15, "cat")]


class FakeTr:
    """Fake tracker that wraps detections into tracks."""

    def update(
        self,
        dets: list[Detection],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Wrap each detection as a track."""
        return [Track(1, d.bbox, d.score, d.class_id, d.class_name) for d in dets]

    def reset(self) -> None:
        """No-op reset."""


class FakeReid:
    """Fake Re-ID encoder returning unit embeddings."""

    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]:
        """Return a 512-dim zero embedding for each track."""
        return [ReidEmbedding(t.track_id, (0.1,) * 512) for t in tracks]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        """Always return no match."""
        return None


class FakePose:
    """Fake pose estimator returning empty keypoints."""

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        """Return a pose result with no keypoints for each track."""
        return [PoseResult(t.track_id, []) for t in tracks]


class FakeNarr:
    """Fake narrative generator returning a fixed string."""

    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput:
        """Return a fixed narrative."""
        return NarrativeOutput(text="猫坐着", confidence=0.9)


def _make_pipeline(vlm_trigger_interval_frames: int = 60) -> FullPipeline:
    return FullPipeline(
        detector=FakeDet(),
        tracker=FakeTr(),
        reid=FakeReid(),
        pose=FakePose(),
        narrative=FakeNarr(),
        vlm_trigger_interval_frames=vlm_trigger_interval_frames,
    )


def test_pipeline_processes_single_frame() -> None:
    """A single frame produces tracks, embeddings, and poses but no narrative."""
    p = _make_pipeline()
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = p.process_frame(frame, frame_idx=0)
        assert isinstance(result, PipelineResult)
        assert len(result.tracks) == 1
        assert len(result.embeddings) == 1
        assert len(result.poses) == 1
    finally:
        p.shutdown()


def test_pipeline_triggers_vlm_on_interval() -> None:
    """Narrative is produced eventually after the interval frame (async)."""
    import time as _time

    p = _make_pipeline(vlm_trigger_interval_frames=3)
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx in range(4):
            p.process_frame(frame, frame_idx=idx)
        _time.sleep(0.1)  # give async worker time to finish (FakeNarr is instant)
        r5 = p.process_frame(frame, frame_idx=5)
        assert r5.narrative is not None
        assert r5.narrative.text == "猫坐着"
        assert r5.narrative_frame_idx == 3
    finally:
        p.shutdown()


def test_pipeline_result_has_narrative_frame_idx() -> None:
    """PipelineResult carries narrative_frame_idx telling consumers which frame VLM was run on."""
    p = _make_pipeline(vlm_trigger_interval_frames=3)
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        r0 = p.process_frame(frame, frame_idx=0)
        r3 = p.process_frame(frame, frame_idx=3)
        assert r0.narrative_frame_idx is None
        assert hasattr(r3, "narrative_frame_idx")
    finally:
        p.shutdown()


def test_pipeline_reset_clears_state() -> None:
    """After reset(), internal buffers are empty and tracker state is cleared."""
    p = _make_pipeline(vlm_trigger_interval_frames=60)
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        p.process_frame(frame, frame_idx=0)
        assert len(p._frame_buffer) == 1
        assert len(p._tracks_history) == 1
        p.reset()
        assert len(p._frame_buffer) == 0
        assert len(p._tracks_history) == 0
    finally:
        p.shutdown()
