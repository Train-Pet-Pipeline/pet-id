"""Full-stack pipeline combining all interfaces."""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from purrai_core.interfaces.detector import Detector
from purrai_core.interfaces.narrative import NarrativeGenerator
from purrai_core.interfaces.pose import PoseEstimator
from purrai_core.interfaces.reid import ReidEncoder
from purrai_core.interfaces.tracker import Tracker
from purrai_core.types import Detection, NarrativeOutput, PoseResult, ReidEmbedding, Track


@dataclass
class PipelineResult:
    """Result produced by one call to FullPipeline.process_frame()."""

    frame_idx: int
    detections: list[Detection]
    tracks: list[Track]
    embeddings: list[ReidEmbedding]
    poses: list[PoseResult]
    narrative: NarrativeOutput | None
    narrative_frame_idx: int | None = None


class FullPipeline:
    """Compose detect → track → reid → pose → (event-triggered) narrative."""

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        reid: ReidEncoder,
        pose: PoseEstimator,
        narrative: NarrativeGenerator,
        vlm_trigger_interval_frames: int = 60,
        parallel_reid_pose: bool = True,
    ) -> None:
        """Initialise pipeline with all five backend instances.

        Args:
            detector: Single-frame object detector.
            tracker: Stateful multi-object tracker.
            reid: Re-identification encoder.
            pose: Skeletal keypoint estimator.
            narrative: VLM-backed natural-language summariser.
            vlm_trigger_interval_frames: Narrative is generated every N frames
                (frame_idx > 0 and frame_idx % N == 0).
            parallel_reid_pose: When True, reid and pose will run concurrently
                (executor set up in a later task); stored here for configuration.
        """
        self.detector = detector
        self.tracker = tracker
        self.reid = reid
        self.pose = pose
        self.narrative = narrative
        self.vlm_interval = vlm_trigger_interval_frames
        self.parallel_reid_pose = parallel_reid_pose
        self._frame_buffer: list[np.ndarray] = []
        self._tracks_history: list[list[Track]] = []

        # Async narrative worker state
        self._logger = getLogger(__name__)
        self._narr_lock = threading.Lock()
        self._narr_in_flight = False
        self._narr_epoch = 0
        self._latest_narrative: NarrativeOutput | None = None
        self._latest_narrative_frame_idx: int | None = None
        self._narr_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="purrai-narrative"
        )
        self._rp_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="purrai-reid-pose")
            if parallel_reid_pose
            else None
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> PipelineResult:
        """Run the full pipeline for one frame.

        Steps: detect → track → reid → pose → (interval-triggered) narrative.

        Args:
            frame: BGR uint8 image array of shape (H, W, 3).
            frame_idx: Zero-based frame counter from the caller.

        Returns:
            PipelineResult with all intermediate outputs and optional narrative.
        """
        dets: list[Detection] = self.detector.detect(frame)
        # Pass frame to tracker so real trackers (ByteTrack via boxmot) get the image.
        tracks: list[Track] = self.tracker.update(dets, frame_idx, frame=frame)
        if self.parallel_reid_pose and self._rp_executor is not None:
            f_reid = self._rp_executor.submit(self.reid.encode, frame, tracks)
            f_pose = self._rp_executor.submit(self.pose.estimate, frame, tracks)
            embs: list[ReidEmbedding] = f_reid.result()
            poses: list[PoseResult] = f_pose.result()
        else:
            embs = self.reid.encode(frame, tracks)
            poses = self.pose.estimate(frame, tracks)

        self._frame_buffer.append(frame)
        self._tracks_history.append(tracks)

        # Keep buffer bounded to the trigger window.
        if len(self._frame_buffer) > self.vlm_interval:
            self._frame_buffer = self._frame_buffer[-self.vlm_interval :]
            self._tracks_history = self._tracks_history[-self.vlm_interval :]

        if frame_idx > 0 and frame_idx % self.vlm_interval == 0:
            with self._narr_lock:
                if not self._narr_in_flight and self._narr_executor is not None:
                    frames_snapshot = list(self._frame_buffer)
                    tracks_snapshot = list(self._tracks_history)
                    self._narr_in_flight = True
                    submit_epoch = self._narr_epoch
                    self._narr_executor.submit(
                        self._run_narrative,
                        frames_snapshot,
                        tracks_snapshot,
                        frame_idx,
                        submit_epoch,
                    )

        with self._narr_lock:
            narr = self._latest_narrative
            narr_idx = self._latest_narrative_frame_idx

        return PipelineResult(
            frame_idx=frame_idx,
            detections=dets,
            tracks=tracks,
            embeddings=embs,
            poses=poses,
            narrative=narr,
            narrative_frame_idx=narr_idx,
        )

    def _run_narrative(
        self,
        frames: list[np.ndarray],
        tracks_hist: list[list[Track]],
        origin_frame_idx: int,
        epoch: int,
    ) -> None:
        """Background worker: run VLM, update latest_narrative only if epoch still current.

        Exceptions are logged and swallowed to keep fast path unaffected.
        """
        try:
            out = self.narrative.describe(frames, tracks_hist)
            with self._narr_lock:
                if self._narr_epoch == epoch:
                    self._latest_narrative = out
                    self._latest_narrative_frame_idx = origin_frame_idx
        except Exception:
            self._logger.exception("narrative worker failed at frame %d", origin_frame_idx)
        finally:
            with self._narr_lock:
                self._narr_in_flight = False

    def shutdown(self) -> None:
        """Release background workers. Safe to call multiple times."""
        narr_exec = getattr(self, "_narr_executor", None)
        if narr_exec is not None:
            narr_exec.shutdown(wait=True)
            self._narr_executor = None
        rp_exec = getattr(self, "_rp_executor", None)
        if rp_exec is not None:
            rp_exec.shutdown(wait=True)
            self._rp_executor = None

    def __del__(self) -> None:
        """Best-effort cleanup when the pipeline instance is garbage-collected."""
        with contextlib.suppress(Exception):
            self.shutdown()

    def reset(self) -> None:
        """Clear frame buffer, tracks history, latest narrative, and reset tracker.

        Any in-flight narrative worker will have its result discarded because its epoch is now stale.
        """
        self._frame_buffer = []
        self._tracks_history = []
        with self._narr_lock:
            self._narr_epoch += 1
            self._latest_narrative = None
            self._latest_narrative_frame_idx = None
        self.tracker.reset()
