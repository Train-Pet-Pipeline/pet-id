"""MMPose AP-10K pose estimation backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from mmpose.apis import inference_topdown, init_model

from purrai_core.backends.pose_schema import AP10K_KPT_NAMES
from purrai_core.interfaces.pose import PoseEstimator
from purrai_core.types import Keypoint, PoseResult, Track


class MMPosePoseEstimator(PoseEstimator):
    """Pose estimator backend using MMPose AP-10K (17-keypoint animal skeleton)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialise the estimator from a config section dict.

        Args:
            cfg: dict with keys ``config``, ``checkpoint_url``, ``device``,
                and ``keypoint_threshold``.
        """
        self.cfg = cfg
        self.keypoint_threshold = float(cfg["keypoint_threshold"])
        self.model = init_model(
            config=str(cfg["config"]),
            checkpoint=str(cfg["checkpoint_url"]),
            device=str(cfg["device"]),
        )

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        """Return per-track AP-10K keypoints for *frame*.

        Args:
            frame: BGR image as ``(H, W, 3)`` uint8 array.
            tracks: active tracks whose bounding boxes are used as pose hints.

        Returns:
            One :class:`~purrai_core.types.PoseResult` per input track.
            Keypoints below ``keypoint_threshold`` are omitted.
        """
        out: list[PoseResult] = []
        if not tracks:
            return out
        bboxes = np.array(
            [[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] for t in tracks],
            dtype=np.float32,
        )
        results = inference_topdown(self.model, frame, bboxes)
        for t, r in zip(tracks, results, strict=False):
            kpts_xy = r.pred_instances.keypoints[0]
            kpts_score = r.pred_instances.keypoint_scores[0]
            kps = [
                Keypoint(
                    name=AP10K_KPT_NAMES[i],
                    x=float(kpts_xy[i, 0]),
                    y=float(kpts_xy[i, 1]),
                    score=float(kpts_score[i]),
                )
                for i in range(len(AP10K_KPT_NAMES))
                if float(kpts_score[i]) >= self.keypoint_threshold
            ]
            out.append(PoseResult(track_id=t.track_id, keypoints=kps))
        return out
