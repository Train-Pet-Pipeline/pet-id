"""PoseEstimator contract tests."""

import numpy as np

from purrai_core.interfaces.pose import PoseEstimator
from purrai_core.types import BBox, PoseResult, Track


class FakePose:
    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]:
        return [PoseResult(track_id=t.track_id, keypoints=[]) for t in tracks]


def test_pose_protocol() -> None:
    p: PoseEstimator = FakePose()
    tracks = [Track(track_id=7, bbox=BBox(0, 0, 1, 1), score=0.9, class_id=16, class_name="dog")]
    results = p.estimate(np.zeros((64, 64, 3), dtype=np.uint8), tracks)
    assert results[0].track_id == 7
    assert results[0].keypoints == []
