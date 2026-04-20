"""MMPose AP-10K pose estimation backend tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mmpose")  # skip collection if mmpose is not installed

from purrai_core.backends.mmpose_pose import AP10K_KPT_NAMES, MMPosePoseEstimator  # noqa: E402
from purrai_core.types import BBox, Track  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = {
    "config": "configs/ap10k.py",
    "checkpoint_url": "https://example.com/ap10k.pth",
    "device": "cpu",
    "keypoint_threshold": 0.3,
}

_NUM_KPT = len(AP10K_KPT_NAMES)  # 17


def _make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        bbox=BBox(10.0, 20.0, 110.0, 120.0),
        score=0.9,
        class_id=15,
        class_name="cat",
    )


def _make_inference_result(scores: np.ndarray) -> MagicMock:
    """Build a fake MMPose DataSample result for one track."""
    kpts_xy = np.random.default_rng(0).random((_NUM_KPT, 2), dtype=np.float32) * 100
    result = MagicMock()
    result.pred_instances.keypoints = [kpts_xy]
    result.pred_instances.keypoint_scores = [scores]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mmpose_returns_poseresult_per_track() -> None:
    """One PoseResult per track; keypoints use AP-10K 17-point names."""
    scores = np.ones(_NUM_KPT, dtype=np.float32)  # all above threshold

    with (
        patch("purrai_core.backends.mmpose_pose.init_model") as mock_init,
        patch("purrai_core.backends.mmpose_pose.inference_topdown") as mock_infer,
    ):
        mock_init.return_value = MagicMock()
        mock_infer.return_value = [_make_inference_result(scores)]

        estimator = MMPosePoseEstimator(_CFG)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracks = [_make_track(track_id=42)]
        results = estimator.estimate(frame, tracks)

    assert len(results) == 1
    pr = results[0]
    assert pr.track_id == 42
    assert len(pr.keypoints) == _NUM_KPT
    kpt_names = [kp.name for kp in pr.keypoints]
    assert kpt_names == AP10K_KPT_NAMES


def test_mmpose_filters_low_confidence_keypoints() -> None:
    """Keypoints below keypoint_threshold are excluded from the result."""
    # First 3 keypoints are below 0.3; the rest are above.
    scores = np.full(_NUM_KPT, 0.8, dtype=np.float32)
    scores[:3] = 0.1  # below threshold

    with (
        patch("purrai_core.backends.mmpose_pose.init_model") as mock_init,
        patch("purrai_core.backends.mmpose_pose.inference_topdown") as mock_infer,
    ):
        mock_init.return_value = MagicMock()
        mock_infer.return_value = [_make_inference_result(scores)]

        estimator = MMPosePoseEstimator(_CFG)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracks = [_make_track(track_id=7)]
        results = estimator.estimate(frame, tracks)

    assert len(results) == 1
    pr = results[0]
    assert pr.track_id == 7
    # 3 dropped, 14 remaining
    assert len(pr.keypoints) == _NUM_KPT - 3
    # The surviving keypoints must not include the first 3 AP-10K names
    surviving_names = {kp.name for kp in pr.keypoints}
    for dropped_name in AP10K_KPT_NAMES[:3]:
        assert dropped_name not in surviving_names


def test_mmpose_empty_tracks_returns_empty() -> None:
    """No tracks → no results, inference_topdown is never called."""
    with (
        patch("purrai_core.backends.mmpose_pose.init_model") as mock_init,
        patch("purrai_core.backends.mmpose_pose.inference_topdown") as mock_infer,
    ):
        mock_init.return_value = MagicMock()
        estimator = MMPosePoseEstimator(_CFG)
        results = estimator.estimate(np.zeros((240, 320, 3), dtype=np.uint8), [])

    assert results == []
    mock_infer.assert_not_called()
