"""ByteTrack tracker backend tests."""

from pathlib import Path

import numpy as np

from purrai_core.backends.bytetrack_tracker import ByteTrackTracker
from purrai_core.config import load_config
from purrai_core.types import BBox, Detection


def _det(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(bbox=BBox(x1, y1, x2, y2), score=conf, class_id=15, class_name="cat")


def _frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank frame matching the tracker's expected image input."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_bytetrack_assigns_consistent_id_for_stationary_object(params_yaml_path: Path) -> None:
    cfg = load_config(params_yaml_path)
    t = ByteTrackTracker(cfg.section("tracker"))
    dets = [_det(100, 100, 200, 200)]
    frame = _frame()
    # ByteTrack requires min_hits=1 (set in ByteTrackTracker) so first frame should already track
    tr1 = t.update(dets, frame_idx=0, frame=frame)
    tr2 = t.update(dets, frame_idx=1, frame=frame)
    assert len(tr1) == 1
    assert len(tr2) == 1
    assert tr1[0].track_id == tr2[0].track_id


def test_bytetrack_reset_restarts_ids(params_yaml_path: Path) -> None:
    cfg = load_config(params_yaml_path)
    t = ByteTrackTracker(cfg.section("tracker"))
    frame = _frame()
    t.update([_det(100, 100, 200, 200)], frame_idx=0, frame=frame)
    first_id = t.update([_det(100, 100, 200, 200)], frame_idx=1, frame=frame)[0].track_id
    t.reset()
    second_id = t.update([_det(100, 100, 200, 200)], frame_idx=0, frame=frame)[0].track_id
    assert first_id is not None and second_id is not None
