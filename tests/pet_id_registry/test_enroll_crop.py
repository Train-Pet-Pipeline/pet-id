from __future__ import annotations

import numpy as np

from pet_id_registry.enroll import largest_bbox_crop
from purrai_core.types import BBox, Detection


class _FakeDetector:
    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._dets


def _det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> Detection:
    return Detection(bbox=BBox(x1, y1, x2, y2), score=score, class_id=15, class_name="cat")


def test_largest_bbox_picks_biggest() -> None:
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    dets = [_det(0, 0, 10, 10), _det(20, 20, 90, 90), _det(40, 40, 60, 60)]
    crop = largest_bbox_crop(frame, _FakeDetector(dets))
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_returns_none_when_no_detection() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    assert largest_bbox_crop(frame, _FakeDetector([])) is None


def test_clips_bbox_to_frame_bounds() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    dets = [_det(-5, -5, 55, 55)]  # overflows frame
    crop = largest_bbox_crop(frame, _FakeDetector(dets))
    assert crop is not None
    assert crop.shape[:2] == (50, 50)


def test_returns_none_when_bbox_is_empty_after_clip() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    dets = [_det(60, 60, 80, 80)]  # fully outside frame
    assert largest_bbox_crop(frame, _FakeDetector(dets)) is None
