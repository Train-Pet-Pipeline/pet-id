"""Detector interface contract tests."""

import numpy as np
import pytest

from purrai_core.interfaces.detector import Detector
from purrai_core.types import BBox, Detection


class FakeDetector:
    """Minimal implementation satisfying the Detector protocol."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(bbox=BBox(0, 0, 10, 10), score=0.9, class_id=15, class_name="cat")]


def test_detector_protocol_accepts_fake() -> None:
    """A plain class with a .detect(np.ndarray) -> list[Detection] satisfies Detector."""
    d: Detector = FakeDetector()
    result = d.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(result, list)
    assert isinstance(result[0], Detection)
    assert result[0].class_id == 15


def test_bbox_validates_coords() -> None:
    with pytest.raises(ValueError):
        BBox(x1=10, y1=0, x2=5, y2=10)  # x2 < x1


def test_detection_has_required_fields() -> None:
    det = Detection(bbox=BBox(0, 0, 1, 1), score=0.9, class_id=15, class_name="cat")
    assert det.score == 0.9
    assert det.class_name == "cat"
