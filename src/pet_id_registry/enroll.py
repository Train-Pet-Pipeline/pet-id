"""Enrollment pipeline: detect → crop → embed → build PetCard → save."""
from __future__ import annotations

import numpy as np

from purrai_core.interfaces.detector import Detector
from purrai_core.types import BBox


def _clip_bbox(bbox: BBox, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def largest_bbox_crop(frame: np.ndarray, detector: Detector) -> np.ndarray | None:
    """Detect, return the crop of the largest-area detection, or None if none usable."""
    dets = detector.detect(frame)
    if not dets:
        return None

    def area(d) -> float:
        return d.bbox.width * d.bbox.height

    for d in sorted(dets, key=area, reverse=True):
        clipped = _clip_bbox(d.bbox, frame)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        return frame[y1:y2, x1:x2].copy()
    return None
