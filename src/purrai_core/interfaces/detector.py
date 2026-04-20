"""Detector protocol — single-frame multi-object detection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from purrai_core.types import Detection


@runtime_checkable
class Detector(Protocol):
    """Detect objects in a single BGR frame (H, W, 3) uint8."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
