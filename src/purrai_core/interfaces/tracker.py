"""Tracker protocol — stateful multi-object tracking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from purrai_core.types import Detection, Track


@runtime_checkable
class Tracker(Protocol):
    """Track animals across frames, maintaining persistent IDs."""

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[Track]: ...
    def reset(self) -> None: ...
