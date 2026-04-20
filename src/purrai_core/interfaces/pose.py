"""PoseEstimator protocol — keypoints per tracked animal."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from purrai_core.types import PoseResult, Track


@runtime_checkable
class PoseEstimator(Protocol):
    """Estimate skeletal keypoints for each tracked animal in a frame."""

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> list[PoseResult]: ...
