"""NarrativeGenerator protocol — VLM-backed natural-language summary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from purrai_core.types import NarrativeOutput, Track


@runtime_checkable
class NarrativeGenerator(Protocol):
    """Generate a natural-language description of animal behavior from a clip of frames."""

    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput: ...
