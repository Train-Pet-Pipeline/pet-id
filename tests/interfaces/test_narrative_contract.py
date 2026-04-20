"""NarrativeGenerator contract tests."""

import numpy as np

from purrai_core.interfaces.narrative import NarrativeGenerator
from purrai_core.types import BBox, NarrativeOutput, Track


class FakeNarrative:
    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput:
        return NarrativeOutput(text="猫正在进食", confidence=0.9)


def test_narrative_protocol() -> None:
    g: NarrativeGenerator = FakeNarrative()
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    tracks_history = [
        [Track(track_id=1, bbox=BBox(0, 0, 1, 1), score=0.9, class_id=15, class_name="cat")]
        for _ in range(3)
    ]
    out = g.describe(frames, tracks_history)
    assert isinstance(out, NarrativeOutput)
    assert out.text == "猫正在进食"
    assert out.confidence == 0.9
