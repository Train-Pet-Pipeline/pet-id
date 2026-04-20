"""ReidEncoder protocol — identity embedding + matching."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from purrai_core.types import ReidEmbedding, Track


@runtime_checkable
class ReidEncoder(Protocol):
    """Encode animal appearance into a fixed-size embedding and match against a gallery."""

    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]: ...
    def match_identity(
        self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]
    ) -> int | None: ...
