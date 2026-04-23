"""Structural protocols for detector / embedder dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Embed a cropped pet image into an L2-normalized float32 vector."""

    embedding_dim: int

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Return a (embedding_dim,) float32 L2-normalized embedding for the BGR crop."""
        ...
