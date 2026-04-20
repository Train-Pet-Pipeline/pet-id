"""Adapter exposing purrai_core.OSNetReid as an Embedder protocol impl."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

_INPUT_H = 256
_INPUT_W = 128


class OSNetEmbedderAdapter:
    """Wrap a loaded OSNetReid instance so it satisfies the Embedder protocol."""

    def __init__(self, reid: Any) -> None:
        self._reid = reid
        self.embedding_dim = int(reid.embedding_dim)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize BGR crop to 128x256, run OSNet, return L2-normalized float32."""
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"crop must be HxWx3 BGR, got shape {crop.shape}")
        resized = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self._reid.device)
            / 255.0
        )
        with torch.no_grad():
            feats = self._reid.model(tensor)
        vec = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()[0]
        return vec.astype(np.float32, copy=False)
