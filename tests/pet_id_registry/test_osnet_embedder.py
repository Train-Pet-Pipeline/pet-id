from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pet_id_registry.backends.osnet_embedder import OSNetEmbedderAdapter
from pet_id_registry.protocols import Embedder


def _fake_reid(dim: int = 512):
    reid = MagicMock()
    reid.embedding_dim = dim
    reid.device = "cpu"
    reid.model = MagicMock()
    # model(tensor) -> torch tensor with shape (1, dim)
    import torch

    reid.model.return_value = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    return reid


def test_adapter_implements_protocol() -> None:
    reid = _fake_reid()
    adapter = OSNetEmbedderAdapter(reid)
    assert isinstance(adapter, Embedder)
    assert adapter.embedding_dim == 512


def test_adapter_returns_l2_normalized_float32() -> None:
    reid = _fake_reid(dim=8)
    adapter = OSNetEmbedderAdapter(reid)
    crop = np.full((256, 128, 3), 128, dtype=np.uint8)
    out = adapter.embed_crop(crop)
    assert out.dtype == np.float32
    assert out.shape == (8,)
    assert np.isclose(float(np.linalg.norm(out)), 1.0, atol=1e-5)


def test_adapter_rejects_wrong_shape() -> None:
    reid = _fake_reid()
    adapter = OSNetEmbedderAdapter(reid)
    with pytest.raises(ValueError):
        adapter.embed_crop(np.zeros((10, 10), dtype=np.uint8))  # 2D
