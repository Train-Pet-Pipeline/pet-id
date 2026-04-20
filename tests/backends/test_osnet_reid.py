"""OSNet Re-ID backend tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

torchreid = pytest.importorskip("torchreid")  # skip collection if torchreid not installed

from purrai_core.backends.osnet_reid import OSNetReid  # noqa: E402
from purrai_core.config import load_config  # noqa: E402
from purrai_core.types import BBox, ReidEmbedding, Track  # noqa: E402


def _cpu_reid_cfg(params_yaml_path: Path) -> dict:  # type: ignore[type-arg]
    """Return reid config section with device forced to cpu for macOS CI."""
    cfg = load_config(params_yaml_path).section("reid")
    cfg["device"] = "cpu"
    return cfg


def test_osnet_encode_returns_embedding_per_track(params_yaml_path: Path) -> None:
    with patch("purrai_core.backends.osnet_reid.build_model") as mock_build:
        fake_net = MagicMock(return_value=torch.ones(1, 512) * 0.1)
        # build_model(...).to(device).eval() must resolve to fake_net so that
        # calling self.model(tensor) returns the fixed tensor.
        mock_build.return_value.to.return_value.eval.return_value = fake_net
        r = OSNetReid(_cpu_reid_cfg(params_yaml_path))
        tracks = [
            Track(track_id=1, bbox=BBox(0, 0, 64, 64), score=0.9, class_id=15, class_name="cat")
        ]
        embs = r.encode(np.zeros((128, 128, 3), dtype=np.uint8), tracks)
        assert len(embs) == 1
        assert len(embs[0].vector) == 512
        assert embs[0].track_id == 1


def test_osnet_match_identity_above_threshold(params_yaml_path: Path) -> None:
    with patch("purrai_core.backends.osnet_reid.build_model"):
        r = OSNetReid(_cpu_reid_cfg(params_yaml_path))
        v = tuple(float(x) for x in np.ones(512) / np.sqrt(512))
        query = ReidEmbedding(track_id=1, vector=v)
        gallery = [ReidEmbedding(track_id=99, vector=v)]
        assert r.match_identity(query, gallery) == 99
