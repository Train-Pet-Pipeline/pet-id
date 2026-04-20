"""ReidEncoder contract tests."""

import numpy as np

from purrai_core.interfaces.reid import ReidEncoder
from purrai_core.types import BBox, ReidEmbedding, Track


class FakeReid:
    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]:
        return [ReidEmbedding(track_id=t.track_id, vector=(0.1,) * 512) for t in tracks]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        return gallery[0].track_id if gallery else None


def test_reid_protocol() -> None:
    r: ReidEncoder = FakeReid()
    tracks = [Track(track_id=1, bbox=BBox(0, 0, 1, 1), score=0.9, class_id=15, class_name="cat")]
    emb = r.encode(np.zeros((64, 64, 3), dtype=np.uint8), tracks)
    assert len(emb[0].vector) == 512
    assert r.match_identity(emb[0], emb) == 1
