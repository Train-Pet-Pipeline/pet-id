"""OSNet Re-ID backend via torchreid."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torchreid.reid.models import build_model

from purrai_core.interfaces.reid import ReidEncoder
from purrai_core.types import ReidEmbedding, Track


class OSNetReid(ReidEncoder):
    """Re-ID backend using OSNet from torchreid."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialise OSNet model from config section."""
        self.cfg = cfg
        self.device = str(cfg["device"])
        self.embedding_dim = int(cfg["embedding_dim"])
        self.similarity_threshold = float(cfg["similarity_threshold"])
        self.model = (
            build_model(name=str(cfg["model_name"]), num_classes=1, pretrained=True)
            .to(self.device)
            .eval()
        )

    def encode(self, frame: np.ndarray, tracks: list[Track]) -> list[ReidEmbedding]:
        """Crop each tracked bbox, run OSNet, return normalised embeddings."""
        if not tracks:
            return []
        crops: list[np.ndarray] = []
        for t in tracks:
            x1, y1 = int(max(0, t.bbox.x1)), int(max(0, t.bbox.y1))
            x2, y2 = int(min(frame.shape[1], t.bbox.x2)), int(min(frame.shape[0], t.bbox.y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((128, 64, 3), dtype=np.uint8))
            else:
                crops.append(frame[y1:y2, x1:x2])
        batch = np.stack([cv2.resize(c, (128, 256)) for c in crops], axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        with torch.no_grad():
            feats = self.model(tensor)
        feats_norm = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        return [
            ReidEmbedding(track_id=t.track_id, vector=tuple(float(x) for x in feats_norm[i]))
            for i, t in enumerate(tracks)
        ]

    def match_identity(self, embedding: ReidEmbedding, gallery: list[ReidEmbedding]) -> int | None:
        """Return the gallery track_id with highest cosine similarity above threshold."""
        if not gallery:
            return None
        q = np.array(embedding.vector, dtype=np.float32)
        best_id: int | None = None
        best_sim = -1.0
        for g in gallery:
            gv = np.array(g.vector, dtype=np.float32)
            sim = float(np.dot(q, gv) / (np.linalg.norm(q) * np.linalg.norm(gv) + 1e-9))
            if sim > best_sim:
                best_sim, best_id = sim, g.track_id
        return best_id if best_sim >= self.similarity_threshold else None
