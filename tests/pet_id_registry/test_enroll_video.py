from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pet_id_registry.card import PetSpecies
from pet_id_registry.enroll import enroll_video
from pet_id_registry.library import Library
from purrai_core.types import BBox, Detection


def _write_video(
    path: Path, *, frames: int, fps: int = 30, size: tuple[int, int] = (320, 240)
) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), (i * 5) % 255, dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path


class _Det:
    def __init__(self, det_every: int = 1) -> None:
        self._det_every = det_every
        self._called = 0

    def detect(self, frame):
        self._called += 1
        if self._called % self._det_every != 0:
            return []
        return [Detection(bbox=BBox(10, 10, 200, 200), score=0.9, class_id=15, class_name="cat")]


class _Emb:
    embedding_dim = 8

    def embed_crop(self, crop):
        v = np.random.default_rng(int(crop.sum())).standard_normal(8).astype(np.float32)
        return v / np.linalg.norm(v)


def test_video_fps_sampling_and_cap(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.mp4", frames=60, fps=30)
    lib = Library(tmp_path / "gal")
    card = enroll_video(
        video_path=video,
        name="M",
        species=PetSpecies.cat,
        detector=_Det(),
        embedder=_Emb(),
        library=lib,
        fps_sample=2,
        max_views=3,
        created_at=datetime(2026, 4, 21),
    )
    # 60 frames at src_fps=30 → 2 seconds → fps_sample=2 yields ~4 samples.
    # max_views=3 caps it.
    assert len(card.views) == 3


def test_video_filters_frames_without_detection(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.mp4", frames=60, fps=30)
    lib = Library(tmp_path / "gal")
    # every 2nd sampled frame returns no detection
    card = enroll_video(
        video_path=video,
        name="M",
        species=PetSpecies.cat,
        detector=_Det(det_every=2),
        embedder=_Emb(),
        library=lib,
        fps_sample=2,
        max_views=10,
        created_at=datetime(2026, 4, 21),
    )
    # some frames skipped, fewer than requested views
    assert 1 <= len(card.views) <= 4
