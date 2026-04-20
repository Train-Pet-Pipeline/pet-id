from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from pet_id_registry.card import PetSpecies
from pet_id_registry.enroll import NoDetectionsError, enroll_photos
from pet_id_registry.library import Library
from purrai_core.types import BBox, Detection


class _FakeDetector:
    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = list(per_frame)
        self.call_count = 0

    def detect(self, frame: np.ndarray):
        i = min(self.call_count, len(self._per_frame) - 1)
        self.call_count += 1
        return self._per_frame[i]


class _FakeEmbedder:
    embedding_dim = 8

    def __init__(self) -> None:
        self.call_count = 0

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        self.call_count += 1
        v = np.zeros(8, dtype=np.float32)
        v[self.call_count % 8] = 1.0
        return v


def _write_jpg(path: Path, color: int = 120) -> Path:
    img = np.full((200, 200, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _det(x1, y1, x2, y2) -> Detection:
    return Detection(bbox=BBox(x1, y1, x2, y2), score=0.9,
                     class_id=15, class_name="cat")


def test_enroll_three_photos_gives_three_views(tmp_path: Path) -> None:
    lib = Library(tmp_path / "gal")
    files = [_write_jpg(tmp_path / f"p{i}.jpg", 50 + 20 * i) for i in range(3)]
    det = _FakeDetector([[_det(10, 10, 150, 180)]] * 3)
    emb = _FakeEmbedder()
    card = enroll_photos(
        image_paths=files, name="Mimi", species=PetSpecies.cat,
        detector=det, embedder=emb, library=lib,
        created_at=datetime(2026, 4, 21),
    )
    assert len(card.views) == 3
    reloaded = lib.load(card.pet_id)
    assert reloaded == card


def test_enroll_skips_frames_without_detection(tmp_path: Path) -> None:
    lib = Library(tmp_path / "gal")
    files = [_write_jpg(tmp_path / f"p{i}.jpg") for i in range(3)]
    det = _FakeDetector([[_det(0, 0, 100, 100)], [], [_det(0, 0, 100, 100)]])
    emb = _FakeEmbedder()
    card = enroll_photos(
        image_paths=files, name="M", species=PetSpecies.cat,
        detector=det, embedder=emb, library=lib,
        created_at=datetime(2026, 4, 21),
    )
    assert len(card.views) == 2
    assert emb.call_count == 2


def test_enroll_raises_when_all_frames_empty(tmp_path: Path) -> None:
    lib = Library(tmp_path / "gal")
    files = [_write_jpg(tmp_path / f"p{i}.jpg") for i in range(2)]
    det = _FakeDetector([[], []])
    with pytest.raises(NoDetectionsError):
        enroll_photos(
            image_paths=files, name="M", species=PetSpecies.cat,
            detector=det, embedder=_FakeEmbedder(), library=lib,
            created_at=datetime(2026, 4, 21),
        )
