"""Enrollment pipeline: detect → crop → embed → build PetCard → save."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from purrai_core.interfaces.detector import Detector
from purrai_core.types import BBox

from pet_id_registry.card import PetCard, PetSpecies, PetSex, RegisteredView, compute_pet_id
from pet_id_registry.library import Library
from pet_id_registry.protocols import Embedder

_SCHEMA_VERSION = "1.0.0"


class NoDetectionsError(RuntimeError):
    """Raised when enrollment exhausts input without any usable crop."""


def _clip_bbox(bbox: BBox, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def largest_bbox_crop(frame: np.ndarray, detector: Detector) -> np.ndarray | None:
    """Detect, return the crop of the largest-area detection, or None if none usable."""
    dets = detector.detect(frame)
    if not dets:
        return None

    def area(d) -> float:
        return d.bbox.width * d.bbox.height

    for d in sorted(dets, key=area, reverse=True):
        clipped = _clip_bbox(d.bbox, frame)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        return frame[y1:y2, x1:x2].copy()
    return None


def enroll_photos(
    *,
    image_paths: Sequence[Path],
    name: str,
    species: PetSpecies,
    detector: Detector,
    embedder: Embedder,
    library: Library,
    created_at: datetime,
    cover_photo: Path | None = None,
    force: bool = False,
    metadata: dict | None = None,
) -> PetCard:
    """Build a PetCard from a list of still images and save it to `library`."""
    crops: list[np.ndarray] = []
    for p in image_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        crop = largest_bbox_crop(frame, detector)
        if crop is None:
            continue
        crops.append(crop)
    if not crops:
        raise NoDetectionsError(
            f"no pet detected across {len(image_paths)} input photo(s)"
        )
    embeddings = [embedder.embed_crop(c) for c in crops]
    pet_id = compute_pet_id(embeddings[0])

    views = []
    for i in range(len(crops)):
        vid = f"{i + 1:04d}"
        views.append(RegisteredView(
            view_id=vid,
            crop_uri=f"views/{vid}.jpg",
            embedding_uri=f"views/{vid}.npy",
        ))
    cover_uri = f"{pet_id}/cover.jpg"
    card = PetCard(
        pet_id=pet_id, name=name, species=species,
        created_at=created_at, schema_version=_SCHEMA_VERSION,
        cover_photo_uri=cover_uri, views=views,
        **(metadata or {}),
    )
    cover_crop = cv2.imread(str(cover_photo)) if cover_photo else None
    assets = list(zip(views, crops, embeddings, strict=True))
    library.save(card, view_assets=assets, cover_crop=cover_crop, force=force)
    return card
