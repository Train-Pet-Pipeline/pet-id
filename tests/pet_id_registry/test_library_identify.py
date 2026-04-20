from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from pet_id_registry.card import PetCard, PetSpecies, RegisteredView
from pet_id_registry.library import IdentifyResult, Library


def _register(lib: Library, pet_id: str, name: str, vecs: list[np.ndarray]) -> None:
    views = []
    assets = []
    for i, v in enumerate(vecs, start=1):
        vid = f"{i:04d}"
        rv = RegisteredView(view_id=vid,
                            crop_uri=f"views/{vid}.jpg",
                            embedding_uri=f"views/{vid}.npy")
        views.append(rv)
        crop = np.full((256, 128, 3), 30, dtype=np.uint8)
        assets.append((rv, crop, v.astype(np.float32)))
    card = PetCard(pet_id=pet_id, name=name, species=PetSpecies.cat,
                   created_at=datetime(2026, 4, 21),
                   schema_version="1.0.0",
                   cover_photo_uri=f"{pet_id}/cover.jpg", views=views)
    lib.save(card, view_assets=assets)


def _unit(*axes: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for a in axes:
        v[a] = 1.0
    return v / np.linalg.norm(v)


def test_identify_hits_best_match(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    _register(lib, "aaaa0001", "A", [_unit(0)])
    _register(lib, "bbbb0002", "B", [_unit(1)])
    q = _unit(0)
    result = lib.identify(q, threshold=0.55)
    assert isinstance(result, IdentifyResult)
    assert result.pet_id == "aaaa0001"
    assert result.name == "A"
    assert result.score > 0.99


def test_identify_multi_view_uses_max(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    _register(lib, "cccc0003", "C", [_unit(0), _unit(1), _unit(2)])
    q = _unit(2)
    result = lib.identify(q, threshold=0.55)
    assert result is not None and result.pet_id == "cccc0003"


def test_identify_below_threshold_returns_unknown(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    _register(lib, "dddd0004", "D", [_unit(0)])
    q = _unit(1)  # orthogonal → cos=0
    result = lib.identify(q, threshold=0.55)
    assert result is None


def test_identify_empty_library(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    assert lib.identify(_unit(0), threshold=0.55) is None
