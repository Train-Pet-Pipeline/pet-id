from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from pet_id_registry.card import (
    PetCard,
    PetSpecies,
    RegisteredView,
    compute_pet_id,
)


def _view(view_id: str = "0001") -> RegisteredView:
    return RegisteredView(
        view_id=view_id,
        pose_hint=None,
        crop_uri=f"views/{view_id}.jpg",
        embedding_uri=f"views/{view_id}.npy",
    )


def _card(**over):
    defaults = dict(
        pet_id="abcd1234",
        name="Mimi",
        species=PetSpecies.cat,
        created_at=datetime(2026, 4, 21, 10, 0, 0),
        schema_version="1.0.0",
        cover_photo_uri="cover.jpg",
        views=[_view()],
    )
    defaults.update(over)
    return PetCard(**defaults)


def test_card_minimal_valid() -> None:
    c = _card()
    assert c.pet_id == "abcd1234"
    assert c.species is PetSpecies.cat
    assert len(c.views) == 1
    assert c.extra == {}


def test_card_requires_at_least_one_view() -> None:
    with pytest.raises(ValidationError):
        _card(views=[])


def test_card_rejects_unknown_species() -> None:
    with pytest.raises(ValidationError):
        _card(species="tiger")


def test_card_round_trip_json() -> None:
    c = _card()
    data = c.model_dump_json()
    c2 = PetCard.model_validate_json(data)
    assert c2 == c
    # pose_hint must survive as None
    assert c2.views[0].pose_hint is None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_compute_pet_id_deterministic_across_dtypes() -> None:
    v64 = _unit(np.arange(1, 9, dtype=np.float64))
    v32 = v64.astype(np.float32)
    assert compute_pet_id(v64) == compute_pet_id(v32)


def test_compute_pet_id_length_8_hex() -> None:
    v = _unit(np.arange(1, 9, dtype=np.float32))
    pid = compute_pet_id(v)
    assert len(pid) == 8
    int(pid, 16)  # must be valid hex


def test_compute_pet_id_requires_normalized() -> None:
    import pytest

    with pytest.raises(ValueError, match="L2-normalized"):
        compute_pet_id(np.array([3.0, 4.0], dtype=np.float32))  # norm=5
