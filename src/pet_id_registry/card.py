"""PetCard Pydantic models + content-addressed pet_id hashing."""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PetSpecies(StrEnum):
    cat = "cat"
    dog = "dog"
    other = "other"


class PetSex(StrEnum):
    male = "male"
    female = "female"
    unknown = "unknown"


class RegisteredView(BaseModel):
    model_config = ConfigDict(frozen=True)

    view_id: str
    pose_hint: str | None = None
    crop_uri: str
    embedding_uri: str


class PetCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pet_id: str
    name: str
    species: PetSpecies
    created_at: datetime
    schema_version: str
    cover_photo_uri: str
    views: list[RegisteredView] = Field(min_length=1)

    breed: str | None = None
    sex: PetSex | None = None
    birthdate: date | None = None
    weight_kg: float | None = None
    markings: str | None = None
    owner_name: str | None = None
    medical_notes: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


def compute_pet_id(embedding: np.ndarray) -> str:
    """Return an 8-hex-char content-addressed id.

    Caller must pass an L2-normalized embedding (norm≈1). The bytes are
    canonicalized to little-endian float32 contiguous before hashing so the
    id is identical across hosts and dtype choices.
    """
    norm = float(np.linalg.norm(embedding))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError(
            f"compute_pet_id requires an L2-normalized embedding (got norm={norm:.4f})"
        )
    arr = np.ascontiguousarray(embedding.astype("<f4"))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:8]
