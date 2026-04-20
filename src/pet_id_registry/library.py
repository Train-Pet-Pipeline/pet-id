"""Flat-file PetCard gallery: save / load / list / delete / identify."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pet_id_registry.card import PetCard, RegisteredView

_INDEX_FILE = "index.json"
_TMP_DIR = ".tmp"


class PetAlreadyExistsError(RuntimeError):
    pass


class PetNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class LibraryEntry:
    pet_id: str
    name: str
    species: str
    view_count: int
    created_at: str


@dataclass(frozen=True)
class IdentifyResult:
    pet_id: str
    name: str
    score: float
    view_id: str


def _iter_view_embeddings(root: Path, pet_id: str) -> Iterable[tuple[str, np.ndarray]]:
    vdir = root / pet_id / "views"
    for npy in sorted(vdir.glob("*.npy")):
        yield npy.stem, np.load(npy)


class Library:
    """Local filesystem PetCard gallery."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _pet_dir(self, pet_id: str) -> Path:
        return self.root / pet_id

    def save(
        self,
        card: PetCard,
        view_assets: Iterable[tuple[RegisteredView, np.ndarray, np.ndarray]],
        *,
        cover_crop: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        target = self._pet_dir(card.pet_id)
        if target.exists() and not force:
            raise PetAlreadyExistsError(card.pet_id)

        tmp_parent = self.root / _TMP_DIR
        tmp_parent.mkdir(exist_ok=True)
        tmp_dir = tmp_parent / f"{card.pet_id}-{uuid.uuid4().hex[:8]}"
        views_dir = tmp_dir / "views"
        views_dir.mkdir(parents=True)

        first_crop: np.ndarray | None = None
        for view, crop, emb in view_assets:
            cv2.imwrite(str(views_dir / Path(view.crop_uri).name), crop)
            np.save(views_dir / Path(view.embedding_uri).name, emb.astype(np.float32))
            if first_crop is None:
                first_crop = crop

        cover = cover_crop if cover_crop is not None else first_crop
        if cover is not None:
            cv2.imwrite(str(tmp_dir / "cover.jpg"), cover)
        (tmp_dir / "card.json").write_text(card.model_dump_json(indent=2))

        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp_dir, target)
        self._rebuild_index()

    def load(self, pet_id: str) -> PetCard:
        card_path = self._pet_dir(pet_id) / "card.json"
        if not card_path.exists():
            raise PetNotFoundError(pet_id)
        return PetCard.model_validate_json(card_path.read_text())

    def list(self) -> list[LibraryEntry]:
        entries: list[LibraryEntry] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            card_path = child / "card.json"
            if not card_path.exists():
                continue
            card = PetCard.model_validate_json(card_path.read_text())
            entries.append(
                LibraryEntry(
                    pet_id=card.pet_id,
                    name=card.name,
                    species=card.species.value,
                    view_count=len(card.views),
                    created_at=card.created_at.isoformat(),
                )
            )
        return entries

    def delete(self, pet_id: str) -> None:
        target = self._pet_dir(pet_id)
        if not target.exists():
            raise PetNotFoundError(pet_id)
        shutil.rmtree(target)
        self._rebuild_index()

    def identify(self, query: np.ndarray, *, threshold: float) -> IdentifyResult | None:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        best: IdentifyResult | None = None
        for entry in self.list():
            for view_id, vec in _iter_view_embeddings(self.root, entry.pet_id):
                score = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9))
                if best is None or score > best.score:
                    best = IdentifyResult(
                        pet_id=entry.pet_id, name=entry.name, score=score, view_id=view_id
                    )
        if best is None or best.score < threshold:
            return None
        return best

    def _rebuild_index(self) -> None:
        index = {e.pet_id: e.name for e in self.list()}
        (self.root / _INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True))
