from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pet_id_registry.card import PetCard, PetSpecies, RegisteredView
from pet_id_registry.library import Library, PetAlreadyExistsError


def _make_card(pet_id: str = "abcd1234", name: str = "Mimi") -> PetCard:
    return PetCard(
        pet_id=pet_id,
        name=name,
        species=PetSpecies.cat,
        created_at=datetime(2026, 4, 21, 10, 0, 0),
        schema_version="1.0.0",
        cover_photo_uri=f"{pet_id}/cover.jpg",
        views=[RegisteredView(view_id="0001",
                              crop_uri=f"{pet_id}/views/0001.jpg",
                              embedding_uri=f"{pet_id}/views/0001.npy")],
    )


def _fake_view_payload():
    crop = np.full((256, 128, 3), 77, dtype=np.uint8)
    emb = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    return crop, emb


def test_save_creates_layout(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()
    lib.save(card, view_assets=[(card.views[0], crop, emb)])

    root = tmp_path / "abcd1234"
    assert (root / "card.json").exists()
    assert (root / "cover.jpg").exists()
    assert (root / "views" / "0001.jpg").exists()
    assert (root / "views" / "0001.npy").exists()
    assert (tmp_path / "index.json").exists()


def test_save_refuses_existing_without_force(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()
    lib.save(card, view_assets=[(card.views[0], crop, emb)])
    with pytest.raises(PetAlreadyExistsError):
        lib.save(card, view_assets=[(card.views[0], crop, emb)])


def test_save_overwrites_with_force(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()
    lib.save(card, view_assets=[(card.views[0], crop, emb)])
    lib.save(card.model_copy(update={"name": "Mimi2"}),
             view_assets=[(card.views[0], crop, emb)], force=True)
    reloaded = lib.load("abcd1234")
    assert reloaded.name == "Mimi2"


def test_load_round_trip(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()
    lib.save(card, view_assets=[(card.views[0], crop, emb)])
    assert lib.load("abcd1234") == card


def test_list(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    for pid, name in [("aa111111", "A"), ("bb222222", "B")]:
        card = _make_card(pid, name)
        crop, emb = _fake_view_payload()
        lib.save(card, view_assets=[(card.views[0], crop, emb)])
    entries = lib.list()
    assert {e.pet_id for e in entries} == {"aa111111", "bb222222"}


def test_delete(tmp_path: Path) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()
    lib.save(card, view_assets=[(card.views[0], crop, emb)])
    lib.delete("abcd1234")
    assert not (tmp_path / "abcd1234").exists()
    assert lib.list() == []


def test_atomic_save_no_partial_on_failure(tmp_path: Path, monkeypatch) -> None:
    lib = Library(tmp_path)
    card = _make_card()
    crop, emb = _fake_view_payload()

    # Break writing *after* the temp dir is created but before rename.
    def boom(src, dst):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        lib.save(card, view_assets=[(card.views[0], crop, emb)])
    assert not (tmp_path / "abcd1234").exists()
