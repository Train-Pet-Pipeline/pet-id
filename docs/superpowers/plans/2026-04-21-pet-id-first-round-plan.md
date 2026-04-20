# pet-id first-round implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a thin PetCard library + `petid` CLI so users can enroll a pet from photos/video and identify a pet by image against the enrolled gallery.

**Architecture:** New sibling package `src/pet_id_registry/` reuses `purrai_core`'s yolov10 detector and OSNet ReID via adapter classes matching local `Detector` / `Embedder` protocols. Gallery is flat-file on disk (`<library_root>/<pet_id>/card.json + views/*.jpg + *.npy`). CLI is a `click` app with 5 subcommands. All numerics from `params.yaml`.

**Tech Stack:** Python 3.11, Pydantic v2, click, numpy, OpenCV, pytest + CliRunner. No new heavy deps.

**Spec:** `docs/superpowers/specs/2026-04-21-pet-id-first-round-design.md`

**Branch:** `feature/pet-id-first-round` (already created, spec commits already landed).

---

## Task 1: Scaffold `pet_id_registry` package and wire build

**Files:**
- Create: `src/pet_id_registry/__init__.py`
- Create: `src/pet_id_registry/protocols.py`
- Create: `src/pet_id_registry/card.py` (empty module stub)
- Create: `src/pet_id_registry/library.py` (empty module stub)
- Create: `src/pet_id_registry/enroll.py` (empty module stub)
- Create: `src/pet_id_registry/cli.py` (empty module stub)
- Create: `tests/pet_id_registry/__init__.py`
- Create: `tests/pet_id_registry/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create package and test directories with empty stubs**

Every `.py` stub begins with:
```python
"""<one-line purpose>."""
from __future__ import annotations
```

`src/pet_id_registry/__init__.py`:
```python
"""pet-id registry: enrollment, gallery, identify."""
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 2: Write `protocols.py` with `Embedder` protocol**

`src/pet_id_registry/protocols.py`:
```python
"""Structural protocols for detector / embedder dependencies."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Embed a cropped pet image into an L2-normalized float32 vector."""

    embedding_dim: int

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Return a (embedding_dim,) float32 L2-normalized embedding for the BGR crop."""
        ...
```

We reuse `purrai_core.interfaces.detector.Detector` for detection — no redefinition.

- [ ] **Step 3: Update pyproject.toml**

Under `[project].dependencies`, append `"click>=8.1"` (keep existing entries, sorted or appended at end).

Replace the `[tool.setuptools.packages.find]` block:
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["purrai_core", "purrai_core.*", "pet_id_registry", "pet_id_registry.*"]
```

Add a new `[project.scripts]` block (just before `[tool.setuptools.packages.find]`):
```toml
[project.scripts]
petid = "pet_id_registry.cli:main"
```

- [ ] **Step 4: Reinstall in the shared conda env so the entry point is live**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pip install -e . --no-deps --quiet
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pip install click --quiet
```

Expected: no errors.

- [ ] **Step 5: Verify the package imports + CLI entry point exists**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -c "import pet_id_registry; print(pet_id_registry.__version__)"
```
Expected: `0.1.0`

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/which petid
```
Expected: prints the `petid` executable path under the env's `bin/`.

- [ ] **Step 6: Commit**

```bash
git add src/pet_id_registry tests/pet_id_registry pyproject.toml
git commit -m "feat(pet-id): scaffold pet_id_registry package + petid entry point"
```

---

## Task 2: `PetCard` and `RegisteredView` Pydantic models + `pet_id` hashing helper

**Files:**
- Modify: `src/pet_id_registry/card.py`
- Test: `tests/pet_id_registry/test_card.py`

- [ ] **Step 1: Write failing tests**

`tests/pet_id_registry/test_card.py`:
```python
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


def test_compute_pet_id_deterministic_across_dtypes() -> None:
    v64 = np.arange(8, dtype=np.float64)
    v32 = v64.astype(np.float32)
    assert compute_pet_id(v64) == compute_pet_id(v32)


def test_compute_pet_id_length_8_hex() -> None:
    v = np.arange(8, dtype=np.float32)
    pid = compute_pet_id(v)
    assert len(pid) == 8
    int(pid, 16)  # must be valid hex
```

- [ ] **Step 2: Run — verify RED**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_card.py -v
```
Expected: all fail with ImportError.

- [ ] **Step 3: Implement `card.py`**

```python
"""PetCard Pydantic models + content-addressed pet_id hashing."""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PetSpecies(str, Enum):
    cat = "cat"
    dog = "dog"
    other = "other"


class PetSex(str, Enum):
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

    Embedding is normalized to little-endian float32 contiguous bytes before
    hashing to ensure the id is identical across hosts and dtype choices.
    """
    arr = np.ascontiguousarray(embedding.astype("<f4"))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:8]
```

- [ ] **Step 4: Run — verify GREEN**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_card.py -v
```
Expected: 6/6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/card.py tests/pet_id_registry/test_card.py
git commit -m "feat(pet-id): PetCard/RegisteredView models + pet_id content hash"
```

---

## Task 3: OSNet `Embedder` adapter

**Files:**
- Create: `src/pet_id_registry/backends/__init__.py`
- Create: `src/pet_id_registry/backends/osnet_embedder.py`
- Test: `tests/pet_id_registry/test_osnet_embedder.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pet_id_registry.backends.osnet_embedder import OSNetEmbedderAdapter
from pet_id_registry.protocols import Embedder


def _fake_reid(dim: int = 512):
    reid = MagicMock()
    reid.embedding_dim = dim
    reid.device = "cpu"
    reid.model = MagicMock()
    # model(tensor) -> torch tensor with shape (1, dim)
    import torch
    reid.model.return_value = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    return reid


def test_adapter_implements_protocol() -> None:
    reid = _fake_reid()
    adapter = OSNetEmbedderAdapter(reid)
    assert isinstance(adapter, Embedder)
    assert adapter.embedding_dim == 512


def test_adapter_returns_l2_normalized_float32() -> None:
    reid = _fake_reid(dim=8)
    adapter = OSNetEmbedderAdapter(reid)
    crop = np.full((256, 128, 3), 128, dtype=np.uint8)
    out = adapter.embed_crop(crop)
    assert out.dtype == np.float32
    assert out.shape == (8,)
    assert np.isclose(float(np.linalg.norm(out)), 1.0, atol=1e-5)


def test_adapter_rejects_wrong_shape() -> None:
    reid = _fake_reid()
    adapter = OSNetEmbedderAdapter(reid)
    with pytest.raises(ValueError):
        adapter.embed_crop(np.zeros((10, 10), dtype=np.uint8))  # 2D
```

- [ ] **Step 2: Run — verify RED**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_osnet_embedder.py -v
```
Expected: all fail with ImportError.

- [ ] **Step 3: Implement adapter**

`src/pet_id_registry/backends/__init__.py`:
```python
"""Backend adapters bridging purrai_core backends to pet_id_registry protocols."""
from __future__ import annotations
```

`src/pet_id_registry/backends/osnet_embedder.py`:
```python
"""Adapter exposing purrai_core.OSNetReid as an Embedder protocol impl."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

_INPUT_H = 256
_INPUT_W = 128


class OSNetEmbedderAdapter:
    """Wrap a loaded OSNetReid instance so it satisfies the Embedder protocol."""

    def __init__(self, reid: Any) -> None:
        self._reid = reid
        self.embedding_dim = int(reid.embedding_dim)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize BGR crop to 128x256, run OSNet, return L2-normalized float32."""
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"crop must be HxWx3 BGR, got shape {crop.shape}")
        resized = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self._reid.device)
            / 255.0
        )
        with torch.no_grad():
            feats = self._reid.model(tensor)
        vec = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()[0]
        return vec.astype(np.float32, copy=False)
```

- [ ] **Step 4: Run — verify GREEN**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_osnet_embedder.py -v
```
Expected: 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/backends tests/pet_id_registry/test_osnet_embedder.py
git commit -m "feat(pet-id): OSNetEmbedderAdapter implementing Embedder protocol"
```

---

## Task 4: `Library` CRUD (save/load/list/delete) with atomic writes

**Files:**
- Modify: `src/pet_id_registry/library.py`
- Test: `tests/pet_id_registry/test_library_crud.py`

- [ ] **Step 1: Write failing tests**

```python
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
    orig_replace = __import__("os").replace

    def boom(src, dst):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        lib.save(card, view_assets=[(card.views[0], crop, emb)])
    assert not (tmp_path / "abcd1234").exists()
```

- [ ] **Step 2: Run — verify RED**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_library_crud.py -v
```
Expected: all fail with ImportError.

- [ ] **Step 3: Implement `library.py` CRUD portion**

```python
"""Flat-file PetCard gallery: save / load / list / delete / identify."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    def _rebuild_index(self) -> None:
        index = {e.pet_id: e.name for e in self.list()}
        (self.root / _INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True))
```

- [ ] **Step 4: Run — verify GREEN**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_library_crud.py -v
```
Expected: 7/7 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/library.py tests/pet_id_registry/test_library_crud.py
git commit -m "feat(pet-id): Library CRUD with atomic save"
```

---

## Task 5: `Library.identify` — max-cosine search

**Files:**
- Modify: `src/pet_id_registry/library.py`
- Test: `tests/pet_id_registry/test_library_identify.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run — verify RED**

Expected: all fail (`IdentifyResult` / `Library.identify` missing).

- [ ] **Step 3: Extend `library.py` with identify**

Append to `library.py`:
```python
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


# Add to the Library class body:
# ---
# def identify(self, query: np.ndarray, *, threshold: float) -> IdentifyResult | None:
#     q = np.asarray(query, dtype=np.float32).reshape(-1)
#     best: IdentifyResult | None = None
#     for entry in self.list():
#         for view_id, vec in _iter_view_embeddings(self.root, entry.pet_id):
#             score = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9))
#             if best is None or score > best.score:
#                 best = IdentifyResult(
#                     pet_id=entry.pet_id, name=entry.name, score=score, view_id=view_id
#                 )
#     if best is None or best.score < threshold:
#         return None
#     return best
```

Place `identify` as a method on the `Library` class (not a free function).

- [ ] **Step 4: Run — verify GREEN**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_library_identify.py -v
```
Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/library.py tests/pet_id_registry/test_library_identify.py
git commit -m "feat(pet-id): Library.identify max-cosine search"
```

---

## Task 6: Enrollment primitives — `largest_bbox_crop` helper

**Files:**
- Create: `src/pet_id_registry/enroll.py` (real content)
- Test: `tests/pet_id_registry/test_enroll_crop.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import numpy as np

from purrai_core.types import BBox, Detection
from pet_id_registry.enroll import largest_bbox_crop


class _FakeDetector:
    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._dets


def _det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> Detection:
    return Detection(bbox=BBox(x1, y1, x2, y2), score=score, class_id=15, class_name="cat")


def test_largest_bbox_picks_biggest() -> None:
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    dets = [_det(0, 0, 10, 10), _det(20, 20, 90, 90), _det(40, 40, 60, 60)]
    crop = largest_bbox_crop(frame, _FakeDetector(dets))
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_returns_none_when_no_detection() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    assert largest_bbox_crop(frame, _FakeDetector([])) is None


def test_clips_bbox_to_frame_bounds() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    dets = [_det(-5, -5, 55, 55)]  # overflows frame
    crop = largest_bbox_crop(frame, _FakeDetector(dets))
    assert crop is not None
    assert crop.shape[:2] == (50, 50)


def test_returns_none_when_bbox_is_empty_after_clip() -> None:
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    dets = [_det(60, 60, 80, 80)]  # fully outside frame
    assert largest_bbox_crop(frame, _FakeDetector(dets)) is None
```

- [ ] **Step 2: Run — verify RED**

Expected: all fail with ImportError.

- [ ] **Step 3: Implement**

`src/pet_id_registry/enroll.py`:
```python
"""Enrollment pipeline: detect → crop → embed → build PetCard → save."""
from __future__ import annotations

import numpy as np

from purrai_core.interfaces.detector import Detector
from purrai_core.types import BBox


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
    # sort by area desc
    def area(d) -> float:
        return d.bbox.width * d.bbox.height
    for d in sorted(dets, key=area, reverse=True):
        clipped = _clip_bbox(d.bbox, frame)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        return frame[y1:y2, x1:x2].copy()
    return None
```

- [ ] **Step 4: Run — verify GREEN**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry/test_enroll_crop.py -v
```
Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/enroll.py tests/pet_id_registry/test_enroll_crop.py
git commit -m "feat(pet-id): largest_bbox_crop enrollment helper"
```

---

## Task 7: `enroll_photos` — full card from an image list

**Files:**
- Modify: `src/pet_id_registry/enroll.py`
- Test: `tests/pet_id_registry/test_enroll_photos.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from purrai_core.types import BBox, Detection
from pet_id_registry.card import PetSpecies
from pet_id_registry.enroll import NoDetectionsError, enroll_photos
from pet_id_registry.library import Library


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
```

- [ ] **Step 2: Run — verify RED**

- [ ] **Step 3: Extend `enroll.py`**

Add at the top of `enroll.py`:
```python
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import cv2

from pet_id_registry.card import PetCard, PetSpecies, PetSex, RegisteredView, compute_pet_id
from pet_id_registry.library import Library
from pet_id_registry.protocols import Embedder

_SCHEMA_VERSION = "1.0.0"


class NoDetectionsError(RuntimeError):
    """Raised when enrollment exhausts input without any usable crop."""
```

Then the function:
```python
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
```

- [ ] **Step 4: Run — verify GREEN**

Expected 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/enroll.py tests/pet_id_registry/test_enroll_photos.py
git commit -m "feat(pet-id): enroll_photos builds PetCard from image list"
```

---

## Task 8: `enroll_video` — FPS-sampled video path

**Files:**
- Modify: `src/pet_id_registry/enroll.py`
- Test: `tests/pet_id_registry/test_enroll_video.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from purrai_core.types import BBox, Detection
from pet_id_registry.card import PetSpecies
from pet_id_registry.enroll import enroll_video
from pet_id_registry.library import Library


def _write_video(path: Path, *, frames: int, fps: int = 30,
                 size: tuple[int, int] = (320, 240)) -> Path:
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
        return [Detection(bbox=BBox(10, 10, 200, 200), score=0.9,
                          class_id=15, class_name="cat")]


class _Emb:
    embedding_dim = 8

    def embed_crop(self, crop):
        return np.random.default_rng(crop.sum()).standard_normal(8).astype(np.float32)


def test_video_fps_sampling_and_cap(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.mp4", frames=60, fps=30)
    lib = Library(tmp_path / "gal")
    card = enroll_video(
        video_path=video, name="M", species=PetSpecies.cat,
        detector=_Det(), embedder=_Emb(), library=lib,
        fps_sample=2, max_views=3,
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
        video_path=video, name="M", species=PetSpecies.cat,
        detector=_Det(det_every=2), embedder=_Emb(), library=lib,
        fps_sample=2, max_views=10,
        created_at=datetime(2026, 4, 21),
    )
    # some frames skipped, fewer than requested views
    assert 1 <= len(card.views) <= 4
```

- [ ] **Step 2: Run — verify RED**

- [ ] **Step 3: Extend `enroll.py`**

```python
def _sample_video_frames(video_path: Path, fps_sample: float) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(src_fps / float(fps_sample))))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def enroll_video(
    *,
    video_path: Path,
    name: str,
    species: PetSpecies,
    detector: Detector,
    embedder: Embedder,
    library: Library,
    fps_sample: float,
    max_views: int,
    created_at: datetime,
    cover_photo: Path | None = None,
    force: bool = False,
    metadata: dict | None = None,
) -> PetCard:
    """Build a PetCard from a video by FPS-sampling + largest-bbox crop + embed."""
    crops: list[np.ndarray] = []
    for frame in _sample_video_frames(video_path, fps_sample):
        if len(crops) >= max_views:
            break
        crop = largest_bbox_crop(frame, detector)
        if crop is None:
            continue
        crops.append(crop)
    if not crops:
        raise NoDetectionsError(f"no pet detected in video: {video_path}")
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
    card = PetCard(
        pet_id=pet_id, name=name, species=species,
        created_at=created_at, schema_version=_SCHEMA_VERSION,
        cover_photo_uri=f"{pet_id}/cover.jpg", views=views,
        **(metadata or {}),
    )
    cover_crop = cv2.imread(str(cover_photo)) if cover_photo else None
    assets = list(zip(views, crops, embeddings, strict=True))
    library.save(card, view_assets=assets, cover_crop=cover_crop, force=force)
    return card
```

Also add `from collections.abc import Iterable` to the imports in `enroll.py`.

- [ ] **Step 4: Run — verify GREEN**

Expected 2/2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/enroll.py tests/pet_id_registry/test_enroll_video.py
git commit -m "feat(pet-id): enroll_video with FPS sampling + max_views cap"
```

---

## Task 9: CLI `petid register`

**Files:**
- Modify: `src/pet_id_registry/cli.py`
- Test: `tests/pet_id_registry/test_cli_register.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from click.testing import CliRunner

from purrai_core.types import BBox, Detection
from pet_id_registry.cli import main


def _write_img(p: Path, color: int = 128) -> Path:
    img = np.full((200, 200, 3), color, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return p


def _stub_factories(monkeypatch):
    """Patch build_detector / build_embedder so the CLI doesn't need torch weights."""

    class D:
        def detect(self, f):
            return [Detection(bbox=BBox(10, 10, 180, 180), score=0.9,
                              class_id=15, class_name="cat")]

    class E:
        embedding_dim = 8

        def embed_crop(self, c):
            return np.ones(8, dtype=np.float32) / np.sqrt(8.0)

    monkeypatch.setattr("pet_id_registry.cli.build_detector", lambda cfg: D())
    monkeypatch.setattr("pet_id_registry.cli.build_embedder", lambda cfg: E())


def _write_params(tmp_path: Path, library_root: str) -> Path:
    params = {
        "detector": {"model_name": "yolov10n", "conf_threshold": 0.3,
                     "iou_threshold": 0.5, "class_whitelist": [15, 16],
                     "device": "cpu", "imgsz": 640},
        "reid": {"model_name": "osnet_x0_25", "embedding_dim": 8,
                 "similarity_threshold": 0.65, "device": "cpu"},
        "pet_id": {"library_root": library_root, "fps_sample": 2,
                   "max_views": 8, "similarity_threshold": 0.55},
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path


def test_register_one_photo(tmp_path: Path, monkeypatch) -> None:
    _stub_factories(monkeypatch)
    img = _write_img(tmp_path / "photo.jpg")
    lib_root = tmp_path / "gallery"
    params = _write_params(tmp_path, str(lib_root))
    runner = CliRunner()
    result = runner.invoke(main, [
        "--params", str(params),
        "register", str(img),
        "--name", "Mimi",
        "--species", "cat",
    ])
    assert result.exit_code == 0, result.output
    assert "enrolled Mimi" in result.output
    assert any(lib_root.iterdir())


def test_register_refuses_duplicate_without_force(tmp_path: Path, monkeypatch) -> None:
    _stub_factories(monkeypatch)
    img = _write_img(tmp_path / "photo.jpg")
    lib_root = tmp_path / "gallery"
    params = _write_params(tmp_path, str(lib_root))
    runner = CliRunner()
    args = ["--params", str(params), "register", str(img), "--name", "Mimi", "--species", "cat"]
    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    second = runner.invoke(main, args)
    assert second.exit_code != 0
    assert "already exists" in second.output.lower() or "pet already" in second.output.lower()


def test_register_exits_nonzero_when_no_detection(tmp_path: Path, monkeypatch) -> None:
    _stub_factories(monkeypatch)

    class Empty:
        def detect(self, _):
            return []

    monkeypatch.setattr("pet_id_registry.cli.build_detector", lambda cfg: Empty())
    img = _write_img(tmp_path / "photo.jpg")
    lib_root = tmp_path / "gallery"
    params = _write_params(tmp_path, str(lib_root))
    runner = CliRunner()
    result = runner.invoke(main, [
        "--params", str(params), "register", str(img),
        "--name", "M", "--species", "cat",
    ])
    assert result.exit_code != 0
    assert "no pet detected" in result.output.lower()
```

- [ ] **Step 2: Run — verify RED**

- [ ] **Step 3: Implement `cli.py` register + supporting helpers**

```python
"""petid CLI — register / identify / list / show / delete."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import click
import yaml

from pet_id_registry.card import PetSex, PetSpecies
from pet_id_registry.enroll import NoDetectionsError, enroll_photos, enroll_video
from pet_id_registry.library import Library, PetAlreadyExistsError

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


def _load_params(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_detector(cfg: dict[str, Any]):
    from purrai_core.backends.yolov10_detector import YOLOv10Detector

    return YOLOv10Detector(cfg)


def build_embedder(cfg: dict[str, Any]):
    from purrai_core.backends.osnet_reid import OSNetReid
    from pet_id_registry.backends.osnet_embedder import OSNetEmbedderAdapter

    return OSNetEmbedderAdapter(OSNetReid(cfg))


def _classify_input(path: Path) -> str:
    if path.is_dir():
        return "dir"
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    raise click.UsageError(f"unsupported input type: {path}")


def _collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


@click.group()
@click.option("--params", "params_path", type=click.Path(exists=True, path_type=Path),
              default=Path("params.yaml"), show_default=True)
@click.pass_context
def main(ctx: click.Context, params_path: Path) -> None:
    """petid: pet identity enrollment + identification CLI."""
    ctx.ensure_object(dict)
    ctx.obj["params"] = _load_params(params_path)


@main.command("register")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--species", type=click.Choice([s.value for s in PetSpecies]), required=True)
@click.option("--breed", default=None)
@click.option("--sex", type=click.Choice([s.value for s in PetSex]), default=None)
@click.option("--birthdate", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option("--weight-kg", type=float, default=None)
@click.option("--markings", default=None)
@click.option("--owner-name", default=None)
@click.option("--medical-notes", default=None)
@click.option("--cover-photo", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.pass_context
def register_cmd(
    ctx: click.Context,
    input_path: Path,
    name: str,
    species: str,
    breed: str | None,
    sex: str | None,
    birthdate,
    weight_kg: float | None,
    markings: str | None,
    owner_name: str | None,
    medical_notes: str | None,
    cover_photo: Path | None,
    library_root: Path | None,
    force: bool,
) -> None:
    """Register a pet from photos, a video, or a directory of photos.

    \b
    Capture tip: record a 5–10 second video walking a full circle around the pet,
    OR take 5+ photos from different angles (front, left side, right side,
    top-down, sitting). More views → better recall on new angles.
    """
    params = ctx.obj["params"]
    pet_id_cfg = params["pet_id"]
    root = Path(library_root) if library_root else Path(pet_id_cfg["library_root"])
    library = Library(root)

    detector = build_detector(params["detector"])
    embedder = build_embedder(params["reid"])

    metadata: dict[str, Any] = {}
    if breed: metadata["breed"] = breed
    if sex: metadata["sex"] = sex
    if birthdate: metadata["birthdate"] = birthdate.date()
    if weight_kg is not None: metadata["weight_kg"] = weight_kg
    if markings: metadata["markings"] = markings
    if owner_name: metadata["owner_name"] = owner_name
    if medical_notes: metadata["medical_notes"] = medical_notes

    kind = _classify_input(input_path)
    try:
        if kind == "image":
            card = enroll_photos(
                image_paths=[input_path], name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        elif kind == "dir":
            images = _collect_images(input_path)
            if not images:
                raise click.UsageError(f"no images found in directory: {input_path}")
            card = enroll_photos(
                image_paths=images, name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        elif kind == "video":
            card = enroll_video(
                video_path=input_path, name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                fps_sample=float(pet_id_cfg["fps_sample"]),
                max_views=int(pet_id_cfg["max_views"]),
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        else:  # pragma: no cover — _classify_input exhausts known kinds
            raise click.UsageError(f"unsupported input: {input_path}")
    except NoDetectionsError as e:
        raise click.ClickException(f"no pet detected: {e}")
    except PetAlreadyExistsError as e:
        raise click.ClickException(
            f"pet already exists (pet_id={e}); rerun with --force to overwrite"
        )

    click.echo(f"enrolled {card.name} [{card.pet_id}] with {len(card.views)} view(s)")
```

- [ ] **Step 4: Run — verify GREEN**

Expected 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/cli.py tests/pet_id_registry/test_cli_register.py
git commit -m "feat(pet-id): petid register CLI"
```

---

## Task 10: CLI `petid identify`

**Files:**
- Modify: `src/pet_id_registry/cli.py`
- Test: `tests/pet_id_registry/test_cli_identify.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from click.testing import CliRunner

from purrai_core.types import BBox, Detection
from pet_id_registry.cli import main


def _write_img(p: Path, color: int = 100) -> Path:
    cv2.imwrite(str(p), np.full((200, 200, 3), color, dtype=np.uint8))
    return p


def _params(tmp_path: Path, lib_root: Path, thr: float = 0.55) -> Path:
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump({
        "detector": {"model_name": "yolov10n", "conf_threshold": 0.3,
                     "iou_threshold": 0.5, "class_whitelist": [15, 16],
                     "device": "cpu", "imgsz": 640},
        "reid": {"model_name": "osnet_x0_25", "embedding_dim": 8,
                 "similarity_threshold": 0.65, "device": "cpu"},
        "pet_id": {"library_root": str(lib_root), "fps_sample": 2,
                   "max_views": 8, "similarity_threshold": thr},
    }))
    return p


def _stub(monkeypatch, vec: np.ndarray) -> None:
    class D:
        def detect(self, f):
            return [Detection(bbox=BBox(10, 10, 180, 180), score=0.9,
                              class_id=15, class_name="cat")]

    class E:
        embedding_dim = 8

        def embed_crop(self, _c):
            return vec.astype(np.float32)

    monkeypatch.setattr("pet_id_registry.cli.build_detector", lambda cfg: D())
    monkeypatch.setattr("pet_id_registry.cli.build_embedder", lambda cfg: E())


def test_identify_hits_enrolled_pet(tmp_path: Path, monkeypatch) -> None:
    lib_root = tmp_path / "gal"
    params = _params(tmp_path, lib_root)
    img = _write_img(tmp_path / "p.jpg")
    runner = CliRunner()

    # enroll with vec_a
    vec_a = np.zeros(8, dtype=np.float32); vec_a[0] = 1.0
    _stub(monkeypatch, vec_a)
    assert runner.invoke(main, [
        "--params", str(params), "register", str(img),
        "--name", "Mimi", "--species", "cat",
    ]).exit_code == 0

    # identify with the same vec
    identify_img = _write_img(tmp_path / "q.jpg", color=200)
    result = runner.invoke(main, [
        "--params", str(params), "identify", str(identify_img), "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["name"] == "Mimi"
    assert data[0]["score"] >= 0.99


def test_identify_unknown_when_far(tmp_path: Path, monkeypatch) -> None:
    lib_root = tmp_path / "gal"
    params = _params(tmp_path, lib_root, thr=0.55)
    img = _write_img(tmp_path / "p.jpg")
    runner = CliRunner()

    vec_a = np.zeros(8, dtype=np.float32); vec_a[0] = 1.0
    _stub(monkeypatch, vec_a)
    runner.invoke(main, ["--params", str(params), "register", str(img),
                         "--name", "A", "--species", "cat"])

    vec_q = np.zeros(8, dtype=np.float32); vec_q[1] = 1.0  # orthogonal
    _stub(monkeypatch, vec_q)
    result = runner.invoke(main, [
        "--params", str(params), "identify", str(img), "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["name"] == "unknown"


def test_identify_empty_library(tmp_path: Path, monkeypatch) -> None:
    lib_root = tmp_path / "gal"
    lib_root.mkdir()
    params = _params(tmp_path, lib_root)
    img = _write_img(tmp_path / "p.jpg")

    vec_q = np.zeros(8, dtype=np.float32); vec_q[0] = 1.0
    _stub(monkeypatch, vec_q)
    result = CliRunner().invoke(main, [
        "--params", str(params), "identify", str(img), "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["name"] == "unknown"


def test_identify_rejects_video(tmp_path: Path, monkeypatch) -> None:
    lib_root = tmp_path / "gal"
    lib_root.mkdir()
    params = _params(tmp_path, lib_root)
    fake_video = tmp_path / "x.mp4"
    fake_video.write_bytes(b"not a real mp4 but suffix is enough for cli check")
    _stub(monkeypatch, np.ones(8, dtype=np.float32) / np.sqrt(8.0))
    result = CliRunner().invoke(main, [
        "--params", str(params), "identify", str(fake_video),
    ])
    assert result.exit_code != 0
    assert "still image" in result.output.lower() or "video" in result.output.lower()
```

- [ ] **Step 2: Run — verify RED**

- [ ] **Step 3: Extend `cli.py` with identify**

Append to `cli.py`:
```python
@main.command("identify")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def identify_cmd(ctx, input_path, library_root, as_json) -> None:
    """Identify pet(s) in a still image against the enrolled library."""
    params = ctx.obj["params"]
    pet_id_cfg = params["pet_id"]
    threshold = float(pet_id_cfg["similarity_threshold"])
    root = Path(library_root) if library_root else Path(pet_id_cfg["library_root"])
    library = Library(root)

    kind = _classify_input(input_path)
    if kind == "video":
        raise click.ClickException(
            "identify takes a still image in first round; extract a frame and retry"
        )
    if kind == "dir":
        image_paths = _collect_images(input_path)
    else:
        image_paths = [input_path]

    detector = build_detector(params["detector"])
    embedder = build_embedder(params["reid"])

    import cv2
    records = []
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            records.append({"file": str(img_path), "error": "cannot decode image"})
            continue
        dets = detector.detect(frame)
        if not dets:
            records.append({"file": str(img_path), "bbox": None, "name": "no detection",
                            "pet_id": None, "score": 0.0})
            continue
        for d in dets:
            x1, y1 = max(0, int(d.bbox.x1)), max(0, int(d.bbox.y1))
            x2, y2 = min(frame.shape[1], int(d.bbox.x2)), min(frame.shape[0], int(d.bbox.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            q = embedder.embed_crop(frame[y1:y2, x1:x2].copy())
            res = library.identify(q, threshold=threshold)
            records.append({
                "file": str(img_path),
                "bbox": [x1, y1, x2, y2],
                "pet_id": res.pet_id if res else None,
                "name": res.name if res else "unknown",
                "score": float(res.score) if res else 0.0,
            })

    if as_json:
        click.echo(json.dumps(records, indent=2))
    else:
        for r in records:
            if "error" in r:
                click.echo(f"{r['file']}: {r['error']}")
            else:
                bb = r["bbox"]
                click.echo(f"{r['file']} bbox={bb} → {r['name']} (score={r['score']:.3f})")
```

Also add `import json` at the top of `cli.py`.

- [ ] **Step 4: Run — verify GREEN**

Expected 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/cli.py tests/pet_id_registry/test_cli_identify.py
git commit -m "feat(pet-id): petid identify CLI"
```

---

## Task 11: CLI `petid list / show / delete`

**Files:**
- Modify: `src/pet_id_registry/cli.py`
- Test: `tests/pet_id_registry/test_cli_list_show_delete.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from click.testing import CliRunner

from purrai_core.types import BBox, Detection
from pet_id_registry.cli import main


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    lib_root = tmp_path / "gal"
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({
        "detector": {"model_name": "yolov10n", "conf_threshold": 0.3,
                     "iou_threshold": 0.5, "class_whitelist": [15, 16],
                     "device": "cpu", "imgsz": 640},
        "reid": {"model_name": "osnet_x0_25", "embedding_dim": 8,
                 "similarity_threshold": 0.65, "device": "cpu"},
        "pet_id": {"library_root": str(lib_root), "fps_sample": 2,
                   "max_views": 8, "similarity_threshold": 0.55},
    }))

    class D:
        def detect(self, _):
            return [Detection(bbox=BBox(0, 0, 100, 100), score=0.9,
                              class_id=15, class_name="cat")]

    class E:
        embedding_dim = 8

        def __init__(self, axis: int) -> None:
            self._axis = axis

        def embed_crop(self, _c):
            v = np.zeros(8, dtype=np.float32); v[self._axis] = 1.0
            return v

    monkeypatch.setattr("pet_id_registry.cli.build_detector", lambda cfg: D())
    axis_ref = {"axis": 0}
    monkeypatch.setattr("pet_id_registry.cli.build_embedder", lambda cfg: E(axis_ref["axis"]))
    return params, lib_root, axis_ref  # caller mutates axis_ref between invocations


def _write_img(p: Path) -> Path:
    cv2.imwrite(str(p), np.full((200, 200, 3), 77, dtype=np.uint8))
    return p


def test_list_and_show_and_delete(tmp_path: Path, monkeypatch) -> None:
    params, lib_root, axis_ref = _setup(tmp_path, monkeypatch)
    img = _write_img(tmp_path / "p.jpg")
    runner = CliRunner()
    axis_ref["axis"] = 0
    runner.invoke(main, ["--params", str(params), "register", str(img),
                         "--name", "A", "--species", "cat"])
    axis_ref["axis"] = 1
    runner.invoke(main, ["--params", str(params), "register", str(img),
                         "--name", "B", "--species", "dog"])

    # list
    r_list = runner.invoke(main, ["--params", str(params), "list", "--json"])
    assert r_list.exit_code == 0
    entries = json.loads(r_list.output)
    assert {e["name"] for e in entries} == {"A", "B"}

    # show one
    pet_id = entries[0]["pet_id"]
    r_show = runner.invoke(main, ["--params", str(params), "show", pet_id, "--json"])
    assert r_show.exit_code == 0
    shown = json.loads(r_show.output)
    assert shown["pet_id"] == pet_id

    # delete with --yes
    r_del = runner.invoke(main, ["--params", str(params), "delete", pet_id, "--yes"])
    assert r_del.exit_code == 0
    r_list2 = runner.invoke(main, ["--params", str(params), "list", "--json"])
    entries2 = json.loads(r_list2.output)
    assert all(e["pet_id"] != pet_id for e in entries2)


def test_show_missing_returns_error(tmp_path: Path, monkeypatch) -> None:
    params, _, _ = _setup(tmp_path, monkeypatch)
    r = CliRunner().invoke(main, ["--params", str(params), "show", "deadbeef"])
    assert r.exit_code != 0
    assert "not found" in r.output.lower()
```

- [ ] **Step 2: Run — verify RED**

- [ ] **Step 3: Extend `cli.py`**

```python
@main.command("list")
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def list_cmd(ctx, library_root, as_json) -> None:
    """List enrolled pets."""
    params = ctx.obj["params"]
    root = Path(library_root) if library_root else Path(params["pet_id"]["library_root"])
    library = Library(root)
    entries = library.list()
    payload = [
        {"pet_id": e.pet_id, "name": e.name, "species": e.species,
         "view_count": e.view_count, "created_at": e.created_at}
        for e in entries
    ]
    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        for e in entries:
            click.echo(f"{e.pet_id}  {e.name:<16}  {e.species:<6}  "
                       f"views={e.view_count}  {e.created_at}")


@main.command("show")
@click.argument("pet_id")
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def show_cmd(ctx, pet_id, library_root, as_json) -> None:
    """Show a PetCard's full JSON."""
    from pet_id_registry.library import PetNotFoundError

    params = ctx.obj["params"]
    root = Path(library_root) if library_root else Path(params["pet_id"]["library_root"])
    library = Library(root)
    try:
        card = library.load(pet_id)
    except PetNotFoundError:
        raise click.ClickException(f"pet_id not found: {pet_id}")
    if as_json:
        click.echo(card.model_dump_json(indent=2))
    else:
        click.echo(f"pet_id:     {card.pet_id}")
        click.echo(f"name:       {card.name}")
        click.echo(f"species:    {card.species.value}")
        click.echo(f"views:      {len(card.views)}")
        click.echo(f"created_at: {card.created_at.isoformat()}")


@main.command("delete")
@click.argument("pet_id")
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--yes", is_flag=True, help="skip confirmation")
@click.pass_context
def delete_cmd(ctx, pet_id, library_root, yes) -> None:
    """Delete an enrolled pet."""
    from pet_id_registry.library import PetNotFoundError

    params = ctx.obj["params"]
    root = Path(library_root) if library_root else Path(params["pet_id"]["library_root"])
    library = Library(root)
    if not yes and not click.confirm(f"delete pet_id {pet_id}?"):
        click.echo("aborted")
        return
    try:
        library.delete(pet_id)
    except PetNotFoundError:
        raise click.ClickException(f"pet_id not found: {pet_id}")
    click.echo(f"deleted {pet_id}")
```

- [ ] **Step 4: Run — verify GREEN**

Expected 2/2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pet_id_registry/cli.py tests/pet_id_registry/test_cli_list_show_delete.py
git commit -m "feat(pet-id): petid list/show/delete CLI"
```

---

## Task 12: `params.yaml` pet_id block + README touch-ups

**Files:**
- Modify: `params.yaml`, `params.cpu.yaml`, `params.mps.yaml`
- Modify: `README.md`

- [ ] **Step 1: Append `pet_id:` block to `params.yaml`, `params.cpu.yaml`, `params.mps.yaml`**

```yaml
# ---- pet-id registry ----
pet_id:
  library_root: "artifacts/pet_id_library"
  fps_sample: 2
  max_views: 8
  similarity_threshold: 0.55
```

Paste it at the bottom of each params file, keeping the existing content untouched.

- [ ] **Step 2: Replace the pet-id `README.md` with a short usage section**

Keep whatever already exists at the top (project description / bootstrap history). Append a **"pet-id registry (v0.1.0)"** section with, literally:

```markdown
## pet-id registry (v0.1.0)

### Install

```bash
pip install -e ".[detector,reid]"
```

Entry point `petid` is registered automatically.

### Enroll a pet

```bash
# one photo
petid register photos/mimi_01.jpg --name Mimi --species cat

# a folder of photos
petid register photos/mimi/ --name Mimi --species cat --breed "Domestic Shorthair"

# a short video
petid register videos/mimi_circle.mp4 --name Mimi --species cat --weight-kg 4.2
```

**Capture tip:** a 5–10 second video walking a full circle around the pet, OR
5+ photos from different angles (front / left / right / top / sitting).

### Identify

```bash
petid identify query.jpg
# → query.jpg bbox=[42, 30, 310, 240] → Mimi (score=0.812)
petid identify query.jpg --json
```

### Browse

```bash
petid list
petid show <pet_id>
petid delete <pet_id> --yes
```

### Configuration

All numerics (library root, FPS sampling, view cap, match threshold) live in
`params.yaml` under the `pet_id:` block. See
`docs/superpowers/specs/2026-04-21-pet-id-first-round-design.md` for the full
design.
```

- [ ] **Step 3: Commit**

```bash
git add params.yaml params.cpu.yaml params.mps.yaml README.md
git commit -m "docs(pet-id): params.yaml pet_id block + README usage"
```

---

## Task 13: Full-suite gate — ruff, mypy, pytest

**Files:** (no source changes unless the gate surfaces real issues)

- [ ] **Step 1: Run ruff**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m ruff check src/pet_id_registry tests/pet_id_registry
```
Expected: All checks passed.

- [ ] **Step 2: Run mypy (strict)**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m mypy src/pet_id_registry
```
Expected: Success: no issues found in N source files. Add explicit return types and type annotations where mypy flags, minimal changes only.

- [ ] **Step 3: Run the full pytest suite**

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest tests/pet_id_registry -v
```
Expected: all tests pass.

```bash
/Users/bamboo/.miniconda3/envs/pet-pipeline/bin/python -m pytest -v
```
Expected: entire repo suite green (including inherited `tests/backends`, `tests/pipelines`, `tests/utils` — they should be unaffected).

- [ ] **Step 4: If any of the above surfaced fixable issues, commit the fix**

```bash
git add <files>
git commit -m "fix(pet-id): <what was fixed> to pass lint/type/test gates"
```

If nothing to fix, no commit.

---

## Task 14: PR → dev → main → tag v0.1.0

**Files:** none

- [ ] **Step 1: Push feature branch**

```bash
git push -u origin feature/pet-id-first-round
```

- [ ] **Step 2: Open PR → dev**

```bash
gh pr create --base dev --title "feat(pet-id): first-round PetCard registry + petid CLI" --body "$(cat <<'EOF'
## Summary

First round of pet-id (thin PetCard library + petid CLI). Implements
the design at docs/superpowers/specs/2026-04-21-pet-id-first-round-design.md.

## What's in

- `src/pet_id_registry/` new sibling package (purrai_core untouched)
  - Pydantic `PetCard` + `RegisteredView` + content-addressed `pet_id`
  - `Library` flat-file gallery (save/load/list/delete/identify)
  - `enroll_photos` / `enroll_video` (FPS sampling + max_views cap)
  - `petid` click CLI: register / identify / list / show / delete
- `params.yaml` gains a `pet_id:` block (library_root, fps_sample, max_views, similarity_threshold)
- README usage section + capture tip
- Unit + integration + CLI tests with injected fake backends

## What's out (documented as non-goals in the spec)

- Integration into `purrai_core.pipelines.full_pipeline` (next round)
- Interactive multi-pet disambiguation, farthest-first view selection, blur filtering
- Auto re-embedding on model upgrade
- `PetCard` upstreamed to pet-schema (Phase 2+)

## Test plan

- [ ] CI green (lint + type + pytest)
- [ ] Manual smoke: `petid register` on one photo + `petid identify` on the same image round-trips

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Wait for CI green (pet-id already has the standard workflow)**

```bash
gh pr checks <PR#> --watch
```

- [ ] **Step 4: Merge → dev**

```bash
gh pr merge <PR#> --squash --admin --delete-branch
```

- [ ] **Step 5: Open release PR dev → main**

```bash
git fetch origin && git checkout dev && git pull
gh pr create --base main --head dev --title "release(pet-id): v0.1.0 — first-round registry + CLI" --body "First pet-id release. See feature PR for details."
```

- [ ] **Step 6: Merge release PR and tag v0.1.0**

```bash
gh pr merge <release PR#> --merge --admin
git checkout main && git pull --ff-only origin main
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --title "pet-id 0.1.0 — first-round registry + CLI" --notes "See docs/superpowers/specs/2026-04-21-pet-id-first-round-design.md"
```

- [ ] **Step 7: Back-merge main → dev to keep branches aligned**

```bash
git checkout dev && git merge --ff-only origin/main && git push origin dev
```

---

## Post-plan checklist

- [ ] All 14 tasks committed on `feature/pet-id-first-round`
- [ ] PR #<N> merged to dev
- [ ] dev → main release PR merged
- [ ] `v0.1.0` tag pushed + GitHub release published
- [ ] Memory update: `project_pet_id_status.md` reflects shipped v0.1.0
