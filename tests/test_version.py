"""Parity tests: both top-level packages' __version__ must match installed metadata."""

from __future__ import annotations

import importlib.metadata

import pet_id_registry
import purrai_core


def test_pet_id_registry_version_matches_metadata() -> None:
    """pet_id_registry.__version__ must equal the installed pet-id package version."""
    installed = importlib.metadata.version("pet-id")
    assert pet_id_registry.__version__ == installed, (
        f"pet_id_registry.__version__ ({pet_id_registry.__version__!r}) does not "
        f"match installed package metadata ({installed!r}). "
        "Update src/pet_id_registry/__init__.py to match pyproject.toml version."
    )


def test_purrai_core_version_matches_pet_id_registry() -> None:
    """purrai_core.__version__ is bumped in lockstep with pet_id_registry (same pkg)."""
    assert purrai_core.__version__ == pet_id_registry.__version__, (
        f"purrai_core.__version__ ({purrai_core.__version__!r}) does not match "
        f"pet_id_registry.__version__ ({pet_id_registry.__version__!r}). "
        "Both packages ship from the same pet-id distribution; bump in lockstep."
    )
