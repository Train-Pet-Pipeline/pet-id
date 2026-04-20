"""Config loader — single source of truth backed by params.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Immutable config view over a parsed YAML file."""

    raw: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise KeyError(f"Config section '{name}' not found in params.yaml")
        return dict(self.raw[name])


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(raw=data)
