"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from purrai_core.config import Config, load_config


@pytest.fixture(scope="session")
def params_yaml_path() -> Path:
    return Path(__file__).parent.parent / "params.yaml"


@pytest.fixture(scope="session")
def config(params_yaml_path: Path) -> Config:
    return load_config(params_yaml_path)


@pytest.fixture(scope="session")
def sample_video() -> Path:
    return Path(__file__).parent / "fixtures" / "sample.mp4"


@pytest.fixture(scope="session")
def sample_clips_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "clips"
