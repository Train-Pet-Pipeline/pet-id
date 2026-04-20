"""Test config loader."""

from pathlib import Path

import pytest

from purrai_core.config import Config, load_config


def test_load_config_returns_config_instance(tmp_path: Path) -> None:
    """load_config should parse params.yaml into a Config instance."""
    yaml = tmp_path / "params.yaml"
    yaml.write_text('version: "0.1.0"\ndetector:\n  conf_threshold: 0.35\n  iou_threshold: 0.5\n')
    cfg = load_config(yaml)
    assert isinstance(cfg, Config)
    assert cfg.version == "0.1.0"
    assert cfg.raw["detector"]["conf_threshold"] == 0.35


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_config_get_section(tmp_path: Path) -> None:
    yaml = tmp_path / "params.yaml"
    yaml.write_text('version: "0.1.0"\ndetector:\n  conf_threshold: 0.3\n')
    cfg = load_config(yaml)
    assert cfg.section("detector")["conf_threshold"] == 0.3


def test_config_get_missing_section_raises(tmp_path: Path) -> None:
    yaml = tmp_path / "params.yaml"
    yaml.write_text('version: "0.1.0"\n')
    cfg = load_config(yaml)
    with pytest.raises(KeyError):
        cfg.section("detector")
