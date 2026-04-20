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
