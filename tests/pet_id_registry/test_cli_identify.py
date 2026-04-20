from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from click.testing import CliRunner

from pet_id_registry.cli import main
from purrai_core.types import BBox, Detection


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
    vec_a = np.zeros(8, dtype=np.float32)
    vec_a[0] = 1.0
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

    vec_a = np.zeros(8, dtype=np.float32)
    vec_a[0] = 1.0
    _stub(monkeypatch, vec_a)
    runner.invoke(main, ["--params", str(params), "register", str(img),
                         "--name", "A", "--species", "cat"])

    vec_q = np.zeros(8, dtype=np.float32)
    vec_q[1] = 1.0  # orthogonal
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

    vec_q = np.zeros(8, dtype=np.float32)
    vec_q[0] = 1.0
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
