from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from click.testing import CliRunner

from pet_id_registry.cli import main
from purrai_core.types import BBox, Detection


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    lib_root = tmp_path / "gal"
    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "detector": {
                    "model_name": "yolov10n",
                    "conf_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "class_whitelist": [15, 16],
                    "device": "cpu",
                    "imgsz": 640,
                },
                "reid": {
                    "model_name": "osnet_x0_25",
                    "embedding_dim": 8,
                    "similarity_threshold": 0.65,
                    "device": "cpu",
                },
                "pet_id": {
                    "library_root": str(lib_root),
                    "fps_sample": 2,
                    "max_views": 8,
                    "similarity_threshold": 0.55,
                },
            }
        )
    )

    class D:
        def detect(self, _):
            return [Detection(bbox=BBox(0, 0, 100, 100), score=0.9, class_id=15, class_name="cat")]

    class E:
        embedding_dim = 8

        def __init__(self, axis: int) -> None:
            self._axis = axis

        def embed_crop(self, _c):
            v = np.zeros(8, dtype=np.float32)
            v[self._axis] = 1.0
            return v

    monkeypatch.setattr("pet_id_registry.cli.build_detector", lambda cfg: D())
    axis_ref = {"axis": 0}
    monkeypatch.setattr("pet_id_registry.cli.build_embedder", lambda cfg: E(axis_ref["axis"]))
    return params, lib_root, axis_ref


def _write_img(p: Path) -> Path:
    cv2.imwrite(str(p), np.full((200, 200, 3), 77, dtype=np.uint8))
    return p


def test_list_and_show_and_delete(tmp_path: Path, monkeypatch) -> None:
    params, lib_root, axis_ref = _setup(tmp_path, monkeypatch)
    img = _write_img(tmp_path / "p.jpg")
    runner = CliRunner()
    axis_ref["axis"] = 0
    runner.invoke(
        main, ["--params", str(params), "register", str(img), "--name", "A", "--species", "cat"]
    )
    axis_ref["axis"] = 1
    runner.invoke(
        main, ["--params", str(params), "register", str(img), "--name", "B", "--species", "dog"]
    )

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
