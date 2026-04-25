"""YOLOv10 detector backend tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# F015 fix: skip when optional 'detector' extra not installed (ultralytics).
# pet-id pyproject [project.optional-dependencies] declares detector = ["ultralytics>=8.3"]
# but base `make test` doesn't install it, causing ImportError at collection time.
pytest.importorskip("ultralytics")

from purrai_core.backends.yolov10_detector import YOLOv10Detector
from purrai_core.config import load_config


@pytest.fixture
def fake_ultra_result() -> MagicMock:
    """Mock an ultralytics Results object."""
    r = MagicMock()
    r.boxes = MagicMock()
    r.boxes.xyxy = MagicMock()
    r.boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[10, 20, 100, 200]], dtype=np.float32
    )
    r.boxes.conf = MagicMock()
    r.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92], dtype=np.float32)
    r.boxes.cls = MagicMock()
    r.boxes.cls.cpu.return_value.numpy.return_value = np.array([15], dtype=np.float32)
    r.names = {15: "cat", 16: "dog"}
    return r


def test_yolov10_detect_filters_by_whitelist(
    params_yaml_path: Path, fake_ultra_result: MagicMock
) -> None:
    cfg = load_config(params_yaml_path)
    with patch("purrai_core.backends.yolov10_detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock(return_value=[fake_ultra_result])
        mock_yolo_cls.return_value = mock_model
        det = YOLOv10Detector(cfg.section("detector"))
        result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].class_name == "cat"
        assert result[0].score == pytest.approx(0.92, rel=1e-3)


def test_yolov10_drops_below_conf(params_yaml_path: Path, fake_ultra_result: MagicMock) -> None:
    fake_ultra_result.boxes.conf.cpu.return_value.numpy.return_value = np.array(
        [0.1], dtype=np.float32
    )
    cfg = load_config(params_yaml_path)
    with patch("purrai_core.backends.yolov10_detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock(return_value=[fake_ultra_result])
        det = YOLOv10Detector(cfg.section("detector"))
        assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_yolov10_drops_class_not_in_whitelist(
    params_yaml_path: Path, fake_ultra_result: MagicMock
) -> None:
    fake_ultra_result.boxes.cls.cpu.return_value.numpy.return_value = np.array(
        [0], dtype=np.float32
    )  # person
    cfg = load_config(params_yaml_path)
    with patch("purrai_core.backends.yolov10_detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock(return_value=[fake_ultra_result])
        det = YOLOv10Detector(cfg.section("detector"))
        assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
