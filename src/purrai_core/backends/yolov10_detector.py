"""YOLOv10 detector backend via ultralytics."""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO

from purrai_core.interfaces.detector import Detector
from purrai_core.types import BBox, Detection


class YOLOv10Detector(Detector):
    """Detector backend using ultralytics YOLOv10."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.conf_threshold = float(cfg["conf_threshold"])
        self.iou_threshold = float(cfg["iou_threshold"])
        self.class_whitelist = {int(c) for c in cfg["class_whitelist"]}
        self.device = str(cfg["device"])
        self.imgsz = int(cfg["imgsz"])
        weights = cfg.get("weights_url") or cfg["model_name"]
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names
        out: list[Detection] = []
        for bbox, conf, cls_id in zip(xyxy, confs, clss, strict=False):
            if int(cls_id) not in self.class_whitelist:
                continue
            if float(conf) < self.conf_threshold:
                continue
            out.append(
                Detection(
                    bbox=BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    score=float(conf),
                    class_id=int(cls_id),
                    class_name=str(names.get(int(cls_id), "unknown")),
                )
            )
        return out
