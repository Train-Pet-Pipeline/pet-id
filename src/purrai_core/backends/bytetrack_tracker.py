"""ByteTrack tracker backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

from purrai_core.interfaces.tracker import Tracker
from purrai_core.types import BBox, Detection, Track

# Sentinel blank frame used when caller does not supply a real frame.
_BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class ByteTrackTracker(Tracker):
    """ByteTrack wrapper conforming to Tracker protocol."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialise tracker from config section."""
        self.cfg = cfg
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> ByteTrack:
        """Instantiate a fresh ByteTrack instance from config."""
        return ByteTrack(
            track_thresh=float(self.cfg["track_thresh"]),
            match_thresh=float(self.cfg["match_thresh"]),
            track_buffer=int(self.cfg["track_buffer"]),
            frame_rate=int(self.cfg["frame_rate"]),
            # Confirm tracks immediately; avoids silent misses in short clips.
            min_hits=1,
        )

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Update tracker with current-frame detections and return active tracks.

        Args:
            detections: Detections from the current frame.
            frame_idx: 0-based frame index (unused internally, kept for protocol compat).
            frame: BGR image as np.ndarray (H×W×3 uint8).  When omitted a blank
                   frame of the default resolution is substituted.

        Returns:
            List of confirmed tracks.
        """
        img = frame if frame is not None else _BLANK_FRAME
        if not detections:
            self._tracker.update(np.empty((0, 6), dtype=np.float32), img)
            return []

        arr = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.score, d.class_id] for d in detections],
            dtype=np.float32,
        )
        raw = self._tracker.update(arr, img)

        out: list[Track] = []
        for row in raw:
            x1, y1, x2, y2, track_id, conf, cls_id = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
            )
            cid = int(cls_id)
            if cid == 15:
                class_name = "cat"
            elif cid == 16:
                class_name = "dog"
            else:
                class_name = "unknown"
            out.append(
                Track(
                    track_id=int(track_id),
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    score=float(conf),
                    class_id=cid,
                    class_name=class_name,
                )
            )
        return out

    def reset(self) -> None:
        """Reset tracker state, restarting track IDs from 1."""
        self._tracker = self._new_tracker()
