"""Video IO helpers wrapping OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from a video file."""

    fps: float
    width: int
    height: int
    frame_count: int


def read_metadata(path: str | Path) -> VideoMetadata:
    """Read video metadata without decoding frames.

    Args:
        path: Path to the video file.

    Returns:
        VideoMetadata with fps, dimensions, and frame count.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        return VideoMetadata(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, max_frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) tuples from a video file.

    Args:
        path: Path to the video file.
        max_frames: Maximum number of frames to yield; None means all frames.

    Yields:
        Tuples of (frame_index, BGR frame array).

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        idx = 0
        while True:
            if max_frames is not None and idx >= max_frames:
                return
            ret, frame = cap.read()
            if not ret:
                return
            yield idx, frame
            idx += 1
    finally:
        cap.release()
