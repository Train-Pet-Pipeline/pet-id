from pathlib import Path

import numpy as np
import pytest

from purrai_core.utils.video_io import VideoMetadata, iter_frames, read_metadata


def test_iter_frames_yields_ndarrays(sample_video: Path) -> None:
    frames = list(iter_frames(sample_video, max_frames=5))
    assert len(frames) == 5
    for _idx, frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3


def test_read_metadata_returns_fps_and_size(sample_video: Path) -> None:
    md = read_metadata(sample_video)
    assert isinstance(md, VideoMetadata)
    assert md.fps > 0
    assert md.width > 0 and md.height > 0
    assert md.frame_count > 0


def test_read_metadata_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_metadata("/nonexistent.mp4")
