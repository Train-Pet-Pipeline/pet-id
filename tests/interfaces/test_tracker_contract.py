"""Tracker contract tests."""

from purrai_core.interfaces.tracker import Tracker
from purrai_core.types import BBox, Detection, Track


class FakeTracker:
    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        return [
            Track(
                track_id=1, bbox=d.bbox, score=d.score, class_id=d.class_id, class_name=d.class_name
            )
            for d in detections
        ]

    def reset(self) -> None:
        pass


def test_tracker_protocol_accepts_fake() -> None:
    t: Tracker = FakeTracker()
    dets = [Detection(bbox=BBox(0, 0, 1, 1), score=0.9, class_id=15, class_name="cat")]
    tracks = t.update(dets, frame_idx=0)
    assert tracks[0].track_id == 1
    t.reset()


def test_tracker_runtime_checkable_rejects_bad() -> None:
    class BadTracker:
        pass

    assert not isinstance(BadTracker(), Tracker)
