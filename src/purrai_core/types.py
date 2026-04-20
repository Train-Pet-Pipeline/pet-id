"""Shared dataclass types used across interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bbox in pixel coordinates (x1,y1) top-left, (x2,y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bbox: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class PoseResult:
    track_id: int
    keypoints: list[Keypoint]


@dataclass(frozen=True)
class ReidEmbedding:
    track_id: int
    vector: tuple[float, ...]
    identity_id: int | None = None


@dataclass(frozen=True)
class NarrativeOutput:
    text: str
    confidence: float | None
    meta: dict[str, str] | None = None
