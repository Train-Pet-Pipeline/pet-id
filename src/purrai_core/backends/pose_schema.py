"""Pose-backend schema constants (pure data — no backend dependencies).

Kept separate from backend implementations (e.g. mmpose_pose.py) so that
callers needing only the keypoint name list can import these constants
without pulling in heavy inference dependencies like mmpose/mmcv.
"""

from __future__ import annotations

AP10K_KPT_NAMES: list[str] = [
    "left_eye",
    "right_eye",
    "nose",
    "neck",
    "root_of_tail",
    "left_shoulder",
    "left_elbow",
    "left_front_paw",
    "right_shoulder",
    "right_elbow",
    "right_front_paw",
    "left_hip",
    "left_knee",
    "left_back_paw",
    "right_hip",
    "right_knee",
    "right_back_paw",
]
