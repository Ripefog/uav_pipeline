"""Core data contracts shared by every module.

These types are load-bearing: the detector, tracker, follower and sinks all
pass them around, so changing a field here ripples through the pipeline.
Coordinates are always in **original-image pixels** unless noted.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """One object detection in original-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int            # integer class id from the model
    name: str = ""      # human label resolved from config names_yaml (model-agnostic)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


class Track:
    """A single tracked object (constant-velocity, ported from visdrone_mot).

    Mirrors ``src/drone_tracker.py::Track`` but is multi-class aware:
    carries ``cls``/``name`` from the matched detection and an optional
    ``plate_text`` filled by the OCR stage.
    """

    __slots__ = (
        "track_id", "bbox", "confidence", "cls", "name", "plate_text",
        "first_seen", "last_seen", "age", "total_visible",
        "trajectory", "max_trajectory_len", "velocity",
    )

    def __init__(self, track_id: int, bbox, confidence: float, frame_id: int,
                 cls: int = -1, name: str = ""):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.cls = int(cls)
        self.name = name
        self.plate_text: Optional[str] = None
        # timing
        self.first_seen = frame_id
        self.last_seen = frame_id
        self.age = 0                 # frames since last matched detection
        self.total_visible = 1       # total frames with a matched detection
        # motion
        self.trajectory: List[Tuple[float, float]] = [self.center]
        self.max_trajectory_len = 60
        self.velocity = np.zeros(2, dtype=np.float32)  # (vx, vy) px/frame

    # ---- geometry ----------------------------------------------------------
    @property
    def center(self) -> Tuple[float, float]:
        return (
            float((self.bbox[0] + self.bbox[2]) / 2.0),
            float((self.bbox[1] + self.bbox[3]) / 2.0),
        )

    @property
    def width(self) -> float:
        return max(0.0, float(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return max(0.0, float(self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return self.center[0]

    @property
    def cy(self) -> float:
        return self.center[1]

    # ---- state transitions -------------------------------------------------
    def update(self, bbox, confidence: float, frame_id: int,
               cls: Optional[int] = None, name: Optional[str] = None):
        """Update with a matched detection (constant-velocity estimate)."""
        old_center = self.center
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        if cls is not None:
            self.cls = int(cls)
        if name is not None:
            self.name = name
        self.last_seen = frame_id
        self.age = 0
        self.total_visible += 1
        new_center = self.center
        self.velocity = np.array(
            [new_center[0] - old_center[0], new_center[1] - old_center[1]],
            dtype=np.float32,
        )
        self.trajectory.append(new_center)
        if len(self.trajectory) > self.max_trajectory_len:
            self.trajectory.pop(0)

    def predict(self) -> np.ndarray:
        """Predict next bbox using constant velocity (no Kalman)."""
        predicted = self.bbox.copy()
        predicted[0] += self.velocity[0]
        predicted[1] += self.velocity[1]
        predicted[2] += self.velocity[0]
        predicted[3] += self.velocity[1]
        return predicted

    def mark_missed(self):
        self.age += 1

    def as_dict(self) -> Dict:
        return {
            "id": self.track_id,
            "bbox": [float(v) for v in self.bbox],
            "score": self.confidence,
            "cls": self.cls,
            "name": self.name,
            "plate_text": self.plate_text,
            "age": self.age,
            "hits": self.total_visible,
        }


@dataclass
class Command:
    """Control command emitted by the Follow layer (gimbal + body rates)."""

    ts: float
    yaw_rate: float = 0.0       # deg/s  (gimbal yaw / body yaw rate)
    pitch_rate: float = 0.0     # deg/s  (gimbal pitch)
    forward_vel: float = 0.0    # m/s    (body forward velocity)
    vertical_vel: float = 0.0   # m/s    (body vertical velocity)
    target_id: Optional[int] = None
    target_lost: bool = False
    raw_errors: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameMeta:
    idx: int
    ts: float
    shape_hw: Tuple[int, int]   # (height, width) of the original frame


@dataclass
class FollowState:
    target_id: Optional[int] = None
    locked: bool = False
    target_lost: bool = False
    mode: str = "idle"          # idle | acquire | tracking | recover


@dataclass
class FrameContext:
    """Everything a sink needs to render/log one frame."""

    meta: FrameMeta
    frame: Optional[np.ndarray] = None     # BGR HxWx3 (may be None if sinks skip draw)
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    follow_state: FollowState = field(default_factory=FollowState)
    command: Optional[Command] = None
    fps: float = 0.0
    extra_stats: Dict[str, str] = field(default_factory=dict)
