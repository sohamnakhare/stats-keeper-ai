"""JSON-friendly dataclasses for pipeline artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

METERS_TO_FEET = 3.280839895

FORMAT_DETECTIONS = "play-detections-v1"
FORMAT_TRACKS = "play-tracks-v1"
FORMAT_TRAJECTORIES = "play-trajectories-v1"
FORMAT_EVENTS = "play-events-v1"


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to JSON-serializable dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class Detection:
    """One YOLO box in one sampled frame. Pixel coords are normalized 0-1."""

    frame: int
    t: float
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    foot_x: float | None = None
    foot_y: float | None = None


@dataclass
class TrackPoint:
    """One observation (or gap) for a tracked object."""

    frame: int
    t: float
    px: float
    py: float
    conf: float | None = None
    missing: bool = False
    predicted: bool = False
    x: float | None = None  # court feet (filled after homography)
    y: float | None = None


@dataclass
class Track:
    track_id: int
    label: str
    points: list[TrackPoint] = field(default_factory=list)
    gaps: list[dict[str, float | int]] = field(default_factory=list)


@dataclass
class Court:
    layout: str  # "half" | "full"
    length: float
    width: float
    unit: str = "ft"
    preset: str | None = None


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    frame: int | None = None
    interpolated: bool = False
    smoothed: bool = False
    conf: float | None = None


@dataclass
class Trajectory:
    player_id: int
    label: str
    points: list[TrajectoryPoint] = field(default_factory=list)
