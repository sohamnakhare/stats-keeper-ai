"""Pydantic models for ball tracking artifacts."""

from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel, Field

_DEFAULT_DEVICE = "mps" if sys.platform == "darwin" else "cuda:0"


class BallTrackPoint(BaseModel):
    frame: int
    t_sec: float = Field(alias="tSec")
    x: float
    y: float
    w: float | None = None
    h: float | None = None
    detected: bool
    predicted: bool = False
    conf: float | None = None

    model_config = {"populate_by_name": True}


class BallTrackArtifact(BaseModel):
    game_id: str = Field(alias="gameId")
    job_id: str = Field(alias="jobId")
    sample_fps: float = Field(alias="sampleFps")
    frame_count: int = Field(alias="frameCount")
    model_version: str = Field(alias="modelVersion")
    detection_rate: float = Field(alias="detectionRate")
    points: list[BallTrackPoint]

    model_config = {"populate_by_name": True}


class DetectorConfig(BaseModel):
    imgsz: int = 704
    conf: float = 0.20
    ball_conf: float = 0.05
    hoop_conf: float = 0.60
    iou: float = 0.5
    device: str = _DEFAULT_DEVICE
    target_class: Literal["basketball"] = "basketball"


class HoopRoi(BaseModel):
    x: float
    y: float
    w: float
    h: float


class CourtPoint(BaseModel):
    x: float
    y: float


class BallPosition(BaseModel):
    x: float
    y: float


class ShotAttempt(BaseModel):
    frame: int
    t_sec: float = Field(alias="tSec")
    confidence: float
    entry_speed: float = Field(alias="entrySpeed")
    ball_pos: BallPosition = Field(alias="ballPos")
    deduped: bool = False
    primary_frame: int | None = Field(default=None, alias="primaryFrame")
    signal_kind: str | None = Field(default=None, alias="signalKind")

    model_config = {"populate_by_name": True}


class RimInteractionConfig(BaseModel):
    """Swish-friendly rim interaction signals and relaxed confirm gates at the rim."""

    enable_crossed_rim_downward: bool = True
    enable_entered_rim_tube: bool = True
    relax_confirm_gates_at_rim: bool = True
    rim_tube_above_depth: float = 0.02
    rim_tube_below_depth: float = 0.05
    min_rim_drop: float = 0.003


class HoopTrackingConfig(BaseModel):
    """Stabilize per-frame hoop ROI when YOLO drops or spuriously jumps."""

    hold_last_good_hoop: bool = True
    reject_hoop_jump: bool = True
    max_hoop_jump_x: float = 0.15
    max_hoop_hold_frames: int = 0  # 0 = hold indefinitely until a fresh detection
    ball_proximity_side_lock: bool = True
    side_switch_ball_x_sep: float = 0.20  # release side hold when ball is far from that side's anchor


class BallDetectionConfig(BaseModel):
    """Hoop-guided crop inference when full-frame ball detection misses."""

    enable_hoop_region: bool = True
    hoop_region_expand: float = 3.5


class BallTrackerConfig(BaseModel):
    """Kalman gap-fill for short ball occlusions."""

    max_predict_frames: int = 8
    max_predict_frames_shot: int = 20
    shot_hoop_x_band: float = 2.5  # half-band = hoop.w * this
    min_upward_velocity: float = 0.003  # normalized y/frame (screen up = negative)


class AttemptDedupeConfig(BaseModel):
    """Post-process clusters likely duplicate triggers from one physical shot."""

    enabled: bool = True
    cluster_window_sec: float = 2.5
    max_ball_x_sep: float = 0.10


class AttemptDetectionConfig(BaseModel):
    ring_level_margin: float = 0.025
    horizontal_expand: float = 2.25
    lookback_frames: int = 8
    min_rise: float = 0.005
    rim_approach_depth: float = 0.08
    horizontal_vertical_depth: float = 0.03
    min_deep_below: float = 0.012
    shot_exit_depth: float = 0.07
    min_horizontal_convergence: float = 0.005
    velocity_window: int = 2
    cooldown_sec: float = 1.0
    dedupe: AttemptDedupeConfig = Field(default_factory=AttemptDedupeConfig)
    rim_interaction: RimInteractionConfig = Field(default_factory=RimInteractionConfig)
    hoop_tracking: HoopTrackingConfig = Field(default_factory=HoopTrackingConfig)
    ball_detection: BallDetectionConfig = Field(default_factory=BallDetectionConfig)
    ball_tracking: BallTrackerConfig = Field(default_factory=BallTrackerConfig)


class FrameHoopSnapshot(BaseModel):
    frame: int
    hoop: HoopRoi
    rim_y: float = Field(alias="rimY")

    model_config = {"populate_by_name": True}


class ManualHoopRoiFile(BaseModel):
    """JSON from hoop-marker.html or Pro Entry rim marking."""

    format: str | None = None
    source: str | None = None
    clip_key: str | None = Field(default=None, alias="clipKey")
    left_hoop: HoopRoi | None = Field(default=None, alias="leftHoop")
    right_hoop: HoopRoi | None = Field(default=None, alias="rightHoop")
    court_hoops: list[HoopRoi] = Field(default_factory=list, alias="courtHoops")
    hoop_roi: HoopRoi | None = Field(default=None, alias="hoopRoi")
    full_court: bool | None = Field(default=None, alias="fullCourt")
    court_polygon: list[CourtPoint] | None = Field(default=None, alias="courtPolygon")

    model_config = {"populate_by_name": True}


class AttemptsArtifact(BaseModel):
    video: str
    sample_fps: float = Field(alias="sampleFps")
    hoop_roi: HoopRoi = Field(alias="hoopRoi")
    rim_y: float = Field(alias="rimY")
    attempt_zone: HoopRoi = Field(alias="attemptZone")
    court_hoops: list[HoopRoi] = Field(alias="courtHoops", default_factory=list)
    full_court: bool = Field(alias="fullCourt", default=False)
    active_hoop_track: list[FrameHoopSnapshot] = Field(alias="activeHoopTrack", default_factory=list)
    hoop_source: Literal["detected", "manual"] = Field(alias="hoopSource", default="detected")
    hoop_detection_rate: float = Field(alias="hoopDetectionRate")
    ball_detection_rate: float = Field(alias="ballDetectionRate")
    ball_track: list[BallTrackPoint] = Field(alias="ballTrack", default_factory=list)
    attempts: list[ShotAttempt]

    model_config = {"populate_by_name": True}
