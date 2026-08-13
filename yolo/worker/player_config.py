"""Pipeline configuration for player movement tracking.

Every stage is configurable through a single JSON file (--config) whose
sections mirror the models below; individual CLI flags override file values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_DEVICE = "mps" if sys.platform == "darwin" else "cuda:0"
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "BODD_yolov8n_0001.pt")


class DetectionConfig(BaseModel):
    """YOLO player/referee detection."""

    model_path: str = Field(default=_DEFAULT_MODEL_PATH, alias="modelPath")
    device: str = _DEFAULT_DEVICE
    imgsz: int = 704
    conf: float = 0.35
    iou: float = 0.5
    sample_fps: float = Field(default=10.0, alias="sampleFps")
    max_width: int = Field(default=1280, alias="maxWidth")
    # Off by default: referees/staff confuse team clustering and the renderer.
    detect_referees: bool = Field(default=False, alias="detectReferees")
    court_polygon_file: str | None = Field(default=None, alias="courtPolygonFile")
    # Drop tracks with fewer than this fraction of points projecting onto the court.
    min_on_court_fraction: float = Field(default=0.7, alias="minOnCourtFraction")

    model_config = {"populate_by_name": True}


class TrackingConfig(BaseModel):
    """ByteTrack parameters (written to a tracker YAML for ultralytics)."""

    track_high_thresh: float = Field(default=0.5, alias="trackHighThresh")
    track_low_thresh: float = Field(default=0.1, alias="trackLowThresh")
    new_track_thresh: float = Field(default=0.6, alias="newTrackThresh")
    # Frames (at sample fps) a lost track is kept alive; higher survives longer occlusions.
    track_buffer: int = Field(default=90, alias="trackBuffer")
    match_thresh: float = Field(default=0.8, alias="matchThresh")
    fuse_score: bool = Field(default=True, alias="fuseScore")
    min_track_points: int = Field(default=15, alias="minTrackPoints")

    model_config = {"populate_by_name": True}


class HomographyConfig(BaseModel):
    """Pixel -> court-meter projection."""

    calibration_file: str | None = Field(default=None, alias="calibrationFile")
    ransac_reproj_threshold: float = Field(default=3.0, alias="ransacReprojThreshold")
    # Points projected further than this outside the court are dropped (meters).
    court_margin: float = Field(default=0.3, alias="courtMargin")
    clamp_to_court: bool = Field(default=True, alias="clampToCourt")

    model_config = {"populate_by_name": True}


class TeamConfig(BaseModel):
    """Jersey-color team classification."""

    enabled: bool = True
    # Torso crop within the player bbox (fractions of bbox height/width).
    torso_top: float = Field(default=0.15, alias="torsoTop")
    torso_bottom: float = Field(default=0.55, alias="torsoBottom")
    torso_x_inset: float = Field(default=0.2, alias="torsoXInset")
    kmeans_clusters: int = Field(default=3, alias="kmeansClusters")
    min_crop_height: int = Field(default=40, alias="minCropHeight")
    # Frames between color samples per track (1 = every sampled frame).
    sample_every: int = Field(default=2, alias="sampleEvery")
    min_samples_per_track: int = Field(default=3, alias="minSamplesPerTrack")

    model_config = {"populate_by_name": True}


class JerseyOcrConfig(BaseModel):
    """Selective jersey-number OCR with confidence voting."""

    enabled: bool = True
    gpu: bool = False
    # Sample a candidate crop every N sampled frames per track.
    crop_interval: int = Field(default=5, alias="cropInterval")
    min_bbox_height_px: int = Field(default=80, alias="minBboxHeightPx")
    # Laplacian variance below this is considered too blurry to OCR.
    min_sharpness: float = Field(default=30.0, alias="minSharpness")
    # Number of best-scoring crops OCR'd per track.
    best_k: int = Field(default=12, alias="bestK")
    min_ocr_conf: float = Field(default=0.5, alias="minOcrConf")
    # Winning number must hold at least this share of confidence-weighted votes.
    min_vote_share: float = Field(default=0.6, alias="minVoteShare")
    min_votes: int = Field(default=2, alias="minVotes")
    upscale: float = 3.0

    model_config = {"populate_by_name": True}


class MergeConfig(BaseModel):
    """Reconnect broken tracks so players who leave sight and return stay one player.

    Matching jersey numbers are the strongest signal; without numbers (e.g.
    --no-ocr) fragments can still merge on team + jersey color + plausible gap.
    Gap filling then inserts interpolated points so the player stays on court.
    """

    enabled: bool = True
    max_gap_sec: float = Field(default=6.0, alias="maxGapSec")
    # Max court distance (meters) between one track's end and the next's start.
    max_court_distance: float = Field(default=10.0, alias="maxCourtDistance")
    # Reject merges that imply unrealistic sprint speed (m/s).
    max_speed_mps: float = Field(default=8.0, alias="maxSpeedMps")
    # Merge fragments without a jersey number using team + jersey color.
    allow_no_jersey: bool = Field(default=True, alias="allowNoJersey")
    # Max RGB distance between fragment jersey colors to consider them the same player.
    max_color_distance: float = Field(default=70.0, alias="maxColorDistance")
    # Insert interpolated points across detection gaps (occlusions, merges).
    fill_gaps: bool = Field(default=True, alias="fillGaps")
    max_fill_gap_sec: float = Field(default=6.0, alias="maxFillGapSec")

    model_config = {"populate_by_name": True}


class PipelineConfig(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    homography: HomographyConfig = Field(default_factory=HomographyConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)
    jersey_ocr: JerseyOcrConfig = Field(default_factory=JerseyOcrConfig, alias="jerseyOcr")
    merge: MergeConfig = Field(default_factory=MergeConfig)

    model_config = {"populate_by_name": True}


def load_pipeline_config(path: Path | str | None) -> PipelineConfig:
    """Load config from a JSON file; missing sections/fields use defaults."""
    if path is None:
        return PipelineConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(data)
