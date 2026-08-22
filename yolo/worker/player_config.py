"""Pipeline configuration for player movement tracking.

Every stage is configurable through a single JSON file (--config) whose
sections mirror the models below; individual CLI flags override file values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_DEFAULT_DEVICE = "mps" if sys.platform == "darwin" else "cuda:0"
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "BODD_yolov8n_0001.pt")
_DEFAULT_POSE_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "yolov8n-pose.pt")


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
    # Blend new foot pixel into the track's previous foot (1 = no smoothing).
    foot_ema_alpha: float = Field(default=0.3, alias="footEmaAlpha")

    model_config = {"populate_by_name": True}


class TrackingConfig(BaseModel):
    """Tracker parameters (written to a YAML for ultralytics `model.track`).

    Default is Deep OC-SORT with ReID + camera-motion compensation. Set
    tracker_type to "bytetrack" to A/B against the previous motion-only tracker.
    """

    tracker_type: Literal["deepocsort", "bytetrack"] = Field(
        default="deepocsort", alias="trackerType"
    )
    track_high_thresh: float = Field(default=0.5, alias="trackHighThresh")
    track_low_thresh: float = Field(default=0.1, alias="trackLowThresh")
    new_track_thresh: float = Field(default=0.6, alias="newTrackThresh")
    # Frames (at sample fps) a lost track is kept alive; higher survives longer occlusions.
    track_buffer: int = Field(default=90, alias="trackBuffer")
    match_thresh: float = Field(default=0.8, alias="matchThresh")
    fuse_score: bool = Field(default=True, alias="fuseScore")
    min_track_points: int = Field(default=15, alias="minTrackPoints")
    # Deep OC-SORT: appearance ReID (model "auto" reuses YOLO backbone features).
    with_reid: bool = Field(default=True, alias="withReid")
    reid_model: str = Field(default="auto", alias="reidModel")
    proximity_thresh: float = Field(default=0.5, alias="proximityThresh")
    # Stricter than Ultralytics 0.9 default: same-team kits look alike.
    appearance_thresh: float = Field(default=0.92, alias="appearanceThresh")
    alpha_fixed_emb: float = Field(default=0.95, alias="alphaFixedEmb")
    # Sideline video usually pans; Ultralytics default is "none".
    gmc_method: Literal["sparseOptFlow", "orb", "sift", "ecc", "none"] = Field(
        default="sparseOptFlow", alias="gmcMethod"
    )
    # OC-SORT observation-centric motion (inherited by Deep OC-SORT).
    delta_t: int = Field(default=3, alias="deltaT")
    inertia: float = 0.3
    use_byte: bool = Field(default=True, alias="useByte")

    model_config = {"populate_by_name": True}


class HomographyConfig(BaseModel):
    """Pixel -> court-meter projection."""

    calibration_file: str | None = Field(default=None, alias="calibrationFile")
    ransac_reproj_threshold: float = Field(default=3.0, alias="ransacReprojThreshold")
    # Points projected further than this outside the court are dropped (meters).
    court_margin: float = Field(default=0.3, alias="courtMargin")
    # If true, drop points outside the court rectangle (do not snap to the edge).
    clamp_to_court: bool = Field(default=False, alias="clampToCourt")
    court_detect_file: str | None = Field(default=None, alias="courtDetectFile")
    # Smooth paint quads over time; only replace H when the EMA drifts this far
    # (normalized image units). Stops per-frame keypoint jitter from pulsing Y.
    paint_ema_alpha: float = Field(default=0.25, alias="paintEmaAlpha")
    paint_adopt_thresh: float = Field(default=0.012, alias="paintAdoptThresh")
    # Consecutive court-detect frames above threshold before H starts ramping.
    paint_adopt_hysteresis: int = Field(default=3, alias="paintAdoptHysteresis")
    # Court-detect frames to blend paint quad (and H) after a confirmed adopt.
    paint_ramp_frames: int = Field(default=8, alias="paintRampFrames")
    # Drift >= thresh * this starts a short ramp immediately (real zoom/pan).
    paint_fast_adopt_mult: float = Field(default=3.0, alias="paintFastAdoptMult")
    # EMA on projected court meters per track (1 = off). Hides residual H steps.
    court_xy_ema_alpha: float = Field(default=0.4, alias="courtXyEmaAlpha")

    model_config = {"populate_by_name": True}


class TeamConfig(BaseModel):
    """Team classification from torso crops (jersey color by default; optional DINOv2)."""

    enabled: bool = True
    # "color" = HSV/Lab K-means + kit book; "dino" = DINOv2-small on torso crops.
    method: Literal["dino", "color"] = "color"
    dino_model: str = Field(default="facebook/dinov2-small", alias="dinoModel")
    kit_book_file: str | None = Field(default=None, alias="kitBookFile")
    # Kit-book nearest match is rejected below this cosine similarity (DINO path).
    min_cosine: float = Field(default=0.35, alias="minCosine")
    # Best torso crops kept per track and batch-encoded at finalize (DINO only).
    max_crops_per_track: int = Field(default=8, alias="maxCropsPerTrack")
    # Torso crop within the player bbox (fractions of bbox height/width).
    # Tight on the chest so floor, head, and shorts do not dominate jersey color.
    torso_top: float = Field(default=0.18, alias="torsoTop")
    torso_bottom: float = Field(default=0.48, alias="torsoBottom")
    torso_x_inset: float = Field(default=0.22, alias="torsoXInset")
    kmeans_clusters: int = Field(default=3, alias="kmeansClusters")
    min_crop_height: int = Field(default=40, alias="minCropHeight")
    # Frames between color samples per track (1 = every sampled frame).
    sample_every: int = Field(default=2, alias="sampleEvery")
    min_samples_per_track: int = Field(default=3, alias="minSamplesPerTrack")
    # COCO YOLO-pose for shoulder/hip jersey crops (rectangle crop if pose misses).
    use_pose: bool = Field(default=True, alias="usePose")
    pose_model: str = Field(default=_DEFAULT_POSE_MODEL_PATH, alias="poseModel")
    pose_min_conf: float = Field(default=0.4, alias="poseMinConf")
    pose_min_keypoints: int = Field(default=3, alias="poseMinKeypoints")
    # Winning Lab cluster must hold at least this share of a track's color samples.
    min_vote_share: float = Field(default=0.6, alias="minVoteShare")
    # Tracks below this median HSV saturation (0-255) may be labeled referee.
    referee_max_sat: float = Field(default=28.0, alias="refereeMaxSat")
    # Min Lab gap (second - best) when assigning a track to the nearest kit.
    kit_lab_margin: float = Field(default=12.0, alias="kitLabMargin")
    # If k=2 Lab centroids are closer than this, re-split on value (navy vs black).
    close_centroid_lab: float = Field(default=25.0, alias="closeCentroidLab")

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
