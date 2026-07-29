"""Pydantic models for scoreboard OCR artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreboardRegion(BaseModel):
    """Normalized bounding box for a scoreboard region (top-left origin)."""

    x: float
    y: float
    w: float
    h: float


class ScoreboardROIConfig(BaseModel):
    """JSON config from scoreboard-marker.html."""

    format: str | None = None
    source: str | None = None
    clip_key: str | None = Field(default=None, alias="clipKey")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    regions: dict[str, ScoreboardRegion]

    model_config = {"populate_by_name": True}


class ScoreboardReading(BaseModel):
    """Single frame's OCR reading from the scoreboard."""

    video_time_sec: float = Field(alias="videoTimeSec")
    game_clock: str | None = Field(default=None, alias="gameClock")
    home_score: int | None = Field(default=None, alias="homeScore")
    away_score: int | None = Field(default=None, alias="awayScore")
    shot_clock_sec: float | None = Field(default=None, alias="shotClockSec")
    quarter: int | None = None
    raw_ocr: dict[str, str] | None = Field(default=None, alias="rawOcr")

    model_config = {"populate_by_name": True}


class GameClockMapping(BaseModel):
    """Maps a game clock OCR string to video timestamp."""

    game_clock: str = Field(alias="gameClock")
    video_time_sec: float = Field(alias="videoTimeSec")

    model_config = {"populate_by_name": True}


class ScoreboardTrackArtifact(BaseModel):
    """Output artifact from scoreboard OCR processing."""

    version: str = "1.0.0"
    video_path: str = Field(alias="videoPath")
    sample_fps: float = Field(alias="sampleFps")
    total_frames: int = Field(alias="totalFrames")
    readings: list[ScoreboardReading]
    game_clock_to_video: list[GameClockMapping] = Field(alias="gameClockToVideo")

    model_config = {"populate_by_name": True}
