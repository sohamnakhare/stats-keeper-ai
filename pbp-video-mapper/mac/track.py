"""Build a scoreboard track from video using the Vision reader."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

from mac import vision_ocr
from worker.extract_frames import (
    iter_remote_frames,
    iter_sampled_frames,
    remote_duration_seconds,
)
from worker.schemas import (
    GameClockMapping,
    ScoreboardReading,
    ScoreboardRegion,
    ScoreboardTrackArtifact,
)


def _clock_to_seconds(clock: str | None) -> int | None:
    if clock is None or ":" not in clock:
        return None
    minutes, seconds = clock.split(":", 1)
    try:
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return None


def _clock_mapping(readings: list[ScoreboardReading]) -> list[GameClockMapping]:
    """First video time for each canonical game clock."""
    clock_to_video: dict[str, float] = {}
    for reading in readings:
        if reading.game_clock is None or reading.game_clock in clock_to_video:
            continue
        clock_to_video[reading.game_clock] = reading.video_time_sec

    def sort_key(item: tuple[str, float]) -> tuple[int, float]:
        clock, video_time = item
        seconds = _clock_to_seconds(clock)
        return (-(seconds if seconds is not None else -1), video_time)

    return [
        GameClockMapping(game_clock=clock, video_time_sec=video_time)
        for clock, video_time in sorted(clock_to_video.items(), key=sort_key)
    ]


def _expected_samples(video_path: Path, sample_fps: float) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if native_fps <= 0 or total_frames <= 0:
        return 0
    return int((total_frames / native_fps) * sample_fps)


def build_track(
    video_path: Path | str,
    scorebug: ScoreboardRegion,
    regions: dict[str, ScoreboardRegion],
    sample_fps: float = 1.0,
    include_raw_ocr: bool = False,
    on_progress: Callable[[int], None] | None = None,
    recognizer: vision_ocr.VisionRecognizer | None = None,
    crop_dir: Path | None = None,
    video_url: str | None = None,
) -> ScoreboardTrackArtifact:
    """Sample a video and read scorebug fields with Apple Vision."""
    path = Path(video_path)
    reader = recognizer if recognizer is not None else vision_ocr.get_recognizer()
    if video_url:
        frames = iter_remote_frames(video_url, sample_fps, scorebug)
        duration = remote_duration_seconds(video_url) or 0
        expected = int(duration * sample_fps) if duration > 0 else 0
        source = video_url
    else:
        frames = iter_sampled_frames(str(path), sample_fps=sample_fps)
        expected = _expected_samples(path, sample_fps)
        source = str(path)
    readings: list[ScoreboardReading] = []
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        logger.info("writing vision crops to %s", crop_dir)

    for sample_index, frame in enumerate(frames, start=1):
        crop = (
            frame.bgr
            if video_url
            else vision_ocr.crop_region(frame.bgr, scorebug)
        )
        raw = reader.read_regions(crop, regions)
        if crop_dir is not None:
            vision_ocr.save_crop(
                crop_dir,
                frame.t_sec,
                "scorebug",
                vision_ocr.draw_debug(crop, regions, vision_ocr.latest_observations(), raw),
            )
        parsed = vision_ocr.parse_fields(raw)
        logger.info(
            "t=%.3fs raw=%s clock=%s home=%s away=%s shot=%s quarter=%s",
            frame.t_sec,
            raw,
            parsed.get("game_clock"),
            parsed.get("home_score"),
            parsed.get("away_score"),
            parsed.get("shot_clock_sec"),
            parsed.get("quarter"),
        )
        readings.append(
            ScoreboardReading(
                video_time_sec=round(frame.t_sec, 3),
                game_clock=parsed.get("game_clock"),
                home_score=parsed.get("home_score"),
                away_score=parsed.get("away_score"),
                shot_clock_sec=parsed.get("shot_clock_sec"),
                quarter=parsed.get("quarter"),
                raw_ocr=raw if include_raw_ocr else None,
            )
        )
        if on_progress and expected > 0:
            on_progress(min(100, int(sample_index / expected * 100)))

    return ScoreboardTrackArtifact(
        version="1.0.0",
        video_path=source,
        sample_fps=sample_fps,
        total_frames=len(readings),
        readings=readings,
        game_clock_to_video=_clock_mapping(readings),
    )
