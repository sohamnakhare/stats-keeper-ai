"""Scoreboard detection and OCR processing for video files."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2

from .extract_frames import iter_sampled_frames
from .ocr_reader import ScoreboardOCRReader, game_clock_candidates
from .schemas import (
    GameClockMapping,
    ScoreboardReading,
    ScoreboardRegion,
    ScoreboardROIConfig,
    ScoreboardTrackArtifact,
)


def load_roi_config(config_path: Path | str) -> ScoreboardROIConfig:
    """Load scoreboard ROI config from JSON file."""
    path = Path(config_path)
    with path.open() as f:
        data = json.load(f)
    return ScoreboardROIConfig.model_validate(data)


def remap_region_into_scorebug(
    inner: ScoreboardRegion,
    scorebug: ScoreboardRegion,
) -> ScoreboardRegion:
    """Convert a full-frame box into coordinates relative to the scorebug crop."""
    return ScoreboardRegion(
        x=(inner.x - scorebug.x) / scorebug.w,
        y=(inner.y - scorebug.y) / scorebug.h,
        w=inner.w / scorebug.w,
        h=inner.h / scorebug.h,
    )


def _region_inside_scorebug(
    inner: ScoreboardRegion,
    scorebug: ScoreboardRegion,
    eps: float = 1e-6,
) -> bool:
    return (
        inner.x >= scorebug.x - eps
        and inner.y >= scorebug.y - eps
        and inner.x + inner.w <= scorebug.x + scorebug.w + eps
        and inner.y + inner.h <= scorebug.y + scorebug.h + eps
    )


def prepare_scorebug_regions(
    regions: dict[str, ScoreboardRegion],
) -> tuple[ScoreboardRegion, dict[str, ScoreboardRegion]]:
    """Split the scorebug crop from inner OCR regions.

    Inner boxes stay in the returned dict, remapped into the scorebug crop.
    """
    if "scorebug" not in regions:
        raise ValueError("ROI config must include 'scorebug' region")

    scorebug = regions["scorebug"]
    if scorebug.w <= 0 or scorebug.h <= 0:
        raise ValueError("scorebug region must have positive width and height")

    if "game_clock" not in regions:
        raise ValueError("ROI config must include 'game_clock' region")

    inner: dict[str, ScoreboardRegion] = {}
    for name, region in regions.items():
        if name == "scorebug":
            continue
        if region.w <= 0 or region.h <= 0:
            raise ValueError(f"Region '{name}' must have positive width and height")
        if not _region_inside_scorebug(region, scorebug):
            raise ValueError(f"Region '{name}' must lie inside the scorebug")
        inner[name] = remap_region_into_scorebug(region, scorebug)
    return scorebug, inner


def _clock_to_seconds(clock: str | None) -> int | None:
    if clock is None:
        return None
    parts = clock.split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    if seconds > 59:
        return None
    return minutes * 60 + seconds


def _seconds_to_clock(total_seconds: int) -> str:
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def _candidates_for_reading(reading: ScoreboardReading) -> list[int]:
    """Seconds-valued clock candidates from raw OCR (preferred) or game_clock."""
    raw_text = None
    if reading.raw_ocr:
        raw_text = reading.raw_ocr.get("game_clock")
    texts = game_clock_candidates(raw_text)
    if not texts and reading.game_clock:
        texts = [reading.game_clock]

    seconds: list[int] = []
    for text in texts:
        value = _clock_to_seconds(text)
        if value is not None and value not in seconds:
            seconds.append(value)
    return seconds


def _smooth_readings(
    readings: list[ScoreboardReading],
    slack: float = 2.5,
) -> list[ScoreboardReading]:
    """Reconstruct game clock from the longest consistent countdown tracks.

    Avoids locking onto a single bad seed (e.g. 7700 -> 7:00) and holding it forever.
    """
    if not readings:
        return readings

    observations: list[list[int]] = [_candidates_for_reading(r) for r in readings]
    n = len(readings)
    assigned: list[int | None] = [None] * n
    used: list[bool] = [False] * n

    def extract_best_track() -> list[tuple[int, int]]:
        """Return [(frame_index, clock_sec), ...] for the best unused track."""
        best_len: list[list[int]] = []
        parent: list[list[tuple[int, int] | None]] = []

        for i in range(n):
            cands = observations[i]
            best_len.append([1] * len(cands))
            parent.append([None] * len(cands))
            if used[i] or not cands:
                continue

            for k, curr in enumerate(cands):
                for j in range(i):
                    if used[j] or not observations[j]:
                        continue
                    dt = max(0.0, readings[i].video_time_sec - readings[j].video_time_sec)
                    # Running links stay local; paused links may span longer.
                    if dt > 45.0:
                        continue
                    for pk, prev in enumerate(observations[j]):
                        drop = prev - curr
                        if curr > prev:
                            continue
                        paused_ok = drop <= slack and dt <= 45.0
                        running_ok = abs(drop - dt) <= slack or drop <= dt + slack
                        if not (paused_ok or (running_ok and dt <= 12.0)):
                            continue
                        cand_len = best_len[j][pk] + 1
                        if cand_len > best_len[i][k]:
                            best_len[i][k] = cand_len
                            parent[i][k] = (j, pk)

        best_i, best_k, best_track_len = -1, -1, 0
        for i in range(n):
            if used[i]:
                continue
            for k, length in enumerate(best_len[i]):
                if length > best_track_len:
                    best_track_len = length
                    best_i, best_k = i, k

        if best_i < 0 or best_track_len < 3:
            return []

        path: list[tuple[int, int]] = []
        i, k = best_i, best_k
        while True:
            path.append((i, observations[i][k]))
            link = parent[i][k]
            if link is None:
                break
            i, k = link
        path.reverse()
        return path

    # Extract multiple countdown tracks (e.g. before/after scoreboard gaps).
    while True:
        path = extract_best_track()
        if not path:
            break
        for i, sec in path:
            assigned[i] = sec
            used[i] = True

    # Fill gaps between known points on the timeline by interpolation.
    filled: list[int | None] = list(assigned)
    known = [(i, sec) for i, sec in enumerate(assigned) if sec is not None]
    for idx in range(len(known) - 1):
        i0, s0 = known[idx]
        i1, s1 = known[idx + 1]
        if i1 - i0 <= 1:
            continue
        t0 = readings[i0].video_time_sec
        t1 = readings[i1].video_time_sec
        dt = t1 - t0
        drop = s0 - s1
        # Only interpolate when the segment looks like continuous countdown/pause.
        if drop < 0 or drop > dt + slack + 1:
            continue
        span = max(dt, 1e-6)
        for i in range(i0 + 1, i1):
            progress = (readings[i].video_time_sec - t0) / span
            expected = int(round(s0 + (s1 - s0) * progress))
            # Prefer a local OCR candidate near the expected countdown value.
            local = observations[i]
            if local:
                best = min(local, key=lambda c: abs(c - expected))
                if abs(best - expected) <= slack:
                    filled[i] = best
                    continue
            filled[i] = expected

    # Hold last known only for very short gaps (not forever).
    last: int | None = None
    hold = 0
    max_hold = 2
    for i in range(n):
        if filled[i] is not None:
            last = filled[i]
            hold = 0
        elif last is not None and hold < max_hold:
            filled[i] = last
            hold += 1

    smoothed: list[ScoreboardReading] = []
    for i, reading in enumerate(readings):
        clock = _seconds_to_clock(filled[i]) if filled[i] is not None else None
        smoothed.append(
            ScoreboardReading(
                video_time_sec=reading.video_time_sec,
                game_clock=clock,
                home_score=reading.home_score,
                away_score=reading.away_score,
                shot_clock_sec=reading.shot_clock_sec,
                quarter=reading.quarter,
                raw_ocr=reading.raw_ocr,
            )
        )
    return smoothed


def _build_clock_mapping(
    readings: list[ScoreboardReading],
) -> list[GameClockMapping]:
    """Build reverse mapping from canonical game clock to video time."""
    clock_to_video: dict[str, float] = {}

    for reading in readings:
        if reading.game_clock is None:
            continue
        if reading.game_clock not in clock_to_video:
            clock_to_video[reading.game_clock] = reading.video_time_sec

    def sort_key(item: tuple[str, float]) -> tuple[int, float]:
        clock, video_t = item
        sec = _clock_to_seconds(clock)
        return (-(sec if sec is not None else -1), video_t)

    return [
        GameClockMapping(game_clock=clock, video_time_sec=video_time)
        for clock, video_time in sorted(clock_to_video.items(), key=sort_key)
    ]


def process_video_to_scoreboard_track(
    video_path: Path | str,
    roi_config_path: Path | str,
    sample_fps: float = 1.0,
    gpu: bool = True,
    smooth: bool = True,
    include_raw_ocr: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> ScoreboardTrackArtifact:
    """Process a video and extract scoreboard data using OCR."""
    video_path = Path(video_path)
    roi_config = load_roi_config(roi_config_path)
    scorebug, regions = prepare_scorebug_regions(dict(roi_config.regions))
    return ocr_scorebug_track(
        video_path=video_path,
        scorebug=scorebug,
        regions=regions,
        sample_fps=sample_fps,
        gpu=gpu,
        smooth=smooth,
        include_raw_ocr=include_raw_ocr,
        on_progress=on_progress,
    )


def ocr_scorebug_track(
    video_path: Path | str,
    scorebug: ScoreboardRegion,
    regions: dict[str, ScoreboardRegion],
    sample_fps: float = 1.0,
    gpu: bool = True,
    smooth: bool = True,
    include_raw_ocr: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> ScoreboardTrackArtifact:
    """OCR crop-relative regions inside an already chosen scorebug crop."""
    video_path = Path(video_path)
    reader = ScoreboardOCRReader(gpu=gpu)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / native_fps if native_fps > 0 else 0
    cap.release()

    expected_samples = int(duration * sample_fps) if duration > 0 else 0

    readings: list[ScoreboardReading] = []
    sample_count = 0

    for frame in iter_sampled_frames(str(video_path), sample_fps=sample_fps):
        cropped = reader.crop_region(frame.bgr, scorebug)
        parsed = reader.read_and_parse(cropped, regions)
        raw = parsed.get("raw_ocr") or {}
        candidates = game_clock_candidates(raw.get("game_clock"))

        reading = ScoreboardReading(
            video_time_sec=round(frame.t_sec, 3),
            # Temporary best guess; track reconstruction overwrites when smooth=True.
            game_clock=candidates[0] if candidates else None,
            home_score=parsed.get("home_score"),
            away_score=parsed.get("away_score"),
            shot_clock_sec=parsed.get("shot_clock_sec"),
            quarter=parsed.get("quarter"),
            # Always keep raw OCR internally for track reconstruction.
            raw_ocr=raw,
        )
        readings.append(reading)

        sample_count += 1
        if on_progress and expected_samples > 0:
            progress = min(100, int(sample_count / expected_samples * 100))
            on_progress(progress)

    if smooth:
        readings = _smooth_readings(readings)

    if not include_raw_ocr:
        readings = [
            ScoreboardReading(
                video_time_sec=r.video_time_sec,
                game_clock=r.game_clock,
                home_score=r.home_score,
                away_score=r.away_score,
                shot_clock_sec=r.shot_clock_sec,
                quarter=r.quarter,
                raw_ocr=None,
            )
            for r in readings
        ]

    clock_mapping = _build_clock_mapping(readings)

    return ScoreboardTrackArtifact(
        version="1.0.0",
        video_path=str(video_path),
        sample_fps=sample_fps,
        total_frames=sample_count,
        readings=readings,
        game_clock_to_video=clock_mapping,
    )


def save_scoreboard_track(
    artifact: ScoreboardTrackArtifact,
    output_path: Path | str,
) -> None:
    """Save scoreboard track artifact to JSON file."""
    path = Path(output_path)
    with path.open("w") as f:
        json.dump(artifact.model_dump(by_alias=True), f, indent=2)
