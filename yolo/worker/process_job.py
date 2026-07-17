"""Run ball detection + tracking on a video file."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from detector import BallDetector, DEFAULT_MODEL_PATH, MODEL_VERSION
from extract_frames import iter_sampled_frames
from schemas import BallTrackArtifact, DetectorConfig
from tracker import BallTracker


def process_video_to_ball_track(
    video_path: str | Path,
    game_id: str,
    job_id: str,
    sample_fps: float = 10.0,
    conf: float = 0.20,
    model_path: Path = DEFAULT_MODEL_PATH,
    on_progress: Callable[[int], None] | None = None,
) -> BallTrackArtifact:
    detector = BallDetector(
        model_path=model_path,
        config=DetectorConfig(conf=conf),
    )
    tracker = BallTracker(sample_fps=sample_fps)

    frames = list(iter_sampled_frames(str(video_path), sample_fps=sample_fps))
    total = len(frames)

    for i, sampled in enumerate(frames):
        det = detector.detect_frame(sampled.bgr)
        tracker.step(sampled.index, det)
        if on_progress and total > 0:
            on_progress(int((i + 1) / total * 100))

    effective_fps = sample_fps
    if frames:
        effective_fps = (frames[-1].index + 1) / max(frames[-1].t_sec, 0.001) if frames[-1].t_sec > 0 else sample_fps

    return BallTrackArtifact(
        gameId=game_id,
        jobId=job_id,
        sampleFps=effective_fps,
        frameCount=len(tracker.points),
        modelVersion=MODEL_VERSION,
        detectionRate=tracker.detection_rate(),
        points=tracker.points,
    )


def write_ball_track_artifact(artifact: BallTrackArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.model_dump(by_alias=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
