#!/usr/bin/env python3
"""Local ball-track worker — E-BARD detection + Kalman tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_ROOT))

from detector import DEFAULT_MODEL_PATH  # noqa: E402
from process_job import process_video_to_ball_track, write_ball_track_artifact  # noqa: E402


def run_job(
    video_path: Path,
    output_path: Path,
    game_id: str = "local-game",
    job_id: str = "local-job",
    sample_fps: float = 10.0,
    conf: float = 0.20,
) -> Path:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model missing at {DEFAULT_MODEL_PATH}. Run: python scripts/download_model.py"
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_progress(pct: int) -> None:
        print(f"Progress: {pct}%", flush=True)

    artifact = process_video_to_ball_track(
        video_path=video_path,
        game_id=game_id,
        job_id=job_id,
        sample_fps=sample_fps,
        conf=conf,
        on_progress=on_progress,
    )
    write_ball_track_artifact(artifact, output_path)
    print(f"Wrote {output_path}")
    print(f"Detection rate: {artifact.detection_rate:.1%}")
    print(f"Frames tracked: {artifact.frame_count}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ball track worker (local)")
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "ball_track.json",
        help="Output ball_track.json path",
    )
    parser.add_argument("--game-id", default="local-game")
    parser.add_argument("--job-id", default="local-job")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    run_job(
        video_path=args.video,
        output_path=args.output,
        game_id=args.game_id,
        job_id=args.job_id,
        sample_fps=args.sample_fps,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
