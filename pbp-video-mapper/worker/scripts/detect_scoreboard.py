#!/usr/bin/env python3
"""Detect scoreboard data from video using OCR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from worker.scoreboard_detector import (  # noqa: E402
    process_video_to_scoreboard_track,
    save_scoreboard_track,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract scoreboard data from video using OCR"
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--roi",
        type=Path,
        required=True,
        help="Scoreboard ROI config JSON (from scoreboard-marker.html)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "scoreboard_track.json",
        help="Output scoreboard track JSON path",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Sample frames per second (default: 1.0)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU for EasyOCR (use CPU)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing/interpolation of OCR results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw OCR text in output for debugging",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if not args.roi.exists():
        print(f"ROI config not found: {args.roi}", file=sys.stderr)
        sys.exit(1)

    def on_progress(pct: int) -> None:
        print(f"Progress: {pct}%", flush=True)

    print(f"Processing video: {args.video}")
    print(f"ROI config: {args.roi}")
    print(f"Sample FPS: {args.fps}")
    print(f"GPU: {'disabled' if args.no_gpu else 'enabled'}")
    print()

    artifact = process_video_to_scoreboard_track(
        video_path=args.video,
        roi_config_path=args.roi,
        sample_fps=args.fps,
        gpu=not args.no_gpu,
        smooth=not args.no_smooth,
        include_raw_ocr=args.debug,
        on_progress=on_progress,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_scoreboard_track(artifact, args.output)

    print()
    print(f"Wrote {args.output}")
    print(f"Total frames processed: {artifact.total_frames}")
    print(
        f"Readings with game clock: "
        f"{sum(1 for r in artifact.readings if r.game_clock is not None)}"
    )
    print(f"Game clock mappings: {len(artifact.game_clock_to_video)}")

    if artifact.readings:
        first = artifact.readings[0]
        last = artifact.readings[-1]
        print()
        print("First reading:")
        print(f"  Video time: {first.video_time_sec:.1f}s")
        if first.game_clock is not None:
            print(f"  Game clock: {first.game_clock}")
        if first.home_score is not None:
            print(f"  Home score: {first.home_score}")
        if first.away_score is not None:
            print(f"  Away score: {first.away_score}")
        if first.quarter is not None:
            print(f"  Quarter: {first.quarter}")

        print()
        print("Last reading:")
        print(f"  Video time: {last.video_time_sec:.1f}s")
        if last.game_clock is not None:
            print(f"  Game clock: {last.game_clock}")
        if last.home_score is not None:
            print(f"  Home score: {last.home_score}")
        if last.away_score is not None:
            print(f"  Away score: {last.away_score}")
        if last.quarter is not None:
            print(f"  Quarter: {last.quarter}")


if __name__ == "__main__":
    main()
