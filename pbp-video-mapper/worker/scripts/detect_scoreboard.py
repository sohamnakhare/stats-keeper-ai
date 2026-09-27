#!/usr/bin/env python3
"""Detect scoreboard data from video using OCR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from worker.download_video import download_video  # noqa: E402
from worker.scorebug_api import fetch_scorebug_video  # noqa: E402
from worker.scoreboard_detector import (  # noqa: E402
    ocr_scorebug_track,
    process_video_to_scoreboard_track,
    save_scoreboard_track,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract scoreboard data from video using OCR"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video path")
    source.add_argument(
        "--url",
        help="YouTube link or direct .mp4, .webm, .mkv, or .mov URL",
    )
    source.add_argument(
        "--id",
        help=(
            "Scorebug video id; loads the URL and markings from "
            "$SCOREBUG_API_BASE (default http://localhost:3000)"
        ),
    )
    parser.add_argument(
        "--roi",
        type=Path,
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

    api_video = None
    if args.id:
        try:
            api_video = fetch_scorebug_video(args.id)
            video_path = download_video(api_video.video_url)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    elif args.roi is None:
        parser.error("--roi is required unless --id is set")
    elif args.url:
        try:
            video_path = download_video(args.url)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    else:
        video_path = args.video
        if not video_path.exists():
            print(f"Video not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    if api_video is None and not args.roi.exists():
        print(f"ROI config not found: {args.roi}", file=sys.stderr)
        sys.exit(1)

    def on_progress(pct: int) -> None:
        print(f"Progress: {pct}%", flush=True)

    print(f"Processing video: {video_path}")
    if api_video is not None:
        print(f"Scorebug video id: {args.id}")
        print(f"Fields: {', '.join(api_video.fields)}")
    else:
        print(f"ROI config: {args.roi}")
    print(f"Sample FPS: {args.fps}")
    print(f"GPU: {'disabled' if args.no_gpu else 'enabled'}")
    print()

    if api_video is not None:
        artifact = ocr_scorebug_track(
            video_path=video_path,
            scorebug=api_video.scorebug,
            regions=api_video.fields,
            sample_fps=args.fps,
            gpu=not args.no_gpu,
            smooth=not args.no_smooth,
            include_raw_ocr=args.debug,
            on_progress=on_progress,
        )
    else:
        artifact = process_video_to_scoreboard_track(
            video_path=video_path,
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
