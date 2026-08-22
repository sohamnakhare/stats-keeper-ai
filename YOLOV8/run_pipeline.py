#!/usr/bin/env python3
"""Run the basketball play-diagram pipeline.

Examples:
  python run_pipeline.py --input ../yolo/new.mp4 --stage detect --output-dir output
  python run_pipeline.py --input ../yolo/new.mp4 --calibration ../yolo/court_calibration.json --output output/play_diagram.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.calibrate import default_calibration_path, run_calibrate  # noqa: E402
from pipeline.detect import run_detect  # noqa: E402
from pipeline.events import run_events  # noqa: E402
from pipeline.render import run_render  # noqa: E402
from pipeline.track import run_track  # noqa: E402
from pipeline.trajectories import run_trajectories  # noqa: E402

STAGES = ("detect", "track", "calibrate", "trajectories", "events", "render")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basketball clip → top-down play diagram")
    p.add_argument("--input", required=True, type=Path, help="Input video clip (mp4)")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Diagram image path (PNG or SVG). Default: <output-dir>/play_diagram.png",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for JSON artifacts and annotated video",
    )
    p.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="all",
        help="Run through this stage (inclusive). Default: all",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="court-calibration-v1 JSON from yolo/court-marker.html",
    )
    p.add_argument("--model", type=Path, default=None, help="YOLO weights (E-BARD by default)")
    p.add_argument("--device", default=None, help="mps / cuda:0 / cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--sample-fps", type=float, default=10.0)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--imgsz", type=int, default=704)
    p.add_argument("--no-annotate", action="store_true", help="Skip Stage 1 annotated MP4")
    p.add_argument("--referees", action="store_true", help="Also detect/track referees")
    return p.parse_args()


def _wanted(stage: str) -> set[str]:
    if stage == "all":
        return set(STAGES)
    return set(STAGES[: STAGES.index(stage) + 1])


def main() -> None:
    args = _parse_args()
    video = args.input
    if not video.exists():
        raise SystemExit(f"Input video not found: {video}")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    diagram = args.output or (out_dir / "play_diagram.png")
    wanted = _wanted(args.stage)
    common = dict(
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        sample_fps=args.sample_fps,
        max_width=args.max_width,
        detect_referees=args.referees,
    )

    if "detect" in wanted:
        run_detect(
            video,
            out_dir,
            annotate=not args.no_annotate,
            **common,
        )
    if "track" in wanted:
        run_track(video, out_dir, **common)
    if "calibrate" in wanted:
        cal = args.calibration or default_calibration_path(video)
        run_calibrate(out_dir, calibration_path=cal, video_path=video)
    if "trajectories" in wanted:
        run_trajectories(out_dir)
    if "events" in wanted:
        run_events(out_dir)
    if "render" in wanted:
        run_render(out_dir, diagram)


if __name__ == "__main__":
    main()
