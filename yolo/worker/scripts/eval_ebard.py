#!/usr/bin/env python3
"""Evaluate E-BARD ball detection on a video clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from detector import BallDetector, DEFAULT_MODEL_PATH  # noqa: E402
from extract_frames import iter_sampled_frames  # noqa: E402
from schemas import DetectorConfig  # noqa: E402

PASS_DETECTION_RATE = 0.60
PASS_MEAN_CONF = 0.35
MAX_FALSE_POS_PER_MIN = 5.0


def evaluate_video(
    video_path: Path,
    sample_fps: float = 10.0,
    conf: float = 0.20,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, float | int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/download_model.py"
        )

    detector = BallDetector(model_path=model_path, config=DetectorConfig(conf=conf))
    total = 0
    detected = 0
    conf_sum = 0.0

    for sampled in iter_sampled_frames(str(video_path), sample_fps=sample_fps):
        total += 1
        det = detector.detect_frame(sampled.bgr)
        if det is not None:
            detected += 1
            conf_sum += det.conf

    detection_rate = detected / total if total else 0.0
    mean_conf = conf_sum / detected if detected else 0.0
    duration_min = (total / sample_fps) / 60.0 if sample_fps > 0 else 0.0
    false_pos_per_min = 0.0  # requires manual labels; placeholder for Phase 1

    return {
        "frames_sampled": total,
        "frames_detected": detected,
        "detection_rate": round(detection_rate, 4),
        "mean_conf_when_detected": round(mean_conf, 4),
        "false_positives_per_minute": false_pos_per_min,
        "duration_minutes": round(duration_min, 2),
        "pass_detection_rate": detection_rate >= PASS_DETECTION_RATE,
        "pass_mean_conf": mean_conf >= PASS_MEAN_CONF if detected else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E-BARD ball detection")
    parser.add_argument("video", type=Path, help="Path to MP4 video")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate_video(
        args.video,
        sample_fps=args.sample_fps,
        conf=args.conf,
        model_path=args.model,
    )

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Video: {args.video}")
        print(f"Frames sampled: {metrics['frames_sampled']}")
        print(f"Detection rate: {metrics['detection_rate']:.1%}")
        print(f"Mean conf (when detected): {metrics['mean_conf_when_detected']:.3f}")
        print(f"Pass detection rate (>= {PASS_DETECTION_RATE:.0%}): {metrics['pass_detection_rate']}")
        print(f"Pass mean conf (>= {PASS_MEAN_CONF}): {metrics['pass_mean_conf']}")

    if not metrics["pass_detection_rate"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
