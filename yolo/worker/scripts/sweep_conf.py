#!/usr/bin/env python3
"""Sweep confidence thresholds for ball detection on a clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from scripts.eval_ebard import evaluate_video  # noqa: E402
from detector import DEFAULT_MODEL_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep E-BARD conf thresholds")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--conf-values",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    )
    args = parser.parse_args()

    print(f"Video: {args.video}")
    print(f"{'conf':>6}  {'rate':>8}  {'mean_conf':>10}  pass")
    print("-" * 40)
    for conf in args.conf_values:
        metrics = evaluate_video(
            args.video,
            sample_fps=args.sample_fps,
            conf=conf,
            model_path=DEFAULT_MODEL_PATH,
        )
        print(
            f"{conf:6.2f}  {metrics['detection_rate']:7.1%}  "
            f"{metrics['mean_conf_when_detected']:10.3f}  "
            f"{'yes' if metrics['pass_detection_rate'] else 'no'}"
        )


if __name__ == "__main__":
    main()
