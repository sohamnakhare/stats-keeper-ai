#!/usr/bin/env python3
"""Draw ball and hoop bounding boxes on a video (E-BARD classes only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from detector import DEFAULT_MODEL_PATH, ObjectDetection, create_ball_hoop_detector  # noqa: E402
from schemas import DetectorConfig  # noqa: E402

CLASS_STYLES: dict[str, tuple[tuple[int, int, int], str]] = {
    "basketball": ((0, 255, 0), "ball"),  # green
    "hoop": ((0, 165, 255), "hoop"),  # orange
}


def resize_frame(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_detection(frame, detection: ObjectDetection, display_name: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    x1 = int((detection.x - detection.w / 2) * width)
    y1 = int((detection.y - detection.h / 2) * height)
    x2 = int((detection.x + detection.w / 2) * width)
    y2 = int((detection.y + detection.h / 2) * height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{display_name} {detection.conf:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y1 = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 4, label_y1 + text_h + baseline + 4), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, label_y1 + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def annotate_video(
    video_path: Path,
    output_path: Path,
    conf: float = 0.20,
    model_path: Path = DEFAULT_MODEL_PATH,
    ball_model_path: Path | None = None,
    max_width: int = 1280,
) -> dict[str, float | int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python worker/scripts/download_model.py"
        )
    if ball_model_path is not None and not ball_model_path.exists():
        raise FileNotFoundError(f"Ball model not found at {ball_model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0

    detector = create_ball_hoop_detector(
        model_path=model_path,
        ball_model_path=ball_model_path,
        config=DetectorConfig(conf=conf, ball_conf=conf),
    )
    class_names = list(CLASS_STYLES.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    total = 0
    ball_detected = 0
    hoop_detected = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_frame(frame, max_width)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    native_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {output_path}")

            total += 1
            detections = detector.detect_objects(frame, class_names)

            ball = detections.get("basketball")
            if ball is not None:
                ball_detected += 1
                color, label = CLASS_STYLES["basketball"]
                draw_detection(frame, ball, label, color)

            hoop = detections.get("hoop")
            if hoop is not None:
                hoop_detected += 1
                color, label = CLASS_STYLES["hoop"]
                draw_detection(frame, hoop, label, color)

            writer.write(frame)

            if total % 30 == 0:
                print(f"Processed {total} frames...", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        "frames_total": total,
        "ball_frames_detected": ball_detected,
        "hoop_frames_detected": hoop_detected,
        "ball_detection_rate": round(ball_detected / total, 4) if total else 0.0,
        "hoop_detection_rate": round(hoop_detected / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ball and hoop boxes on a video")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "ball_hoop_boxes.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="E-BARD weights for hoop")
    parser.add_argument(
        "--ball-model",
        type=Path,
        default=None,
        help="Fine-tuned ball-only weights (hybrid with --model for hoop)",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Max frame width for detection")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    metrics = annotate_video(
        video_path=args.video,
        output_path=args.output,
        conf=args.conf,
        model_path=args.model,
        ball_model_path=args.ball_model,
        max_width=args.max_width,
    )

    print(f"Wrote {args.output}")
    print(f"Frames: {metrics['frames_total']}")
    print(f"Ball detection rate: {metrics['ball_detection_rate']:.1%}")
    print(f"Hoop detection rate: {metrics['hoop_detection_rate']:.1%}")


if __name__ == "__main__":
    main()
