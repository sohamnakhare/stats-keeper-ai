#!/usr/bin/env python3
"""Ball-only attempt visualization with on-screen ATTEMPT DETECTED flash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from attempt_detector import load_court_polygon, process_video_to_attempts  # noqa: E402
from detector import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_MODEL_PATH,
    BallDetection,
    BallDetector,
    ObjectDetection,
    resolve_inference_device,
)
from schemas import AttemptDetectionConfig, AttemptsArtifact, DetectorConfig, ShotAttempt  # noqa: E402

BALL_COLOR = (0, 255, 0)
FLASH_TEXT = "ATTEMPT DETECTED"
FLASH_BG = (0, 0, 0)
FLASH_FG = (255, 255, 255)
FLASH_BORDER = (255, 255, 255)


def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _load_artifact(path: Path) -> AttemptsArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AttemptsArtifact.model_validate(data)


def _unique_attempts(attempts: list[ShotAttempt]) -> list[ShotAttempt]:
    return [attempt for attempt in attempts if not attempt.deduped]


def _flash_active(t_sec: float, attempts: list[ShotAttempt], flash_sec: float) -> bool:
    for attempt in attempts:
        if attempt.t_sec <= t_sec <= attempt.t_sec + flash_sec:
            return True
    return False


def _draw_ball_box(frame: np.ndarray, ball: BallDetection | ObjectDetection) -> None:
    height, width = frame.shape[:2]
    x1 = int((ball.x - ball.w / 2) * width)
    y1 = int((ball.y - ball.h / 2) * height)
    x2 = int((ball.x + ball.w / 2) * width)
    y2 = int((ball.y + ball.h / 2) * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
    label = f"ball {ball.conf:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        BALL_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_attempt_flash(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(FLASH_TEXT, font, scale, thickness)
    pad_x, pad_y = 20, 12
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2
    x1 = (width - box_w) // 2
    y1 = 16
    x2 = x1 + box_w
    y2 = y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), FLASH_BG, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), FLASH_BORDER, 1)
    cv2.putText(
        frame,
        FLASH_TEXT,
        (x1 + pad_x, y1 + pad_y + text_h),
        font,
        scale,
        FLASH_FG,
        thickness,
        cv2.LINE_AA,
    )


def render_attempts_ball_video(
    video_path: Path,
    artifact: AttemptsArtifact,
    output_path: Path,
    ball_conf: float,
    model_path: Path,
    ball_model_path: Path | None,
    device: str,
    max_width: int,
    flash_sec: float,
    court_polygon: list[tuple[float, float]] | None,
) -> None:
    from court_geometry import ball_center_in_court

    attempts = _unique_attempts(artifact.attempts)
    weights = ball_model_path or model_path
    detector = BallDetector(
        model_path=weights,
        config=DetectorConfig(ball_conf=ball_conf, conf=ball_conf, device=device),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or artifact.sample_fps
    if native_fps <= 0:
        native_fps = artifact.sample_fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _resize_frame(frame, max_width)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    native_fps,
                    (width, height),
                )

            ball_det = detector.detect_frame(frame)
            if ball_det is not None:
                ball_obj = ObjectDetection(
                    label="basketball",
                    x=ball_det.x,
                    y=ball_det.y,
                    w=ball_det.w,
                    h=ball_det.h,
                    conf=ball_det.conf,
                )
                if ball_center_in_court(ball_obj, court_polygon):
                    _draw_ball_box(frame, ball_det)

            t_sec = frame_idx / native_fps
            if _flash_active(t_sec, attempts, flash_sec):
                _draw_attempt_flash(frame)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ball-only boxes with ATTEMPT DETECTED flash overlay",
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--attempts",
        type=Path,
        default=None,
        help="Existing attempts.json (if omitted, runs attempt detection first)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "attempts_ball_only.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="When detecting, optional path to save attempts JSON",
    )
    parser.add_argument("--ball-conf", type=float, default=0.05)
    parser.add_argument("--hoop-conf", type=float, default=0.60)
    parser.add_argument("--sample-fps", type=float, default=20.0)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ball-model", type=Path, default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--hoop-roi",
        type=Path,
        default=None,
        help="Optional hoop_roi.json for courtPolygon ball filtering",
    )
    parser.add_argument(
        "--flash-sec",
        type=float,
        default=2.0,
        help="Seconds to show ATTEMPT DETECTED after each unique attempt",
    )
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if args.attempts is not None and not args.attempts.exists():
        print(f"Attempts file not found: {args.attempts}", file=sys.stderr)
        sys.exit(1)
    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if args.ball_model is not None and not args.ball_model.exists():
        print(f"Ball model not found: {args.ball_model}", file=sys.stderr)
        sys.exit(1)
    if args.hoop_roi is not None and not args.hoop_roi.exists():
        print(f"Hoop ROI file not found: {args.hoop_roi}", file=sys.stderr)
        sys.exit(1)

    try:
        inference_device = resolve_inference_device(args.device)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.attempts is not None:
        artifact = _load_artifact(args.attempts)
    else:
        artifact = process_video_to_attempts(
            video_path=args.video,
            sample_fps=args.sample_fps,
            ball_conf=args.ball_conf,
            hoop_conf=args.hoop_conf,
            model_path=args.model,
            ball_model_path=args.ball_model,
            config=AttemptDetectionConfig(cooldown_sec=args.cooldown),
            hoop_roi_path=args.hoop_roi,
            device=inference_device,
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(artifact.model_dump(by_alias=True), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {args.output_json}")

    court_polygon = load_court_polygon(args.hoop_roi) if args.hoop_roi is not None else None
    render_attempts_ball_video(
        video_path=args.video,
        artifact=artifact,
        output_path=args.output,
        ball_conf=args.ball_conf,
        model_path=args.model,
        ball_model_path=args.ball_model,
        device=inference_device,
        max_width=args.max_width,
        flash_sec=args.flash_sec,
        court_polygon=court_polygon,
    )

    unique = len(_unique_attempts(artifact.attempts))
    print(f"Wrote {args.output}")
    print(f"Unique attempts flashed: {unique}")
    print(f"Flash duration: {args.flash_sec}s")


if __name__ == "__main__":
    main()
