#!/usr/bin/env python3
"""Detect wheelchair shot attempts from ball track + hoop ROI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from wheelchair_attempt_detector import (  # noqa: E402
    load_court_polygon,
    process_video_to_attempts,
    rim_crossing_band,
)
from detector import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_MODEL_PATH,
    EbardDetector,
    HybridBallHoopDetector,
    ObjectDetection,
    create_ball_hoop_detector,
    resolve_inference_device,
)
from schemas import (  # noqa: E402
    AttemptDetectionConfig,
    AttemptsArtifact,
    BallTrackPoint,
    DetectorConfig,
    HoopRoi,
)

def _roi_to_pixels(roi: HoopRoi, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int((roi.x - roi.w / 2) * width)
    y1 = int((roi.y - roi.h / 2) * height)
    x2 = int((roi.x + roi.w / 2) * width)
    y2 = int((roi.y + roi.h / 2) * height)
    return x1, y1, x2, y2


def _draw_roi(frame, roi: HoopRoi, color: tuple[int, int, int], label: str) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _roi_to_pixels(roi, width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_detection(frame, detection: ObjectDetection, color: tuple[int, int, int], label: str) -> None:
    height, width = frame.shape[:2]
    x1 = int((detection.x - detection.w / 2) * width)
    y1 = int((detection.y - detection.h / 2) * height)
    x2 = int((detection.x + detection.w / 2) * width)
    y2 = int((detection.y + detection.h / 2) * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_ball_track_point(
    frame,
    point: BallTrackPoint,
    width: int,
    height: int,
) -> None:
    px = int(point.x * width)
    py = int(point.y * height)
    if point.detected:
        if point.w is not None and point.h is not None:
            x1 = int((point.x - point.w / 2) * width)
            y1 = int((point.y - point.h / 2) * height)
            x2 = int((point.x + point.w / 2) * width)
            y2 = int((point.y + point.h / 2) * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_part = f"{point.conf:.2f}" if point.conf is not None else "?"
            label = f"ball {conf_part}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(frame, (px, py), 6, (0, 255, 0), 2)
    elif point.predicted:
        cv2.circle(frame, (px, py), 8, (0, 255, 255), 2)
        cv2.putText(
            frame,
            "pred",
            (px + 10, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _resize_frame(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _draw_court_polygon(
    frame,
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> None:
    if len(polygon) < 3:
        return
    pts = [(int(x * width), int(y * height)) for x, y in polygon]
    pts = np.array(pts, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 180, 80))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 100), thickness=2)


def visualize_attempts(
    video_path: Path,
    artifact: AttemptsArtifact,
    output_path: Path,
    ball_conf: float = 0.05,
    hoop_conf: float = 0.60,
    model_path: Path = DEFAULT_MODEL_PATH,
    ball_model_path: Path | None = None,
    device: str = DEFAULT_DEVICE,
    max_width: int = 1280,
    hoop_roi_path: Path | None = None,
) -> None:
    attempt_frames = {a.frame for a in artifact.attempts}
    track_by_frame = {snap.frame: snap for snap in artifact.active_hoop_track}
    ball_by_frame = {point.frame: point for point in artifact.ball_track}
    court_hoops = artifact.court_hoops or [artifact.hoop_roi]
    court_polygon = load_court_polygon(hoop_roi_path) if hoop_roi_path is not None else None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or artifact.sample_fps
    if native_fps <= 0:
        native_fps = artifact.sample_fps

    frame_interval = max(1, int(round(native_fps / max(artifact.sample_fps, 0.1))))
    sample_idx = 0
    frame_idx = 0
    writer: cv2.VideoWriter | None = None
    hoop_detector: EbardDetector | HybridBallHoopDetector | None = None
    if artifact.hoop_source == "detected":
        hoop_detector = create_ball_hoop_detector(
            model_path=model_path,
            ball_model_path=ball_model_path,
            config=DetectorConfig(ball_conf=ball_conf, hoop_conf=hoop_conf, device=device),
            court_polygon=court_polygon,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            is_sample = frame_idx % frame_interval == 0
            if is_sample:
                height, width = frame.shape[:2]
                if court_polygon is not None:
                    _draw_court_polygon(frame, court_polygon, width, height)
                ball_point = ball_by_frame.get(sample_idx)
                if ball_point is not None:
                    _draw_ball_track_point(frame, ball_point, width, height)

                if artifact.hoop_source == "detected" and hoop_detector is not None:
                    detections = hoop_detector.detect_objects(frame, ["hoop"])
                    hoop = detections.get("hoop")
                    if hoop is not None:
                        _draw_detection(frame, hoop, (0, 165, 255), "hoop")

                snap = track_by_frame.get(sample_idx)
                if snap is not None:
                    active_hoop = snap.hoop
                    active_rim_y = snap.rim_y
                else:
                    active_hoop = artifact.hoop_roi
                    active_rim_y = artifact.rim_y

                for idx, court_hoop in enumerate(court_hoops):
                    label = "court" if len(court_hoops) == 1 else f"court{idx + 1}"
                    _draw_roi(frame, court_hoop, (128, 128, 128), label)

                _draw_roi(frame, active_hoop, (0, 165, 255), "active hoop")
                active_zone = rim_crossing_band(
                    active_hoop,
                    AttemptDetectionConfig().horizontal_expand,
                )
                _draw_roi(frame, active_zone, (255, 0, 255), "rim band")
                height, width = frame.shape[:2]
                rim_px = int(active_rim_y * height)
                cv2.line(frame, (0, rim_px), (width, rim_px), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    "rim",
                    (8, max(rim_px - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if sample_idx in attempt_frames:
                    height, width = frame.shape[:2]
                    frame_attempts = [a for a in artifact.attempts if a.frame == sample_idx]
                    has_deduped = any(a.deduped for a in frame_attempts)
                    label = "ATTEMPT (deduped)" if has_deduped else "ATTEMPT"
                    color = (0, 165, 255) if has_deduped else (0, 0, 255)
                    cv2.putText(
                        frame,
                        label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3,
                        cv2.LINE_AA,
                    )
                    for attempt in frame_attempts:
                        cv2.circle(
                            frame,
                            (int(attempt.ball_pos.x * width), int(attempt.ball_pos.y * height)),
                            8,
                            color,
                            -1,
                        )
                sample_idx += 1

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect wheelchair shot attempts (full-court, left-camera)"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "attempts.json",
        help="Output attempts JSON path",
    )
    parser.add_argument("--ball-conf", type=float, default=0.05, help="Min ball detection confidence")
    parser.add_argument("--hoop-conf", type=float, default=0.60, help="Min hoop detection confidence")
    parser.add_argument("--sample-fps", type=float, default=20.0)
    parser.add_argument("--cooldown", type=float, default=1.0, help="Seconds between attempts")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="E-BARD weights for hoop")
    parser.add_argument(
        "--ball-model",
        type=Path,
        default=None,
        help="Fine-tuned ball-only weights (hybrid with --model for hoop)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Inference device (mps on Mac, cuda:0 on NVIDIA). CPU is not allowed.",
    )
    parser.add_argument(
        "--hoop-roi",
        type=Path,
        default=None,
        help="hoop_roi.json from hoop-marker.html (manual hoops and/or courtPolygon)",
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        help="Optional output MP4 with attempt markers",
    )
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
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

    config = AttemptDetectionConfig(cooldown_sec=args.cooldown)

    try:
        inference_device = resolve_inference_device(args.device)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print(f"Inference device: {inference_device}", flush=True)

    def on_progress(pct: int) -> None:
        print(f"Progress: {pct}%", flush=True)

    artifact = process_video_to_attempts(
        video_path=args.video,
        sample_fps=args.sample_fps,
        ball_conf=args.ball_conf,
        hoop_conf=args.hoop_conf,
        model_path=args.model,
        ball_model_path=args.ball_model,
        config=config,
        hoop_roi_path=args.hoop_roi,
        device=inference_device,
        on_progress=on_progress,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.model_dump(by_alias=True), indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {args.output}")
    print(f"Hoop source: {artifact.hoop_source}")
    print(f"Full court mode: {artifact.full_court} ({len(artifact.court_hoops)} basket(s))")
    print(f"Ball detection rate: {artifact.ball_detection_rate:.1%}")
    if artifact.ball_track:
        predicted = sum(1 for point in artifact.ball_track if point.predicted)
        print(f"Ball track: {len(artifact.ball_track)} samples ({predicted} predicted)")
    print("Ball tracker: kalman")
    print(f"Hoop detection rate: {artifact.hoop_detection_rate:.1%}")
    print(f"Attempts detected: {len(artifact.attempts)}")
    unique = sum(1 for attempt in artifact.attempts if not attempt.deduped)
    deduped = sum(1 for attempt in artifact.attempts if attempt.deduped)
    print(f"Unique attempts (after dedupe): {unique}")
    if deduped:
        print(f"Marked deduped (review manually): {deduped}")
    for attempt in artifact.attempts:
        suffix = ""
        if attempt.deduped:
            suffix = f" [deduped -> frame {attempt.primary_frame}]"
        print(
            f"  t={attempt.t_sec:.2f}s frame={attempt.frame} "
            f"conf={attempt.confidence:.2f} speed={attempt.entry_speed:.3f}"
            f"{suffix}"
            f"{f' signal={attempt.signal_kind}' if attempt.signal_kind else ''}"
        )

    if args.visualize is not None:
        visualize_attempts(
            video_path=args.video,
            artifact=artifact,
            output_path=args.visualize,
            ball_conf=args.ball_conf,
            hoop_conf=args.hoop_conf,
            model_path=args.model,
            ball_model_path=args.ball_model,
            device=inference_device,
            max_width=args.max_width,
            hoop_roi_path=args.hoop_roi,
        )
        print(f"Wrote {args.visualize}")


if __name__ == "__main__":
    main()
