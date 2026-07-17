#!/usr/bin/env python3
"""Export sampled frames into positives/negatives with YOLO labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from attempt_detector import load_court_polygon  # noqa: E402
from detector import DEFAULT_DEVICE, DEFAULT_MODEL_PATH, EbardDetector, ObjectDetection, resolve_inference_device  # noqa: E402
from extract_frames import iter_sampled_frames  # noqa: E402
from schemas import DetectorConfig  # noqa: E402


def _draw_box(frame, detection: ObjectDetection, color: tuple[int, int, int], label: str) -> None:
    height, width = frame.shape[:2]
    x1 = int((detection.x - detection.w / 2) * width)
    y1 = int((detection.y - detection.h / 2) * height)
    x2 = int((detection.x + detection.w / 2) * width)
    y2 = int((detection.y + detection.h / 2) * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {detection.conf:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _write_yolo_label(label_path: Path, class_id: int, det: ObjectDetection) -> None:
    # YOLO format: class x_center y_center width height (normalized)
    line = f"{class_id} {det.x:.6f} {det.y:.6f} {det.w:.6f} {det.h:.6f}\n"
    label_path.write_text(line, encoding="utf-8")


def export_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    ball_conf: float,
    hoop_conf: float,
    model_path: Path,
    device: str,
    max_width: int,
    hoop_roi_path: Path | None,
    include_hoop_overlay: bool,
    export_negatives: bool,
    min_positive_conf: float,
) -> dict[str, int | float]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python worker/scripts/download_model.py"
        )

    court_polygon = load_court_polygon(hoop_roi_path) if hoop_roi_path is not None else None
    detector = EbardDetector(
        model_path=model_path,
        config=DetectorConfig(ball_conf=ball_conf, hoop_conf=hoop_conf, device=device),
        court_polygon=court_polygon,
    )

    positives_img_dir = output_dir / "positives" / "images"
    positives_lbl_dir = output_dir / "positives" / "labels"
    negatives_img_dir = output_dir / "negatives" / "images"
    positives_img_dir.mkdir(parents=True, exist_ok=True)
    positives_lbl_dir.mkdir(parents=True, exist_ok=True)
    if export_negatives:
        negatives_img_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    sampled = 0
    positives = 0
    negatives = 0

    for frame in iter_sampled_frames(str(video_path), sample_fps=sample_fps, max_width=max_width):
        sampled += 1
        detections = detector.detect_objects(frame.bgr, ["basketball", "hoop"])
        ball = detections.get("basketball")
        hoop = detections.get("hoop")

        stem = f"frame_{frame.index:06d}"
        image = frame.bgr.copy()

        if ball is not None and ball.conf >= min_positive_conf:
            _draw_box(image, ball, (0, 255, 0), "ball")
            if include_hoop_overlay and hoop is not None:
                _draw_box(image, hoop, (0, 165, 255), "hoop")

            image_path = positives_img_dir / f"{stem}.jpg"
            label_path = positives_lbl_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), image)
            _write_yolo_label(label_path, class_id=0, det=ball)
            positives += 1
            manifest.append(
                {
                    "frame": frame.index,
                    "tSec": round(frame.t_sec, 3),
                    "split": "positive",
                    "image": str(image_path),
                    "label": str(label_path),
                    "ballConf": round(ball.conf, 4),
                    "hasHoop": hoop is not None,
                }
            )
        else:
            if export_negatives:
                image_path = negatives_img_dir / f"{stem}.jpg"
                cv2.imwrite(str(image_path), image)
                negatives += 1
                manifest.append(
                    {
                        "frame": frame.index,
                        "tSec": round(frame.t_sec, 3),
                        "split": "negative",
                        "image": str(image_path),
                        "hasHoop": hoop is not None,
                    }
                )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "sampled_frames": sampled,
        "positives": positives,
        "negatives": negatives,
        "manifest_entries": len(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export positive/negative sampled frames with YOLO labels for positives"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKER_ROOT / "output" / "frame_export",
        help="Output directory for positives/negatives and manifest",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Sampling FPS")
    parser.add_argument("--ball-conf", type=float, default=0.05, help="Ball confidence threshold")
    parser.add_argument("--hoop-conf", type=float, default=0.60, help="Hoop confidence threshold")
    parser.add_argument(
        "--min-positive-conf",
        type=float,
        default=0.0,
        help="Require at least this confidence to export positive frame",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Model weights path")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Inference device (mps on Mac, cuda:0 on NVIDIA). CPU is not allowed.",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Resize frame width cap")
    parser.add_argument(
        "--hoop-roi",
        type=Path,
        default=None,
        help="Optional hoop_roi.json to apply courtPolygon filtering",
    )
    parser.add_argument(
        "--include-hoop-overlay",
        action="store_true",
        help="Draw hoop box on positive images when detected",
    )
    parser.add_argument(
        "--no-negatives",
        action="store_true",
        help="Do not export negative images",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if args.hoop_roi is not None and not args.hoop_roi.exists():
        print(f"Hoop ROI file not found: {args.hoop_roi}", file=sys.stderr)
        sys.exit(1)

    try:
        inference_device = resolve_inference_device(args.device)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    metrics = export_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        ball_conf=args.ball_conf,
        hoop_conf=args.hoop_conf,
        model_path=args.model,
        device=inference_device,
        max_width=args.max_width,
        hoop_roi_path=args.hoop_roi,
        include_hoop_overlay=args.include_hoop_overlay,
        export_negatives=not args.no_negatives,
        min_positive_conf=args.min_positive_conf,
    )

    print(f"Output dir: {args.output_dir}")
    print(f"Sampled frames: {metrics['sampled_frames']}")
    print(f"Positive frames: {metrics['positives']}")
    print(f"Negative frames: {metrics['negatives']}")
    print(f"Manifest entries: {metrics['manifest_entries']}")


if __name__ == "__main__":
    main()
