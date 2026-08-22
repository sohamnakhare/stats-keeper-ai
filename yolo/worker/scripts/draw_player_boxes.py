#!/usr/bin/env python3
"""Draw player bounding boxes on a video (E-BARD player class)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from court_geometry import point_in_polygon  # noqa: E402
from court_projection import CourtProjector  # noqa: E402
from detector import resolve_class_id, resolve_inference_device  # noqa: E402
from player_config import DetectionConfig, HomographyConfig, TrackingConfig  # noqa: E402
from player_tracker import load_court_polygon, write_tracker_yaml  # noqa: E402

PLAYER_COLOR = (0, 220, 255)  # yellow-orange BGR
REFEREE_COLOR = (255, 80, 180)  # pink BGR


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


def draw_box(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y1 = max(y1 - text_h - baseline - 4, 0)
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + text_w + 4, label_y1 + text_h + baseline + 4),
        color,
        -1,
    )
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
    *,
    model_path: Path,
    device: str,
    conf: float,
    imgsz: int,
    max_width: int,
    referees: bool,
    track: bool,
    no_reid: bool,
    calibration_file: Path | None,
    court_margin: float | None,
    court_polygon_file: Path | None,
) -> dict[str, float | int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python worker/scripts/download_model.py"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0

    resolved_device = resolve_inference_device(device)
    model = YOLO(str(model_path))
    class_labels: dict[int, str] = {}
    player_id = resolve_class_id(model, "player")
    class_labels[player_id] = "player"
    class_ids = [player_id]
    if referees:
        try:
            referee_id = resolve_class_id(model, "referee")
            class_labels[referee_id] = "referee"
            class_ids.append(referee_id)
        except ValueError:
            print("Referee class not in model; drawing players only", file=sys.stderr)

    tracker_yaml: Path | None = None
    if track:
        tracking = TrackingConfig()
        if no_reid:
            tracking.with_reid = False
        tracker_yaml = write_tracker_yaml(tracking)

    projector: CourtProjector | None = None
    if calibration_file is not None:
        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibration not found: {calibration_file}")
        homo = HomographyConfig()
        if court_margin is not None:
            homo.court_margin = court_margin
        projector = CourtProjector.from_calibration_file(calibration_file, homo)

    court_polygon = load_court_polygon(court_polygon_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    total = 0
    boxes_drawn = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_frame(frame, max_width)
            height, width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    native_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {output_path}")

            total += 1
            infer_kwargs = {
                "source": frame,
                "imgsz": imgsz,
                "conf": conf,
                "classes": class_ids,
                "device": resolved_device,
                "verbose": False,
            }
            if track:
                results = model.track(
                    persist=True,
                    tracker=str(tracker_yaml),
                    **infer_kwargs,
                )
            else:
                results = model.predict(**infer_kwargs)

            boxes = results[0].boxes if results else None
            if boxes is not None and len(boxes) > 0:
                ids = boxes.id
                for idx in range(len(boxes)):
                    label = class_labels.get(int(boxes.cls[idx].item()))
                    if label is None:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[idx].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    foot_x = ((x1 + x2) / 2) / width
                    foot_y = y2 / height
                    if projector is not None and projector.project(foot_x, foot_y) is None:
                        continue
                    if court_polygon is not None and not point_in_polygon(
                        foot_x, foot_y, court_polygon
                    ):
                        continue
                    conf_val = float(boxes.conf[idx].item())
                    if ids is not None:
                        text = f"{label} #{int(ids[idx].item())} {conf_val:.2f}"
                    else:
                        text = f"{label} {conf_val:.2f}"
                    color = PLAYER_COLOR if label == "player" else REFEREE_COLOR
                    draw_box(frame, x1, y1, x2, y2, text, color)
                    boxes_drawn += 1

            writer.write(frame)
            if total % 30 == 0:
                print(f"Processed {total} frames...", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        "frames_total": total,
        "boxes_drawn": boxes_drawn,
        "mean_boxes_per_frame": round(boxes_drawn / total, 2) if total else 0.0,
    }


def main() -> None:
    defaults = DetectionConfig()
    parser = argparse.ArgumentParser(description="Draw player boxes on a video")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "player_boxes.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--model", type=Path, default=Path(defaults.model_path))
    parser.add_argument("--device", default=defaults.device, help="Inference device (mps / cuda:0)")
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--max-width", type=int, default=defaults.max_width)
    parser.add_argument("--referees", action="store_true", help="Also draw referee boxes")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Run Deep OC-SORT and label boxes with track IDs",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="With --track, disable appearance ReID",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="court_calibration.json from court-marker.html (keep boxes whose feet project on court)",
    )
    parser.add_argument(
        "--court-margin",
        type=float,
        default=None,
        help="Meters outside the court to still accept a foot projection (default 0.3)",
    )
    parser.add_argument(
        "--court-polygon",
        type=Path,
        default=None,
        help="hoop_roi.json with courtPolygon (keep boxes whose feet are on court)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = annotate_video(
            video_path=args.video,
            output_path=args.output,
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            max_width=args.max_width,
            referees=args.referees,
            track=args.track,
            no_reid=args.no_reid,
            calibration_file=args.calibration,
            court_margin=args.court_margin,
            court_polygon_file=args.court_polygon,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.output}")
    print(f"Frames: {metrics['frames_total']}")
    print(f"Boxes drawn: {metrics['boxes_drawn']}")
    print(f"Mean boxes / frame: {metrics['mean_boxes_per_frame']}")


if __name__ == "__main__":
    main()
