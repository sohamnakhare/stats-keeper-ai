#!/usr/bin/env python3
"""Detect basketball court pose (D keypoints) and overlay the rest of the court."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from court_overlay import (  # noqa: E402
    DEFAULT_KEYPOINT_ORDER,
    EMA_ALPHA,
    FLOW_KEYPOINT_CONF,
    MIN_KEYPOINT_CONF,
    PRESETS,
    SNAP_RADIUS,
    OverlayFit,
    draw_court_diagram,
    fit_overlay,
    gradient_magnitude,
    keypoints_confident,
    outline_polygon_normalized,
    parse_keypoint_order,
    snap_keypoints,
    warp_keypoints_lk,
)
from detector import DEFAULT_DEVICE, resolve_inference_device  # noqa: E402

DEFAULT_MODEL_PATH = WORKER_ROOT / "models" / "court_pose_v1.pt"
KEYPOINT_COLOR = (0, 255, 255)  # yellow BGR
KEYPOINT_COUNT = 4


def _flow_overlay(
    prev_gray: np.ndarray | None,
    gray: np.ndarray,
    last_pts: np.ndarray | None,
    preset: dict[str, float | str],
    width: int,
    height: int,
    locked_order: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, OverlayFit] | None:
    """Warp last paint quad with LK flow and accept only a scored overlay fit."""
    if prev_gray is None or last_pts is None:
        return None
    warped = warp_keypoints_lk(prev_gray, gray, last_pts)
    if warped is None:
        return None
    fit = fit_overlay(
        [tuple(p) for p in warped],
        preset,
        width,
        height,
        order=locked_order,
    )
    if fit is None:
        return None
    return warped, fit


def _write_flow_keypoints(
    record: dict[str, Any],
    warped: np.ndarray,
    width: int,
    height: int,
    order: tuple[int, int, int, int] | None,
    paint_index: int,
) -> None:
    record["source"] = "flow"
    record["paintIndex"] = paint_index
    if order is not None:
        record["keypointOrder"] = list(order)
    record["keypoints"] = [
        {
            "x": round(float(px) / width, 6),
            "y": round(float(py) / height, 6),
            "conf": FLOW_KEYPOINT_CONF,
            "source": "flow",
        }
        for px, py in warped
    ]


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _best_court(result) -> dict[str, Any] | None:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or len(boxes) == 0 or keypoints is None or keypoints.xy is None:
        return None

    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    xywhn = boxes.xywhn[best_i].cpu().numpy()
    kpts_xy = keypoints.xy[best_i].cpu().numpy()
    kpts_xyn = keypoints.xyn[best_i].cpu().numpy()
    kpts_conf = None
    if keypoints.conf is not None:
        kpts_conf = keypoints.conf[best_i].cpu().numpy()

    corners: list[dict[str, float]] = []
    pixel_pts: list[tuple[float, float]] = []
    for i in range(min(KEYPOINT_COUNT, len(kpts_xy))):
        kx, ky = float(kpts_xyn[i][0]), float(kpts_xyn[i][1])
        px, py = float(kpts_xy[i][0]), float(kpts_xy[i][1])
        kconf = float(kpts_conf[i]) if kpts_conf is not None else 1.0
        corners.append({"x": round(kx, 6), "y": round(ky, 6), "conf": round(kconf, 4)})
        pixel_pts.append((px, py))

    if len(corners) != KEYPOINT_COUNT:
        return None

    return {
        "conf": round(float(confs[best_i]), 4),
        "box": {
            "x": round(float(xywhn[0]), 6),
            "y": round(float(xywhn[1]), 6),
            "w": round(float(xywhn[2]), 6),
            "h": round(float(xywhn[3]), 6),
        },
        "keypoints": corners,
        "pixel_pts": pixel_pts,
    }


def _draw_keypoints(frame: np.ndarray, pixel_pts: list[tuple[float, float]]) -> None:
    for i, (px, py) in enumerate(pixel_pts):
        pt = (int(round(px)), int(round(py)))
        cv2.circle(frame, pt, 6, KEYPOINT_COLOR, -1)
        cv2.circle(frame, pt, 7, (0, 0, 0), 1)
        cv2.putText(
            frame,
            str(i),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            KEYPOINT_COLOR,
            2,
            cv2.LINE_AA,
        )


def detect_court(
    video_path: Path,
    output_path: Path,
    json_path: Path,
    *,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    max_width: int,
    preset_name: str,
    keypoint_order: tuple[int, int, int, int] | None = DEFAULT_KEYPOINT_ORDER,
) -> dict[str, float | int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Court pose model not found at {model_path}. "
            "Train with: python worker/scripts/train_court_model.py"
        )
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Choose from: {', '.join(PRESETS)}")

    preset = PRESETS[preset_name]
    resolved_device = resolve_inference_device(device)
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    frames: list[dict[str, Any]] = []
    total = 0
    detected = 0
    overlayed = 0
    best: dict[str, Any] | None = None
    locked_order = keypoint_order
    locked_paint_index: int | None = None
    last_homography: np.ndarray | None = None
    ema_pts: np.ndarray | None = None
    last_track_pts: np.ndarray | None = None
    prev_gray: np.ndarray | None = None

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
            t_sec = round((total - 1) / native_fps, 4)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                device=resolved_device,
                verbose=False,
            )
            court = _best_court(results[0]) if results else None
            record: dict[str, Any] = {
                "index": total - 1,
                "t_sec": t_sec,
                "conf": None,
            }
            overlay_h: np.ndarray | None = None
            pose_ok = False
            snapped: list[tuple[float, float]] | None = None

            if court is not None:
                detected += 1
                record["conf"] = court["conf"]
                record["box"] = court["box"]
                record["keypoints"] = court["keypoints"]

                mag = gradient_magnitude(frame)
                snapped = snap_keypoints(court["pixel_pts"], mag, SNAP_RADIUS)
                kpt_confs = [kp["conf"] for kp in court["keypoints"]]

                if keypoints_confident(kpt_confs, MIN_KEYPOINT_CONF):
                    pts = np.array(snapped, dtype=np.float64)
                    if ema_pts is None:
                        ema_pts = pts
                    else:
                        ema_pts = (1.0 - EMA_ALPHA) * ema_pts + EMA_ALPHA * pts
                    fit: OverlayFit | None = fit_overlay(
                        [tuple(p) for p in ema_pts],
                        preset,
                        width,
                        height,
                        order=locked_order,
                    )
                    if fit is None and locked_order is not None and keypoint_order is None:
                        fit = fit_overlay(
                            [tuple(p) for p in ema_pts],
                            preset,
                            width,
                            height,
                            order=None,
                        )
                        if fit is not None:
                            locked_order = fit.order
                    if fit is not None:
                        pose_ok = True
                        if locked_order is None:
                            locked_order = fit.order
                        locked_paint_index = fit.paint_index
                        last_homography = fit.homography
                        overlay_h = fit.homography
                        last_track_pts = ema_pts.copy()
                        record["source"] = "pose"
                        record["keypointOrder"] = list(fit.order)
                        record["paintIndex"] = fit.paint_index
                        for i, (px, py) in enumerate(ema_pts):
                            record["keypoints"][i]["x"] = round(float(px) / width, 6)
                            record["keypoints"][i]["y"] = round(float(py) / height, 6)

            if not pose_ok:
                flowed = _flow_overlay(
                    prev_gray, gray, last_track_pts, preset, width, height, locked_order
                )
                if flowed is not None:
                    warped, fit = flowed
                    if locked_order is None:
                        locked_order = fit.order
                    locked_paint_index = fit.paint_index
                    last_homography = fit.homography
                    overlay_h = fit.homography
                    last_track_pts = warped
                    _write_flow_keypoints(
                        record, warped, width, height, locked_order, fit.paint_index
                    )

            if overlay_h is None and last_homography is not None:
                overlay_h = last_homography

            if overlay_h is not None:
                overlayed += 1
                draw_court_diagram(frame, overlay_h, preset)
                record["courtPolygon"] = outline_polygon_normalized(
                    overlay_h, preset, width, height
                )

            if snapped is not None:
                _draw_keypoints(frame, snapped)
            elif record.get("source") == "flow" and last_track_pts is not None:
                _draw_keypoints(frame, [tuple(p) for p in last_track_pts])

            if court is not None and (best is None or court["conf"] > best["conf"]):
                best = {
                    "index": record["index"],
                    "t_sec": t_sec,
                    "conf": court["conf"],
                    "box": court["box"],
                    "keypoints": record.get("keypoints") or court["keypoints"],
                    "keypointOrder": record.get("keypointOrder"),
                    "paintIndex": record.get("paintIndex"),
                    "courtPolygon": record.get("courtPolygon")
                    or [{"x": kp["x"], "y": kp["y"]} for kp in court["keypoints"]],
                }

            frames.append(record)
            writer.write(frame)
            prev_gray = gray

            if total % 30 == 0:
                print(f"Processed {total} frames...", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    payload = {
        "format": "court-detect-v1",
        "source": "yolo-pose",
        "clip": video_path.name,
        "model": str(model_path),
        "preset": preset_name,
        "keypointOrder": list(locked_order) if locked_order is not None else None,
        "paintIndex": locked_paint_index,
        "conf": conf,
        "imgsz": imgsz,
        "maxWidth": max_width,
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "frames": frames,
        "best": best,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return {
        "frames_total": total,
        "court_frames_detected": detected,
        "court_frames_overlayed": overlayed,
        "court_detection_rate": round(detected / total, 4) if total else 0.0,
        "court_overlay_rate": round(overlayed / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect the paint (D) and overlay the rest of the court"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "court_detect.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=WORKER_ROOT / "output" / "court_detect.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained court pose weights",
    )
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-width", type=int, default=1280, help="Max frame width for detection")
    parser.add_argument(
        "--preset",
        default="fibaHalf",
        choices=sorted(PRESETS.keys()),
        help="Court dimensions used to project paint/arc/boundary from the D",
    )
    parser.add_argument(
        "--keypoint-order",
        default="0,1,2,3",
        help=(
            "Model keypoint indices for baseL,baseR,ftR,ftL. "
            "Default 0,1,2,3 = bottom-left, bottom-right, top-right, top-left. "
            "Pass 'auto' to search permutations."
        ),
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        order = parse_keypoint_order(args.keypoint_order)
        if order is None and args.keypoint_order.strip().lower() != "auto":
            order = DEFAULT_KEYPOINT_ORDER
        metrics = detect_court(
            video_path=args.video,
            output_path=args.output,
            json_path=args.json,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            max_width=args.max_width,
            preset_name=args.preset,
            keypoint_order=order,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.json}")
    print(f"Frames: {metrics['frames_total']}")
    print(f"Court detection rate: {metrics['court_detection_rate']:.1%}")
    print(f"Court overlay rate: {metrics['court_overlay_rate']:.1%}")


if __name__ == "__main__":
    main()
