"""Stage 1: per-frame player and ball detection with Ultralytics YOLOv8.

Default weights are E-BARD YOLOv8n (basketball / hoop / player / referee),
which is far more accurate on court footage than COCO yolov8n.pt.

Fine-tuning notes (do not train in this pipeline — flag only):
- Tiny / motion-blurred ball is the usual failure mode. A ball-only fine-tune
  (see yolo/worker/scripts/train_ball_model.py in this repo) helps more than
  switching YOLO generations.
- Sideline, bench, and crowd often fire as `player`. A court polygon or a
  higher confidence threshold is the cheap fix; a court-aware fine-tune is
  the proper one.
- Vanilla COCO (`person`, `sports ball`) is a fallback only. It will miss the
  ball often and treat coaches/spectators as players.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.schemas import FORMAT_DETECTIONS, Detection, to_dict
from pipeline.video import iter_sampled_frames, make_video_writer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "BODD_yolov8n_0001.pt"
COCO_FALLBACK = "yolov8n.pt"

# Preferred class names, then COCO aliases.
PLAYER_ALIASES = ("player", "person")
BALL_ALIASES = ("basketball", "ball", "sports ball")
HOOP_ALIASES = ("hoop",)
REFEREE_ALIASES = ("referee",)

BOX_COLORS = {
    "player": (40, 180, 70),
    "basketball": (0, 140, 255),
    "hoop": (200, 180, 40),
    "referee": (160, 160, 160),
    "person": (40, 180, 70),
    "sports ball": (0, 140, 255),
}


def resolve_device(device: str | None = None) -> str:
    """Prefer Apple GPU, then CUDA, then CPU."""
    if device:
        return device
    try:
        import torch

        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def default_model_path() -> Path:
    local = ROOT / "models" / DEFAULT_MODEL_NAME
    if local.exists():
        return local
    sibling = ROOT.parent / "yolo" / "worker" / "models" / DEFAULT_MODEL_NAME
    if sibling.exists():
        return sibling
    return local


def resolve_class_id(model: YOLO, aliases: tuple[str, ...]) -> int | None:
    names = model.names
    items: list[tuple[int, str]]
    if isinstance(names, dict):
        items = [(int(i), str(n).lower()) for i, n in names.items()]
    else:
        items = [(i, str(n).lower()) for i, n in enumerate(names)]
    wanted = {a.lower() for a in aliases}
    for idx, name in items:
        if name in wanted:
            return idx
    return None


def load_detector(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[YOLO, dict[str, int], str]:
    """Load YOLO weights and map logical labels to class ids.

    Returns (model, label_to_id, resolved_device).
    """
    path = Path(model_path) if model_path else default_model_path()
    if not path.exists():
        print(
            f"E-BARD weights not found at {path}. "
            f"Falling back to COCO {COCO_FALLBACK} — ball/player accuracy will be poor. "
            "Run: python scripts/download_model.py",
            file=sys.stderr,
        )
        path = Path(COCO_FALLBACK)

    device_id = resolve_device(device)
    model = YOLO(str(path))
    label_to_id: dict[str, int] = {}
    player_id = resolve_class_id(model, PLAYER_ALIASES)
    ball_id = resolve_class_id(model, BALL_ALIASES)
    if player_id is None:
        raise ValueError(f"No player/person class in model names: {model.names}")
    label_to_id["player"] = player_id
    if ball_id is not None:
        label_to_id["basketball"] = ball_id
    hoop_id = resolve_class_id(model, HOOP_ALIASES)
    if hoop_id is not None:
        label_to_id["hoop"] = hoop_id
    referee_id = resolve_class_id(model, REFEREE_ALIASES)
    if referee_id is not None:
        label_to_id["referee"] = referee_id
    return model, label_to_id, device_id


def _boxes_to_detections(
    result,
    frame_index: int,
    t_sec: float,
    width: int,
    height: int,
    id_to_label: dict[int, str],
) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    detections: list[Detection] = []
    cls = boxes.cls
    confs = boxes.conf
    xyxy = boxes.xyxy
    for i in range(len(boxes)):
        cls_id = int(cls[i].item())
        label = id_to_label.get(cls_id)
        if label is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
        nx1, ny1 = max(0.0, x1 / width), max(0.0, y1 / height)
        nx2, ny2 = min(1.0, x2 / width), min(1.0, y2 / height)
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        is_person = label in {"player", "referee", "person"}
        detections.append(
            Detection(
                frame=frame_index,
                t=round(t_sec, 4),
                label=label,
                conf=round(float(confs[i].item()), 4),
                x1=round(nx1, 5),
                y1=round(ny1, 5),
                x2=round(nx2, 5),
                y2=round(ny2, 5),
                cx=round(cx, 5),
                cy=round(cy, 5),
                foot_x=round(cx, 5) if is_person else None,
                foot_y=round(ny2, 5) if is_person else None,
            )
        )
    return detections


def annotate_frame(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for det in detections:
        color = BOX_COLORS.get(det.label, (255, 255, 255))
        p1 = (int(det.x1 * w), int(det.y1 * h))
        p2 = (int(det.x2 * w), int(det.y2 * h))
        cv2.rectangle(out, p1, p2, color, 2)
        caption = f"{det.label} {det.conf:.2f}"
        cv2.putText(
            out,
            caption,
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        if det.foot_x is not None and det.foot_y is not None:
            cv2.circle(
                out,
                (int(det.foot_x * w), int(det.foot_y * h)),
                4,
                color,
                -1,
            )
    return out


def run_detect(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 704,
    sample_fps: float = 10.0,
    max_width: int = 1280,
    annotate: bool = True,
    detect_referees: bool = False,
) -> dict:
    """Run Stage 1. Writes detections.json and optionally detections_annotated.mp4."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, label_to_id, device_id = load_detector(model_path, device)
    class_ids = [label_to_id["player"]]
    if "basketball" in label_to_id:
        class_ids.append(label_to_id["basketball"])
    if "hoop" in label_to_id:
        class_ids.append(label_to_id["hoop"])
    if detect_referees and "referee" in label_to_id:
        class_ids.append(label_to_id["referee"])
    id_to_label = {v: k for k, v in label_to_id.items() if v in class_ids}

    detections: list[Detection] = []
    writer = None
    sample_fps_eff = sample_fps
    frame_count = 0
    counts: Counter[str] = Counter()

    for sampled in iter_sampled_frames(video_path, sample_fps=sample_fps, max_width=max_width):
        frame = sampled.bgr
        h, w = frame.shape[:2]
        sample_fps_eff = sampled.sample_fps
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=class_ids,
            device=device_id,
            verbose=False,
        )
        frame_dets = (
            _boxes_to_detections(results[0], sampled.index, sampled.t_sec, w, h, id_to_label)
            if results
            else []
        )
        detections.extend(frame_dets)
        for det in frame_dets:
            counts[det.label] += 1
        frame_count += 1

        if annotate:
            if writer is None:
                writer = make_video_writer(
                    output_dir / "detections_annotated.mp4",
                    w,
                    h,
                    sample_fps_eff,
                )
            writer.write(annotate_frame(frame, frame_dets))

    if writer is not None:
        writer.release()

    artifact = {
        "format": FORMAT_DETECTIONS,
        "video": str(Path(video_path).resolve()),
        "model": str(Path(model_path) if model_path else default_model_path()),
        "device": device_id,
        "sampleFps": round(sample_fps_eff, 4),
        "frameCount": frame_count,
        "classCounts": dict(counts),
        "detections": [to_dict(d) for d in detections],
    }
    out_json = output_dir / "detections.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(
        f"Stage 1 detect: {frame_count} frames, {len(detections)} boxes "
        f"({dict(counts)}) → {out_json}"
    )
    if annotate:
        print(f"  annotated video: {output_dir / 'detections_annotated.mp4'}")
    if counts.get("basketball", 0) < max(1, frame_count * 0.2):
        print(
            "  note: ball detections are sparse. Fine-tune the ball class or "
            "lower --conf before relying on pass/dribble geometry.",
            file=sys.stderr,
        )
    return artifact
