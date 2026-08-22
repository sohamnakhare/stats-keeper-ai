"""COCO YOLO-pose for jersey torso crops and ankle foot points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from player_config import TeamConfig
from player_tracker import PlayerObservation

# COCO-17 indices used for the jersey rectangle and ground point.
COCO_L_SHOULDER = 5
COCO_R_SHOULDER = 6
COCO_L_HIP = 11
COCO_R_HIP = 12
COCO_L_ANKLE = 15
COCO_R_ANKLE = 16
TORSO_IDX = (COCO_L_SHOULDER, COCO_R_SHOULDER, COCO_L_HIP, COCO_R_HIP)
ANKLE_IDX = (COCO_L_ANKLE, COCO_R_ANKLE)


@dataclass(frozen=True)
class PoseInstance:
    """One COCO-pose person. Boxes are normalized 0-1; keypoints are pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    kpts_xy: np.ndarray  # (17, 2) pixel coords
    kpts_conf: np.ndarray  # (17,)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes (any unit, as long as they match)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def torso_xyxy_from_keypoints(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    *,
    min_conf: float = 0.4,
    min_keypoints: int = 3,
    pad: float = 0.12,
) -> tuple[int, int, int, int] | None:
    """Pixel xyxy covering shoulders+hips. None if too few confident torso points."""
    points: list[tuple[float, float]] = []
    for idx in TORSO_IDX:
        if idx >= len(kpts_xy) or idx >= len(kpts_conf):
            continue
        if float(kpts_conf[idx]) < min_conf:
            continue
        x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
        if x <= 0 and y <= 0:
            continue
        points.append((x, y))
    if len(points) < min_keypoints:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(x2 - x1, 4.0), max(y2 - y1, 4.0)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad * 0.35
    y2 += bh * pad * 0.15
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def intersect_xyxy(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def crop_torso_from_pose(
    frame_bgr: np.ndarray,
    obs: PlayerObservation,
    pose: PoseInstance,
    config: TeamConfig,
) -> np.ndarray | None:
    """Jersey crop from pose keypoints, clipped to the player box."""
    height, width = frame_bgr.shape[:2]
    torso = torso_xyxy_from_keypoints(
        pose.kpts_xy,
        pose.kpts_conf,
        min_conf=config.pose_min_conf,
        min_keypoints=config.pose_min_keypoints,
    )
    if torso is None:
        return None
    player_box = (
        int(obs.x1 * width),
        int(obs.y1 * height),
        int(obs.x2 * width),
        int(obs.y2 * height),
    )
    clipped = intersect_xyxy(torso, player_box)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame_bgr[y1:y2, x1:x2]


def foot_xy_from_pose(
    obs: PlayerObservation,
    pose: PoseInstance,
    frame_wh: tuple[int, int],
    *,
    min_conf: float = 0.4,
) -> tuple[float, float] | None:
    """Normalized foot point from ankle keypoints.

    Midpoint of both confident ankles. Returns None when either ankle is
    missing so the caller can use bbox bottom-center (a single ankle jumps).
    """
    width, height = frame_wh
    if width <= 0 or height <= 0:
        return None
    points: list[tuple[float, float]] = []
    for idx in ANKLE_IDX:
        if idx >= len(pose.kpts_xy) or idx >= len(pose.kpts_conf):
            continue
        if float(pose.kpts_conf[idx]) < min_conf:
            continue
        x, y = float(pose.kpts_xy[idx][0]), float(pose.kpts_xy[idx][1])
        if x <= 0 and y <= 0:
            continue
        nx, ny = x / width, y / height
        if nx < obs.x1 - 0.05 or nx > obs.x2 + 0.05:
            continue
        if ny < obs.y1 or ny > min(1.0, obs.y2 + 0.08):
            continue
        points.append((nx, ny))
    if not points:
        return None
    if len(points) < 2:
        return None
    return (
        (points[0][0] + points[1][0]) / 2.0,
        (points[0][1] + points[1][1]) / 2.0,
    )


def match_pose(
    obs: PlayerObservation, poses: list[PoseInstance], min_iou: float = 0.25
) -> PoseInstance | None:
    """Best pose instance overlapping this player box."""
    player = (obs.x1, obs.y1, obs.x2, obs.y2)
    best: PoseInstance | None = None
    best_iou = min_iou
    for pose in poses:
        iou = box_iou(player, (pose.x1, pose.y1, pose.x2, pose.y2))
        if iou > best_iou:
            best_iou = iou
            best = pose
    return best


class PlayerPoseEstimator:
    """Full-frame COCO pose; one predict per sampled frame."""

    def __init__(self, config: TeamConfig, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.config = config
        self.device = device
        model_path = config.pose_model
        if not Path(model_path).exists():
            model_path = "yolov8n-pose.pt"
        self.model = YOLO(model_path)

    def estimate(self, frame_bgr: np.ndarray) -> list[PoseInstance]:
        kwargs: dict = {"source": frame_bgr, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or len(boxes) == 0 or keypoints is None or keypoints.xy is None:
            return []

        xyxyn = boxes.xyxyn.cpu().numpy()
        kpts_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is not None:
            kpts_conf = keypoints.conf.cpu().numpy()
        else:
            kpts_conf = np.ones((len(xyxyn), kpts_xy.shape[1]), dtype=np.float32)

        poses: list[PoseInstance] = []
        for i in range(len(xyxyn)):
            x1, y1, x2, y2 = (float(v) for v in xyxyn[i].tolist())
            poses.append(
                PoseInstance(
                    x1=max(0.0, x1),
                    y1=max(0.0, y1),
                    x2=min(1.0, x2),
                    y2=min(1.0, y2),
                    kpts_xy=kpts_xy[i],
                    kpts_conf=kpts_conf[i],
                )
            )
        return poses
