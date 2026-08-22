"""Stage 2: ByteTrack player IDs + Kalman association for the ball.

ByteTrack (via Ultralytics `model.track`) is used for players. The ball is a
single tiny object, so ByteTrack IDs flicker; a constant-velocity Kalman filter
with nearest-neighbor gating is more stable.

Gaps: each player track records `gaps` where the ID was lost and later resumed.
Ball points with `missing=True` / `predicted=True` are Kalman coasting through
occlusions — treat them as low confidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.detect import load_detector
from pipeline.schemas import FORMAT_TRACKS, Track, TrackPoint, to_dict
from pipeline.video import iter_sampled_frames

TRACKER_YAML = Path(__file__).resolve().parent / "bytetrack.yaml"

# Max normalized-pixel distance to associate a ball detection with the filter.
BALL_GATE = 0.18
# Coast this many sampled frames without a detection before dropping the lock.
BALL_MAX_MISSES = 12


class BallKalman:
    """Constant-velocity Kalman filter on normalized (cx, cy)."""

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt
        self.x: np.ndarray | None = None
        self.P = np.eye(4, dtype=np.float64)
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        q = 0.02
        self.Q = np.diag([q, q, q * 4, q * 4]).astype(np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * 0.04
        self.misses = 0

    def predict(self) -> tuple[float, float] | None:
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, zx: float, zy: float) -> tuple[float, float]:
        z = np.array([zx, zy], dtype=np.float64)
        if self.x is None:
            self.x = np.array([zx, zy, 0.0, 0.0], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64)
            self.misses = 0
            return zx, zy
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.misses = 0
        return float(self.x[0]), float(self.x[1])

    def step(
        self, det_xy: tuple[float, float] | None
    ) -> tuple[float, float, bool, bool] | None:
        """Returns (x, y, missing, predicted) or None if unlocked."""
        pred = self.predict()
        if det_xy is not None:
            if pred is None or np.hypot(det_xy[0] - pred[0], det_xy[1] - pred[1]) <= BALL_GATE:
                x, y = self.update(*det_xy)
                return x, y, False, False
            # Detection is far from prediction — start a new lock (ID switch / bounce).
            self.x = np.array([det_xy[0], det_xy[1], 0.0, 0.0], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64)
            self.misses = 0
            return det_xy[0], det_xy[1], False, False
        if pred is None:
            return None
        self.misses += 1
        if self.misses > BALL_MAX_MISSES:
            self.x = None
            return None
        return pred[0], pred[1], True, True


def _player_point(xyxy: list[float], width: int, height: int) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    nx1, ny1 = max(0.0, x1 / width), max(0.0, y1 / height)
    nx2, ny2 = min(1.0, x2 / width), min(1.0, y2 / height)
    foot_x = (nx1 + nx2) / 2
    foot_y = ny2
    return nx1, ny1, nx2, ny2, foot_x, foot_y


def _gap_list(points: list[TrackPoint], sample_fps: float) -> list[dict[str, float | int]]:
    if len(points) < 2:
        return []
    step = 1.0 / max(sample_fps, 0.1)
    gaps: list[dict[str, float | int]] = []
    ordered = sorted(points, key=lambda p: p.t)
    for prev, nxt in zip(ordered, ordered[1:]):
        dt = nxt.t - prev.t
        if dt > step * 1.75:
            gaps.append(
                {
                    "tStart": round(prev.t, 3),
                    "tEnd": round(nxt.t, 3),
                    "frames": int(round(dt * sample_fps)) - 1,
                }
            )
    return gaps


def run_track(
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
    detect_referees: bool = False,
) -> dict:
    """Run Stage 2. Writes tracks.json with player IDs and a ball series."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, label_to_id, device_id = load_detector(model_path, device)
    player_ids = [label_to_id["player"]]
    if detect_referees and "referee" in label_to_id:
        player_ids.append(label_to_id["referee"])
    ball_id = label_to_id.get("basketball")
    track_classes = list(player_ids)
    if ball_id is not None:
        track_classes.append(ball_id)
    id_to_label = {v: k for k, v in label_to_id.items()}

    player_points: dict[int, list[TrackPoint]] = defaultdict(list)
    player_labels: dict[int, str] = {}
    ball_points: list[TrackPoint] = []
    kalman = BallKalman(dt=1.0 / max(sample_fps, 0.1))
    sample_fps_eff = sample_fps
    frame_count = 0
    missing_ball = 0

    # persist=True keeps ByteTrack state across sampled frames.
    for sampled in iter_sampled_frames(video_path, sample_fps=sample_fps, max_width=max_width):
        frame = sampled.bgr
        h, w = frame.shape[:2]
        sample_fps_eff = sampled.sample_fps
        kalman.dt = 1.0 / max(sample_fps_eff, 0.1)
        kalman.F[0, 2] = kalman.dt
        kalman.F[1, 3] = kalman.dt

        results = model.track(
            source=frame,
            persist=True,
            tracker=str(TRACKER_YAML),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=track_classes,
            device=device_id,
            verbose=False,
        )
        result = results[0] if results else None
        boxes = result.boxes if result is not None else None
        ball_det: tuple[float, float] | None = None
        best_ball_conf = -1.0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy
            cls = boxes.cls
            confs = boxes.conf
            ids = boxes.id
            for i in range(len(boxes)):
                cls_id = int(cls[i].item())
                label = id_to_label.get(cls_id, "player")
                conf_i = float(confs[i].item())
                x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
                if ball_id is not None and cls_id == ball_id:
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    if conf_i > best_ball_conf:
                        best_ball_conf = conf_i
                        ball_det = (cx, cy)
                    continue
                if ids is None:
                    continue
                track_id = int(ids[i].item())
                _, _, _, _, foot_x, foot_y = _player_point([x1, y1, x2, y2], w, h)
                player_labels[track_id] = label
                player_points[track_id].append(
                    TrackPoint(
                        frame=sampled.index,
                        t=round(sampled.t_sec, 4),
                        px=round(foot_x, 5),
                        py=round(foot_y, 5),
                        conf=round(conf_i, 4),
                        missing=False,
                        predicted=False,
                    )
                )

        stepped = kalman.step(ball_det)
        if stepped is not None:
            bx, by, missing, predicted = stepped
            if missing:
                missing_ball += 1
            ball_points.append(
                TrackPoint(
                    frame=sampled.index,
                    t=round(sampled.t_sec, 4),
                    px=round(float(np.clip(bx, 0, 1)), 5),
                    py=round(float(np.clip(by, 0, 1)), 5),
                    conf=None if missing else round(best_ball_conf, 4),
                    missing=missing,
                    predicted=predicted,
                )
            )
        frame_count += 1

    players = []
    for track_id, points in sorted(player_points.items()):
        ordered = sorted(points, key=lambda p: p.t)
        players.append(
            Track(
                track_id=track_id,
                label=player_labels.get(track_id, "player"),
                points=ordered,
                gaps=_gap_list(ordered, sample_fps_eff),
            )
        )

    ball_gaps = _gap_list([p for p in ball_points if not p.missing], sample_fps_eff)
    artifact = {
        "format": FORMAT_TRACKS,
        "video": str(Path(video_path).resolve()),
        "device": device_id,
        "sampleFps": round(sample_fps_eff, 4),
        "frameCount": frame_count,
        "playerCount": len(players),
        "players": [to_dict(p) for p in players],
        "ball": {
            "label": "basketball",
            "points": [to_dict(p) for p in ball_points],
            "gaps": ball_gaps,
            "predictedCount": missing_ball,
        },
        "notes": {
            "playerTracker": "Ultralytics ByteTrack",
            "ballTracker": "constant-velocity Kalman + nearest-neighbor gate",
            "occlusion": (
                "Player gaps[] list occlusions/re-entry. Ball points with "
                "missing/predicted=true are coasted through missed detections."
            ),
        },
    }
    out_json = output_dir / "tracks.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    n_gaps = sum(len(p.gaps) for p in players)
    print(
        f"Stage 2 track: {len(players)} player IDs, {len(ball_points)} ball points "
        f"({missing_ball} predicted), {n_gaps} player gaps → {out_json}"
    )
    return artifact


def load_tracks(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
