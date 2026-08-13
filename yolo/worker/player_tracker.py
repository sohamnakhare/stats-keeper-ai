"""Player detection (YOLO E-BARD) + ByteTrack tracking on sampled frames."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from court_geometry import normalize_court_polygon, point_in_polygon
from detector import resolve_class_id, resolve_inference_device
from extract_frames import SampledFrame, iter_sampled_frames
from player_config import DetectionConfig, TrackingConfig


@dataclass(frozen=True)
class PlayerObservation:
    """One tracked player (or referee) in one sampled frame.

    Bbox and foot point are normalized to the frame (0-1).
    """

    frame: int
    t_sec: float
    track_id: int
    label: str  # "player" | "referee"
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def foot_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def foot_y(self) -> float:
        return self.y2

    @property
    def bbox_height(self) -> float:
        return self.y2 - self.y1


def write_tracker_yaml(config: TrackingConfig) -> Path:
    """Materialize ByteTrack params as a tracker YAML for ultralytics."""
    content = "\n".join(
        [
            "tracker_type: bytetrack",
            f"track_high_thresh: {config.track_high_thresh}",
            f"track_low_thresh: {config.track_low_thresh}",
            f"new_track_thresh: {config.new_track_thresh}",
            f"track_buffer: {config.track_buffer}",
            f"match_thresh: {config.match_thresh}",
            f"fuse_score: {config.fuse_score}",
            "",
        ]
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="bytetrack_", delete=False, encoding="utf-8"
    )
    with tmp:
        tmp.write(content)
    return Path(tmp.name)


def load_court_polygon(path: Path | str | None) -> list[tuple[float, float]] | None:
    """Load a normalized-image-coords courtPolygon from hoop_roi.json-style files."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("courtPolygon")
    if not raw:
        return None
    polygon = normalize_court_polygon(raw)
    return polygon if len(polygon) >= 3 else None


class PlayerTracker:
    """Streams (sampled frame, player observations) pairs for a video."""

    def __init__(self, detection: DetectionConfig, tracking: TrackingConfig) -> None:
        self.detection = detection
        self.tracking = tracking
        self._device = resolve_inference_device(detection.device)
        self.model = YOLO(str(detection.model_path))
        self._tracker_yaml = write_tracker_yaml(tracking)

        self._class_labels: dict[int, str] = {}
        player_id = resolve_class_id(self.model, "player")
        self._class_labels[player_id] = "player"
        self._class_ids = [player_id]
        if detection.detect_referees:
            try:
                referee_id = resolve_class_id(self.model, "referee")
                self._class_labels[referee_id] = "referee"
                self._class_ids.append(referee_id)
            except ValueError:
                pass

        self.court_polygon = load_court_polygon(detection.court_polygon_file)

    def _observations_from_result(
        self, result, frame_index: int, t_sec: float, width: int, height: int
    ) -> list[PlayerObservation]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        observations: list[PlayerObservation] = []
        ids = boxes.id
        cls = boxes.cls
        confs = boxes.conf
        xyxy = boxes.xyxy
        for idx in range(len(boxes)):
            label = self._class_labels.get(int(cls[idx].item()))
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[idx].tolist())
            obs = PlayerObservation(
                frame=frame_index,
                t_sec=t_sec,
                track_id=int(ids[idx].item()),
                label=label,
                x1=max(0.0, x1 / width),
                y1=max(0.0, y1 / height),
                x2=min(1.0, x2 / width),
                y2=min(1.0, y2 / height),
                conf=float(confs[idx].item()),
            )
            if self.court_polygon is not None and not point_in_polygon(
                obs.foot_x, obs.foot_y, self.court_polygon
            ):
                continue
            observations.append(obs)
        return observations

    def iter_observations(
        self, video_path: str
    ) -> Iterator[tuple[SampledFrame, list[PlayerObservation]]]:
        for sampled in iter_sampled_frames(
            video_path,
            sample_fps=self.detection.sample_fps,
            max_width=self.detection.max_width,
        ):
            frame = sampled.bgr
            assert isinstance(frame, np.ndarray)
            height, width = frame.shape[:2]
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=str(self._tracker_yaml),
                imgsz=self.detection.imgsz,
                conf=self.detection.conf,
                iou=self.detection.iou,
                classes=self._class_ids,
                device=self._device,
                verbose=False,
            )
            observations = (
                self._observations_from_result(
                    results[0], sampled.index, sampled.t_sec, width, height
                )
                if results
                else []
            )
            yield sampled, observations
