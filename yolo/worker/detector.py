"""E-BARD YOLOv8n basketball detection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from court_geometry import ball_center_in_court
from schemas import DetectorConfig, HoopRoi

MODEL_VERSION = "ebard-yolov8n-bodd-0001"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "BODD_yolov8n_0001.pt"
DEFAULT_DEVICE = "mps" if sys.platform == "darwin" else "cuda:0"


def resolve_inference_device(device: str = DEFAULT_DEVICE) -> str:
    """Resolve and validate the inference device. CPU fallback is not allowed."""
    normalized = device.strip().lower()
    if normalized == "cpu":
        raise ValueError(
            "CPU inference is disabled. Use device='mps' on Apple Silicon or device='cuda:0' on NVIDIA GPUs."
        )

    if normalized in {"mps", "mps:0"}:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS (Apple GPU) was requested but is not available. "
                "Ensure PyTorch is installed with MPS support and macOS 14+."
            )
        return "mps"

    cuda_requested = normalized.startswith("cuda") or normalized.isdigit() or "," in normalized
    if cuda_requested:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was requested but is not available. "
                "Install a CUDA-enabled PyTorch build or use device='mps' on Apple Silicon."
            )
        return device

    raise ValueError(f"Unknown inference device: {device!r}. Use 'mps' or 'cuda:0'.")


@dataclass(frozen=True)
class BallDetection:
    x: float  # normalized center 0-1
    y: float
    w: float
    h: float
    conf: float


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    x: float
    y: float
    w: float
    h: float
    conf: float


def resolve_class_id(model: YOLO, class_name: str) -> int:
    target = class_name.lower()
    names = model.names
    if isinstance(names, dict):
        for idx, name in names.items():
            if str(name).lower() == target:
                return int(idx)
    if isinstance(names, list):
        for idx, name in enumerate(names):
            if str(name).lower() == target:
                return idx
    raise ValueError(f"{class_name!r} class not found in model names: {names}")


def resolve_basketball_class_id(model: YOLO) -> int:
    try:
        return resolve_class_id(model, "basketball")
    except ValueError:
        pass
    try:
        return resolve_class_id(model, "ball")
    except ValueError:
        pass
    names = model.names
    if isinstance(names, dict) and len(names) == 1:
        return int(next(iter(names.keys())))
    if isinstance(names, list) and len(names) == 1:
        return 0
    raise ValueError(f"'basketball' class not found in model names: {names}")


def _box_to_detection(
    label: str,
    xyxy: list[float],
    conf: float,
    width: int,
    height: int,
) -> ObjectDetection:
    x1, y1, x2, y2 = xyxy
    return ObjectDetection(
        label=label,
        x=((x1 + x2) / 2) / width,
        y=((y1 + y2) / 2) / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
        conf=conf,
    )


def hoop_search_region(hoop: HoopRoi, expand: float = 3.5) -> tuple[float, float, float, float]:
    """Normalized crop bounds (x1, y1, x2, y2) around the hoop for ball search."""
    half_w = (hoop.w / 2) * expand
    half_h = (hoop.h / 2) * expand
    x1 = max(0.0, hoop.x - half_w)
    x2 = min(1.0, hoop.x + half_w)
    y1 = max(0.0, hoop.y - half_h)
    y2 = min(1.0, hoop.y + half_h)
    return x1, y1, x2, y2


def _map_crop_detection_to_frame(
    detection: ObjectDetection,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> BallDetection:
    abs_cx = crop_x1 + detection.x * crop_w
    abs_cy = crop_y1 + detection.y * crop_h
    return BallDetection(
        x=abs_cx / frame_w,
        y=abs_cy / frame_h,
        w=detection.w * crop_w / frame_w,
        h=detection.h * crop_h / frame_h,
        conf=detection.conf,
    )


class BallDetector:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        config: DetectorConfig | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self._device = resolve_inference_device(self.config.device)
        self.model = YOLO(str(model_path))
        self.basketball_class_id = resolve_basketball_class_id(self.model)

    def detect_frame(self, frame_bgr: np.ndarray) -> BallDetection | None:
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return None

        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            classes=[self.basketball_class_id],
            verbose=False,
            device=self._device,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx].item())
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        return BallDetection(x=cx, y=cy, w=bw, h=bh, conf=conf)


class EbardDetector:
    """Detect E-BARD objects (ball, hoop, etc.) in a single inference pass."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        config: DetectorConfig | None = None,
        court_polygon: list[tuple[float, float]] | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.court_polygon = court_polygon
        self._device = resolve_inference_device(self.config.device)
        self.model = YOLO(str(model_path))
        self._class_ids: dict[str, int] = {}

    def _filter_ball_for_court(self, ball: ObjectDetection | None) -> ObjectDetection | None:
        if ball is None:
            return None
        if not ball_center_in_court(ball, self.court_polygon):
            return None
        return ball

    def class_id(self, class_name: str) -> int:
        key = class_name.lower()
        if key not in self._class_ids:
            self._class_ids[key] = resolve_class_id(self.model, key)
        return self._class_ids[key]

    def detect_objects(
        self,
        frame_bgr: np.ndarray,
        class_names: list[str],
    ) -> dict[str, ObjectDetection | None]:
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return {name.lower(): None for name in class_names}

        names = [name.lower() for name in class_names]
        class_ids = [self.class_id(name) for name in names]
        id_to_label = {self.class_id(name): name for name in names}

        class_thresholds = {
            "basketball": self.config.ball_conf,
            "hoop": self.config.hoop_conf,
        }
        predict_conf = min(self.config.conf, *(class_thresholds.get(n, self.config.conf) for n in names))

        predict = self.model.predict(
            source=frame_bgr,
            imgsz=self.config.imgsz,
            conf=predict_conf,
            iou=self.config.iou,
            classes=class_ids,
            verbose=False,
            device=self._device,
        )

        results: dict[str, ObjectDetection | None] = {name: None for name in names}
        if not predict:
            return results

        boxes = predict[0].boxes
        if boxes is None or len(boxes) == 0:
            return results

        cls_tensor = boxes.cls
        conf_tensor = boxes.conf
        xyxy_tensor = boxes.xyxy

        for idx in range(len(boxes)):
            cls_id = int(cls_tensor[idx].item())
            label = id_to_label.get(cls_id)
            if label is None:
                continue

            conf = float(conf_tensor[idx].item())
            min_conf = class_thresholds.get(label, self.config.conf)
            if conf < min_conf:
                continue

            det = _box_to_detection(label, xyxy_tensor[idx].tolist(), conf, width, height)
            current = results[label]
            if current is None or det.conf > current.conf:
                results[label] = det

        if "basketball" in results:
            results["basketball"] = self._filter_ball_for_court(results["basketball"])

        return results

    def detect_ball_in_hoop_region(
        self,
        frame_bgr: np.ndarray,
        hoop: HoopRoi,
        expand: float = 3.5,
    ) -> BallDetection | None:
        """Run ball detection on a crop around the hoop (larger ball in frame)."""
        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        x1n, y1n, x2n, y2n = hoop_search_region(hoop, expand)
        px1 = int(x1n * frame_w)
        py1 = int(y1n * frame_h)
        px2 = int(x2n * frame_w)
        py2 = int(y2n * frame_h)
        crop_w = px2 - px1
        crop_h = py2 - py1
        if crop_w < 8 or crop_h < 8:
            return None

        crop = frame_bgr[py1:py2, px1:px2]
        crop_det = self.detect_objects(crop, ["basketball"]).get("basketball")
        if crop_det is None:
            return None

        return _map_crop_detection_to_frame(crop_det, px1, py1, crop_w, crop_h, frame_w, frame_h)

    def detect_ball_with_hoop_fallback(
        self,
        frame_bgr: np.ndarray,
        hoop: HoopRoi | None,
        expand: float = 3.5,
        detect_hoop_classes: list[str] | None = None,
    ) -> tuple[BallDetection | None, dict[str, ObjectDetection | None]]:
        """Full-frame detect; if ball misses and hoop anchor exists, retry on hoop crop."""
        class_names = detect_hoop_classes or ["basketball"]
        detections = self.detect_objects(frame_bgr, class_names)
        ball_obj = detections.get("basketball")
        if ball_obj is not None:
            return (
                BallDetection(
                    x=ball_obj.x,
                    y=ball_obj.y,
                    w=ball_obj.w,
                    h=ball_obj.h,
                    conf=ball_obj.conf,
                ),
                detections,
            )

        if hoop is None:
            return None, detections

        region_ball = self.detect_ball_in_hoop_region(frame_bgr, hoop, expand)
        if region_ball is None or not ball_center_in_court(region_ball, self.court_polygon):
            return None, detections

        detections = dict(detections)
        detections["basketball"] = ObjectDetection(
            "basketball",
            region_ball.x,
            region_ball.y,
            region_ball.w,
            region_ball.h,
            region_ball.conf,
        )
        return region_ball, detections

    def detect_hoops(self, frame_bgr: np.ndarray) -> list[ObjectDetection]:
        """All hoop detections above threshold, best confidence first."""
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return []

        min_conf = self.config.hoop_conf
        predict_conf = min(self.config.conf, min_conf)

        predict = self.model.predict(
            source=frame_bgr,
            imgsz=self.config.imgsz,
            conf=predict_conf,
            iou=self.config.iou,
            classes=[self.class_id("hoop")],
            verbose=False,
            device=self._device,
        )

        if not predict:
            return []

        boxes = predict[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        hoops: list[ObjectDetection] = []
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx].item())
            if conf < min_conf:
                continue
            xyxy = boxes.xyxy[idx].tolist()
            hoops.append(_box_to_detection("hoop", xyxy, conf, width, height))

        hoops.sort(key=lambda h: h.conf, reverse=True)
        return hoops


def _ball_detection_to_object(ball: BallDetection) -> ObjectDetection:
    return ObjectDetection(
        label="basketball",
        x=ball.x,
        y=ball.y,
        w=ball.w,
        h=ball.h,
        conf=ball.conf,
    )


class HybridBallHoopDetector:
    """Fine-tuned ball detector + E-BARD hoop detector."""

    def __init__(
        self,
        ball_model_path: Path | str,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        config: DetectorConfig | None = None,
        court_polygon: list[tuple[float, float]] | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.court_polygon = court_polygon
        ball_config = self.config.model_copy(update={"conf": self.config.ball_conf})
        self._ball = BallDetector(model_path=ball_model_path, config=ball_config)
        self._hoop = EbardDetector(
            model_path=model_path,
            config=self.config,
            court_polygon=court_polygon,
        )

    def _filter_ball_for_court(self, ball: ObjectDetection | None) -> ObjectDetection | None:
        if ball is None:
            return None
        if not ball_center_in_court(ball, self.court_polygon):
            return None
        return ball

    def _detect_ball_object(self, frame_bgr: np.ndarray) -> ObjectDetection | None:
        ball = self._ball.detect_frame(frame_bgr)
        if ball is None:
            return None
        return self._filter_ball_for_court(_ball_detection_to_object(ball))

    def detect_objects(
        self,
        frame_bgr: np.ndarray,
        class_names: list[str],
    ) -> dict[str, ObjectDetection | None]:
        results: dict[str, ObjectDetection | None] = {name.lower(): None for name in class_names}
        names = [name.lower() for name in class_names]

        if "basketball" in names:
            results["basketball"] = self._detect_ball_object(frame_bgr)

        if "hoop" in names:
            hoops = self._hoop.detect_hoops(frame_bgr)
            results["hoop"] = hoops[0] if hoops else None

        return results

    def detect_ball_in_hoop_region(
        self,
        frame_bgr: np.ndarray,
        hoop: HoopRoi,
        expand: float = 3.5,
    ) -> BallDetection | None:
        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        x1n, y1n, x2n, y2n = hoop_search_region(hoop, expand)
        px1 = int(x1n * frame_w)
        py1 = int(y1n * frame_h)
        px2 = int(x2n * frame_w)
        py2 = int(y2n * frame_h)
        crop_w = px2 - px1
        crop_h = py2 - py1
        if crop_w < 8 or crop_h < 8:
            return None

        crop = frame_bgr[py1:py2, px1:px2]
        crop_ball = self._ball.detect_frame(crop)
        if crop_ball is None:
            return None

        mapped = BallDetection(
            x=(px1 + crop_ball.x * crop_w) / frame_w,
            y=(py1 + crop_ball.y * crop_h) / frame_h,
            w=crop_ball.w * crop_w / frame_w,
            h=crop_ball.h * crop_h / frame_h,
            conf=crop_ball.conf,
        )
        if not ball_center_in_court(mapped, self.court_polygon):
            return None
        return mapped

    def detect_ball_with_hoop_fallback(
        self,
        frame_bgr: np.ndarray,
        hoop: HoopRoi | None,
        expand: float = 3.5,
        detect_hoop_classes: list[str] | None = None,
    ) -> tuple[BallDetection | None, dict[str, ObjectDetection | None]]:
        class_names = detect_hoop_classes or ["basketball"]
        detections = self.detect_objects(frame_bgr, class_names)
        ball_obj = detections.get("basketball")
        if ball_obj is not None:
            return (
                BallDetection(
                    x=ball_obj.x,
                    y=ball_obj.y,
                    w=ball_obj.w,
                    h=ball_obj.h,
                    conf=ball_obj.conf,
                ),
                detections,
            )

        if hoop is None:
            return None, detections

        region_ball = self.detect_ball_in_hoop_region(frame_bgr, hoop, expand)
        if region_ball is None:
            return None, detections

        detections = dict(detections)
        detections["basketball"] = _ball_detection_to_object(region_ball)
        return region_ball, detections

    def detect_hoops(self, frame_bgr: np.ndarray) -> list[ObjectDetection]:
        return self._hoop.detect_hoops(frame_bgr)


def create_ball_hoop_detector(
    model_path: Path | str,
    config: DetectorConfig,
    court_polygon: list[tuple[float, float]] | None = None,
    ball_model_path: Path | str | None = None,
) -> EbardDetector | HybridBallHoopDetector:
    if ball_model_path is not None:
        return HybridBallHoopDetector(
            ball_model_path=ball_model_path,
            model_path=model_path,
            config=config,
            court_polygon=court_polygon,
        )
    return EbardDetector(
        model_path=model_path,
        config=config,
        court_polygon=court_polygon,
    )
