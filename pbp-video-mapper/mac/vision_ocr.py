"""Read scorebug crops with Apple's Vision framework."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from worker.ocr_reader import canonicalize_game_clock, game_clock_candidates
from worker.schemas import ScoreboardRegion

logger = logging.getLogger(__name__)

# Vision misses broadcast digits when the crop is only a few pixels tall.
_MIN_CROP_HEIGHT = 64


def crop_region(frame: np.ndarray, region: ScoreboardRegion) -> np.ndarray:
    """Crop a normalized region from a BGR frame or scorebug crop."""
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(region.x * width)))
    y1 = max(0, min(height, int(region.y * height)))
    x2 = max(0, min(width, int((region.x + region.w) * width)))
    y2 = max(0, min(height, int((region.y + region.h) * height)))
    return frame[y1:y2, x1:x2]


def _upscale(image: np.ndarray) -> np.ndarray:
    """Grow a short crop so Vision has enough pixels to read the glyphs."""
    height, width = image.shape[:2]
    if height <= 0 or height >= _MIN_CROP_HEIGHT:
        return image
    scale = _MIN_CROP_HEIGHT / height
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), _MIN_CROP_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def _box_from_vision(observation: object) -> tuple[float, float, float, float]:
    """Vision boxes use a bottom-left origin. Field boxes use the top-left."""
    rect = observation.boundingBox()  # type: ignore[attr-defined]
    origin = getattr(rect, "origin", None)
    size = getattr(rect, "size", None)
    if origin is not None and size is not None:
        x = float(origin.x)
        y = float(origin.y)
        width = float(size.width)
        height = float(size.height)
    else:
        x, y, width, height = (float(value) for value in rect)
    return (x, 1.0 - (y + height), width, height)


def _read_observations(image: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Return (text, confidence, top-left box) for each string Vision finds."""
    import Vision
    from Foundation import NSData

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return []
    payload = encoded.tobytes()
    ns_data = NSData.dataWithBytes_length_(payload, len(payload))

    request = Vision.VNRecognizeTextRequest.alloc().init()
    accurate = getattr(Vision, "VNRequestTextRecognitionLevelAccurate", 1)
    request.setRecognitionLevel_(accurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])
    try:
        revisions = Vision.VNRecognizeTextRequest.supportedRevisions()
        request.setRevision_(max(int(revision) for revision in revisions))
    except Exception:
        logger.debug("Vision revision selection skipped", exc_info=True)

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})
    success, error = handler.performRequests_error_([request], None)
    if not success or error is not None:
        message = str(error) if error is not None else "Vision request failed"
        raise RuntimeError(message)

    observations: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        if not text:
            continue
        observations.append(
            (text, float(candidates[0].confidence()), _box_from_vision(observation))
        )
    return observations


def _choose_text(field: str, observations: list[tuple[str, float]]) -> str:
    """Pick one observation. Joining them mixes neighboring digits across frames."""
    if not observations:
        return ""

    def rank(item: tuple[str, float]) -> tuple[int, float]:
        text, confidence = item
        if field in {"game_clock", "shot_clock"} and game_clock_candidates(text):
            return (2, confidence)
        if field in {"home_score", "away_score"} and parse_score(text) is not None:
            return (2, confidence)
        if field == "quarter" and parse_quarter(text) is not None:
            return (2, confidence)
        return (1, confidence)

    text, confidence = max(observations, key=rank)
    logger.info(
        "vision %s chose %r (%.2f) from %s",
        field,
        text,
        confidence,
        [(item[0], round(item[1], 2)) for item in observations],
    )
    return text


def _to_scorebug(
    box: tuple[float, float, float, float],
    region: ScoreboardRegion,
) -> tuple[float, float, float, float]:
    """Move a box from a field crop into scorebug coordinates."""
    x, y, width, height = box
    return (
        region.x + x * region.w,
        region.y + y * region.h,
        width * region.w,
        height * region.h,
    )


def _read_field(
    image: np.ndarray,
    name: str,
    region: ScoreboardRegion,
) -> tuple[str, list[tuple[str, float, tuple[float, float, float, float]]]]:
    """Read one marking and return its text plus boxes in scorebug coordinates."""
    field_crop = crop_region(image, region)
    if field_crop.size == 0:
        logger.info("vision %s empty crop", name)
        return "", []

    height, width = field_crop.shape[:2]
    prepared = _upscale(field_crop)
    prepared_height, prepared_width = prepared.shape[:2]
    if (prepared_width, prepared_height) != (width, height):
        logger.info(
            "vision %s upscaled crop %sx%s to %sx%s",
            name,
            width,
            height,
            prepared_width,
            prepared_height,
        )
    observations = _read_observations(prepared)
    if not observations:
        logger.info("vision %s no text in %sx%s crop", name, prepared_width, prepared_height)
        return "", []
    text = _choose_text(name, [(item[0], item[1]) for item in observations])
    mapped = [
        (item[0], item[1], _to_scorebug(item[2], region))
        for item in observations
    ]
    return text, mapped


_latest_observations: list[tuple[str, float, tuple[float, float, float, float]]] = []


def latest_observations() -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Vision boxes from the most recent scorebug read."""
    return _latest_observations


_FIELD_LABELS = {
    "game_clock": "clock",
    "home_score": "home",
    "away_score": "away",
    "shot_clock": "shot",
    "quarter": "qtr",
}
_MARKING_COLOR = (0, 220, 0)
_VISION_COLOR = (0, 140, 255)
_DEBUG_MIN_HEIGHT = 160


def inspect_regions(
    image: np.ndarray,
    regions: dict[str, ScoreboardRegion],
) -> tuple[dict[str, str], list[tuple[str, float, tuple[float, float, float, float]]]]:
    """Read each marking on its own and return that field's text."""
    global _latest_observations
    if image.size == 0:
        logger.info("vision scorebug empty crop")
        _latest_observations = []
        return {name: "" for name in regions}, []

    texts: dict[str, str] = {}
    observations: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for name, region in regions.items():
        text, field_observations = _read_field(image, name, region)
        texts[name] = text
        observations.extend(field_observations)
    _latest_observations = observations
    return texts, observations


def recognize_regions(
    image: np.ndarray,
    regions: dict[str, ScoreboardRegion],
) -> dict[str, str]:
    """Read each marked field and store the text under that field name."""
    texts, _observations = inspect_regions(image, regions)
    return texts


def _pixel_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        int(round(x * width)),
        int(round(y * height)),
        int(round((x + w) * width)),
        int(round((y + h) * height)),
    )


def _draw_labeled_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _pixel_box(box, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    text_y = y1 - 4
    if text_y - text_height < 0:
        text_y = min(height - 2, y2 + text_height + 2)
    text_x = max(0, min(x1, width - text_width - 1))
    cv2.rectangle(
        image,
        (text_x, text_y - text_height - 1),
        (min(width - 1, text_x + text_width + 1), min(height - 1, text_y + baseline)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_debug(
    image: np.ndarray,
    regions: dict[str, ScoreboardRegion],
    observations: list[tuple[str, float, tuple[float, float, float, float]]],
    texts: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw marked fields in green and the text Vision read inside them in orange."""
    height, width = image.shape[:2]
    if height <= 0:
        return image
    canvas = image
    if height < _DEBUG_MIN_HEIGHT:
        scale = _DEBUG_MIN_HEIGHT / height
        canvas = cv2.resize(
            image,
            (max(1, int(round(width * scale))), _DEBUG_MIN_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
    canvas = canvas.copy()
    chosen = texts or {}
    for name, region in regions.items():
        label = _FIELD_LABELS.get(name, name)
        if chosen.get(name):
            label = f"{label}: {chosen[name]}"
        _draw_labeled_box(
            canvas,
            (region.x, region.y, region.w, region.h),
            _MARKING_COLOR,
            label,
        )
    for text, _confidence, box in observations:
        _draw_labeled_box(canvas, box, _VISION_COLOR, text)
    return canvas


def save_crop(directory: Path, t_sec: float, name: str, image: np.ndarray) -> None:
    """Write a crop so the box can be checked by eye."""
    if image.size == 0:
        logger.info("vision %s t=%.3fs empty crop, not saved", name, t_sec)
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"t{t_sec:08.3f}_{name}.png"
    if not cv2.imwrite(str(path), image):
        logger.warning("vision failed to write %s", path)


def parse_score(text: str | None) -> int | None:
    """Parse a score from OCR text. Keeps values from 0 through 999."""
    if not text:
        return None
    normalized = (
        text.upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
    )
    digits = re.sub(r"[^\d]+", "", normalized)
    if not digits:
        return None
    score = int(digits)
    if 0 <= score <= 999:
        return score
    return None


def parse_clock_seconds(text: str | None) -> float | None:
    """Parse a clock to seconds, including a bare shot-clock number."""
    canonical = canonicalize_game_clock(text)
    if canonical is None:
        digits = re.sub(r"[^\d]", "", text or "")
        if digits.isdigit() and 1 <= len(digits) <= 2:
            seconds = int(digits)
            if seconds <= 35:
                return float(seconds)
        return None
    minutes, seconds = canonical.split(":")
    return int(minutes) * 60 + int(seconds)


def parse_quarter(text: str | None) -> int | None:
    """Parse a quarter or overtime marker."""
    if not text:
        return None
    upper = text.upper().strip()
    ot_match = re.search(r"(\d)?OT", upper)
    if ot_match:
        ot_num = int(ot_match.group(1)) if ot_match.group(1) else 1
        return 4 + ot_num
    q_match = re.search(r"Q?(\d)", upper)
    if q_match:
        return int(q_match.group(1))
    for pattern, value in {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}.items():
        if pattern in upper:
            return value
    return None


def parse_fields(raw: dict[str, str]) -> dict:
    """Turn raw field text into clock, score, and quarter values."""
    parsed: dict = {"raw_ocr": raw}
    if "game_clock" in raw:
        candidates = game_clock_candidates(raw["game_clock"])
        parsed["game_clock"] = candidates[0] if candidates else None
    if "home_score" in raw:
        parsed["home_score"] = parse_score(raw["home_score"])
    if "away_score" in raw:
        parsed["away_score"] = parse_score(raw["away_score"])
    if "shot_clock" in raw:
        parsed["shot_clock_sec"] = parse_clock_seconds(raw["shot_clock"])
    if "quarter" in raw:
        parsed["quarter"] = parse_quarter(raw["quarter"])
    return parsed


_recognizer: VisionRecognizer | None = None


class VisionRecognizer:
    """Process-wide Vision reader. The framework loads on the first crop."""

    def read_regions(
        self,
        image: np.ndarray,
        regions: dict[str, ScoreboardRegion],
    ) -> dict[str, str]:
        return recognize_regions(image, regions)


def get_recognizer() -> VisionRecognizer:
    """Return the process-wide recognizer, creating it on first use."""
    global _recognizer
    if _recognizer is None:
        _recognizer = VisionRecognizer()
    return _recognizer


def reset_recognizer() -> None:
    """Drop the cached recognizer. Used by tests."""
    global _recognizer
    _recognizer = None
