"""OCR reader for scoreboard regions using PaddleOCR."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None  # type: ignore

from .schemas import ScoreboardRegion


def _format_clock(minutes: int, seconds: int) -> str | None:
    """Format a valid basketball game clock as M:SS / MM:SS."""
    if minutes < 0 or minutes > 15 or seconds < 0 or seconds > 59:
        return None
    return f"{minutes}:{seconds:02d}"


def game_clock_candidates(text: str | None, max_minutes: int = 15) -> list[str]:
    """Return all plausible M:SS interpretations of noisy OCR clock text."""
    if not text:
        return []

    normalized = (
        text.upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
    )
    cleaned = re.sub(r"[^\d:.]+", "", normalized)
    if not cleaned:
        return []

    found: list[str] = []

    def add(minutes: int, seconds: int) -> None:
        if minutes > max_minutes:
            return
        formatted = _format_clock(minutes, seconds)
        if formatted is not None and formatted not in found:
            found.append(formatted)

    mm_ss_match = re.match(r"^(\d{1,2}):(\d{2})(?:\.\d)?$", cleaned)
    if mm_ss_match:
        add(int(mm_ss_match.group(1)), int(mm_ss_match.group(2)))
        return found

    digits = re.sub(r"[^\d]", "", cleaned)
    if not digits:
        return []

    if len(digits) == 3:
        add(int(digits[0]), int(digits[1:]))

    elif len(digits) == 4:
        # Prefer single-minute readings for broadcast overlays (9:58), then MMSS.
        add(int(digits[0]), int(digits[2:]))       # drop 2nd (colon noise): 9149 -> 9:49
        add(int(digits[0]), int(digits[1:3]))       # drop last: 9581 -> 9:58
        add(int(digits[0]), int(digits[1] + digits[3]))  # drop 3rd: 9556 -> 9:56
        add(int(digits[:2]), int(digits[2:]))       # MMSS: 1215 -> 12:15

    elif len(digits) == 5:
        add(int(digits[0]), int(digits[2:4]))
        add(int(digits[0]), int(digits[3:]))
        add(int(digits[:2]), int(digits[2:4]))

    return found


def canonicalize_game_clock(text: str | None, max_minutes: int = 15) -> str | None:
    """Normalize noisy OCR clock text to M:SS / MM:SS."""
    candidates = game_clock_candidates(text, max_minutes=max_minutes)
    return candidates[0] if candidates else None


@dataclass
class OCRResult:
    """Raw OCR result with confidence."""

    text: str
    confidence: float


class ScoreboardOCRReader:
    """Read and parse scoreboard regions using PaddleOCR."""

    def __init__(self, gpu: bool = True):
        if PaddleOCR is None:
            raise ImportError(
                "paddleocr is required for scoreboard OCR. "
                "Install with: pip install -r requirements.txt"
            )
        import os
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "ERROR")
        device = "gpu" if gpu else "cpu"
        self.reader = PaddleOCR(
            lang="en",
            device=device,
        )

    def crop_region(
        self,
        frame: np.ndarray,
        region: ScoreboardRegion,
    ) -> np.ndarray:
        """Crop a normalized region from the frame."""
        h, w = frame.shape[:2]
        x1 = int(region.x * w)
        y1 = int(region.y * h)
        x2 = int((region.x + region.w) * w)
        y2 = int((region.y + region.h) * h)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        return frame[y1:y2, x1:x2]

    def preprocess_for_ocr(self, crop: np.ndarray, scale: float = 3.0) -> np.ndarray:
        """Preprocess cropped region for better OCR accuracy on broadcast overlays."""
        if crop.size == 0:
            return crop

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        if scale != 1.0:
            h, w = gray.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Sharpen to improve edge clarity on broadcast fonts.
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

        # Simple Otsu threshold works better than adaptive for broadcast overlays
        # because the background is more uniform (semi-transparent dark bar).
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def _readtext(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Run PaddleOCR on an image and return (text, confidence) pairs."""
        if image.size == 0:
            return []

        # PaddleOCR expects BGR uint8.
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # v3 API: predict() returns a list of result objects, one per input image.
        # Each result is dict-like with keys "rec_texts" and "rec_scores".
        results = self.reader.predict(image)
        pairs: list[tuple[str, float]] = []

        if not results:
            return pairs

        result = results[0]
        if result is None:
            return pairs

        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        for text, score in zip(texts, scores):
            pairs.append((str(text), float(score)))

        return pairs

    def _region_pattern(self, region_name: str) -> re.Pattern[str]:
        if region_name in {"game_clock", "shot_clock"}:
            return re.compile(r"^\d{1,2}:\d{2}(?:\.\d)?$|^\d{3,4}$")
        if region_name in {"home_score", "away_score"}:
            return re.compile(r"^\d{1,3}$")
        if region_name == "quarter":
            return re.compile(r"^(?:Q?\d|[1-4](?:ST|ND|RD|TH)?|OT|[1-3]OT)$", re.IGNORECASE)
        return re.compile(r".+")

    def _normalize_candidate(self, text: str, region_name: str) -> str:
        cleaned = text.upper().strip()
        if region_name in {"game_clock", "shot_clock", "quarter"}:
            cleaned = (
                cleaned.replace("O", "0")
                .replace("I", "1")
                .replace("L", "1")
                .replace("S", "5")
            )
        if region_name in {"home_score", "away_score"}:
            cleaned = re.sub(r"[^\d]", "", cleaned)
        elif region_name in {"game_clock", "shot_clock"}:
            cleaned = re.sub(r"[^\d:.]", "", cleaned)
        return cleaned

    def _score_candidate(
        self,
        candidate: str,
        conf: float,
        region_name: str,
    ) -> float:
        if not candidate:
            return -1.0
        score = conf
        if self._region_pattern(region_name).match(candidate):
            score += 1.0
        if region_name in {"game_clock", "shot_clock"} and ":" in candidate:
            score += 0.5
        if region_name in {"home_score", "away_score"} and len(candidate) <= 3:
            score += 0.2
        if len(candidate) > 6:
            score -= 0.5
        return score

    def read_region_raw(
        self,
        frame: np.ndarray,
        region: ScoreboardRegion,
        region_name: str,
        preprocess: bool = True,
    ) -> OCRResult:
        """Run OCR on a region and return the best candidate text with confidence."""
        crop = self.crop_region(frame, region)
        if crop.size == 0:
            return OCRResult(text="", confidence=0.0)

        # Build image variants: preprocessed, inverted, original color.
        variants: list[np.ndarray] = []
        if preprocess:
            processed = self.preprocess_for_ocr(crop)
            if processed.size > 0:
                variants.append(processed)
                variants.append(cv2.bitwise_not(processed))
        variants.append(crop)

        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for variant in variants:
            for text, conf in self._readtext(variant):
                candidate = self._normalize_candidate(text, region_name)
                cand_score = self._score_candidate(candidate, conf, region_name)
                if cand_score > best_score:
                    best_score = cand_score
                    best_text = candidate
                    best_conf = conf

        return OCRResult(text=best_text, confidence=best_conf)

    def parse_clock(self, text: str) -> float | None:
        """Parse game/shot clock text to total seconds."""
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

    def parse_score(self, text: str) -> int | None:
        """Parse score text to integer."""
        if not text:
            return None

        digits = re.sub(r"[^\d]+", "", text)
        if not digits:
            return None

        try:
            score = int(digits)
            if 0 <= score <= 999:
                return score
        except ValueError:
            pass

        return None

    def parse_quarter(self, text: str) -> int | None:
        """Parse quarter/period indicator.

        Handles: "1", "Q1", "1ST", "2ND", "3RD", "4TH", "OT", "2OT", etc.
        """
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

        ordinal_map = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}
        for pattern, value in ordinal_map.items():
            if pattern in upper:
                return value

        return None

    def read_all_regions(
        self,
        frame: np.ndarray,
        regions: dict[str, ScoreboardRegion],
    ) -> dict[str, str]:
        """Read all scoreboard regions and return raw OCR text."""
        results = {}
        for name, region in regions.items():
            ocr_result = self.read_region_raw(frame, region, name)
            results[name] = ocr_result.text
        return results

    def read_and_parse(
        self,
        frame: np.ndarray,
        regions: dict[str, ScoreboardRegion],
    ) -> dict:
        """Read all regions and parse into structured data."""
        raw = self.read_all_regions(frame, regions)

        parsed: dict = {"raw_ocr": raw}

        if "game_clock" in raw:
            parsed["game_clock_sec"] = self.parse_clock(raw["game_clock"])

        if "home_score" in raw:
            parsed["home_score"] = self.parse_score(raw["home_score"])

        if "away_score" in raw:
            parsed["away_score"] = self.parse_score(raw["away_score"])

        if "shot_clock" in raw:
            parsed["shot_clock_sec"] = self.parse_clock(raw["shot_clock"])

        if "quarter" in raw:
            parsed["quarter"] = self.parse_quarter(raw["quarter"])

        return parsed
