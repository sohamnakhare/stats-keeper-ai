"""Selective jersey-number OCR: score crops during tracking, OCR only the best.

For each track we keep the best-K torso crops (scored by size and sharpness)
and run PaddleOCR once at the end, assigning numbers by confidence-weighted
voting so a few misreads don't flip the result.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from player_config import JerseyOcrConfig
from player_tracker import PlayerObservation

_NUMBER_RE = re.compile(r"^\d{1,2}$")


@dataclass(frozen=True)
class JerseyResult:
    number: str
    confidence: float  # share of confidence-weighted votes for the winner
    votes: int


@dataclass(order=True)
class _ScoredCrop:
    score: float
    seq: int  # unique tie-breaker so heapq never compares arrays
    crop: np.ndarray = field(compare=False)


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class JerseyOcrCollector:
    """Collects promising crops per track; OCRs them lazily in finalize()."""

    def __init__(self, config: JerseyOcrConfig) -> None:
        self.config = config
        self._crops: dict[int, list[_ScoredCrop]] = defaultdict(list)
        self._seen: dict[int, int] = defaultdict(int)
        self._seq = 0

    def observe(self, frame_bgr: np.ndarray, obs: PlayerObservation) -> None:
        if not self.config.enabled or obs.label != "player":
            return

        self._seen[obs.track_id] += 1
        if (self._seen[obs.track_id] - 1) % max(1, self.config.crop_interval) != 0:
            return

        height, width = frame_bgr.shape[:2]
        bbox_h_px = obs.bbox_height * height
        if bbox_h_px < self.config.min_bbox_height_px:
            return

        # Torso region where the jersey number lives (chest or back).
        bx1 = obs.x1 * width
        bx2 = obs.x2 * width
        by1 = obs.y1 * height
        bw = bx2 - bx1
        bh = obs.bbox_height * height
        x1 = int(bx1 + bw * 0.1)
        x2 = int(bx2 - bw * 0.1)
        y1 = int(by1 + bh * 0.12)
        y2 = int(by1 + bh * 0.55)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return

        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = _sharpness(gray)
        if sharp < self.config.min_sharpness:
            return

        score = bbox_h_px * min(sharp / 100.0, 3.0)
        heap = self._crops[obs.track_id]
        self._seq += 1
        item = _ScoredCrop(score=score, seq=self._seq, crop=crop.copy())
        if len(heap) < self.config.best_k:
            heapq.heappush(heap, item)
        elif score > heap[0].score:
            heapq.heapreplace(heap, item)

    # --- OCR ---

    def _make_reader(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for jersey number OCR. "
                "Install with: pip install paddleocr paddlepaddle "
                "(or disable with jerseyOcr.enabled=false)"
            ) from exc
        import os

        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "ERROR")
        device = "gpu" if self.config.gpu else "cpu"
        return PaddleOCR(lang="en", device=device)

    def _read_numbers(self, reader, crop: np.ndarray) -> list[tuple[str, float]]:
        """OCR a crop and return plausible jersey numbers with confidence."""
        scale = self.config.upscale
        if scale != 1.0:
            h, w = crop.shape[:2]
            crop = cv2.resize(
                crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )

        results = reader.predict(crop)
        pairs: list[tuple[str, float]] = []
        if not results or results[0] is None:
            return pairs

        result = results[0]
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        for text, score in zip(texts, scores):
            conf = float(score)
            if conf < self.config.min_ocr_conf:
                continue
            cleaned = re.sub(r"[^\d]", "", str(text).strip())
            if _NUMBER_RE.match(cleaned):
                pairs.append((cleaned, conf))
        return pairs

    def finalize(self) -> dict[int, JerseyResult]:
        """Run OCR on collected crops and vote per track."""
        if not self.config.enabled:
            return {}

        tracks_with_crops = {tid: heap for tid, heap in self._crops.items() if heap}
        if not tracks_with_crops:
            return {}

        reader = self._make_reader()
        results: dict[int, JerseyResult] = {}

        for track_id, heap in tracks_with_crops.items():
            votes: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for item in heap:
                for number, conf in self._read_numbers(reader, item.crop):
                    votes[number] += conf
                    counts[number] += 1

            if not votes:
                continue

            total = sum(votes.values())
            winner = max(votes, key=lambda n: votes[n])
            share = votes[winner] / total if total > 0 else 0.0
            if share >= self.config.min_vote_share and counts[winner] >= self.config.min_votes:
                results[track_id] = JerseyResult(
                    number=winner, confidence=round(share, 3), votes=counts[winner]
                )

        return results
