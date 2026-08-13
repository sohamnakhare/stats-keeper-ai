"""Pixel -> court-meter homography from court-marker.html calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from player_config import HomographyConfig


@dataclass(frozen=True)
class CourtDimensions:
    length: float  # meters, along x
    width: float  # meters, along y
    unit: str = "m"
    preset: str | None = None
    layout: str = "full"  # "full" | "half"


class CourtProjector:
    """Projects normalized pixel coordinates to court meters.

    Calibration pixels are normalized (0-1), so the homography maps
    normalized-pixel space directly to court space; this is consistent
    regardless of the resolution frames were sampled at.
    """

    def __init__(
        self,
        court: CourtDimensions,
        homography: np.ndarray,
        config: HomographyConfig,
    ) -> None:
        self.court = court
        self.matrix = homography
        self.config = config

    @classmethod
    def from_calibration_file(
        cls, path: Path | str, config: HomographyConfig | None = None
    ) -> CourtProjector:
        config = config or HomographyConfig()
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        points = data.get("points") or []
        if len(points) < 4:
            raise ValueError(
                f"Calibration file {path} has {len(points)} points; at least 4 are required."
            )

        court_info = data.get("court") or {}
        length = float(court_info.get("length", 28.0))
        layout = court_info.get("layout")
        if layout not in ("full", "half"):
            # Legacy calibrations without layout: short courts were half/3x3.
            layout = "half" if length < 20 else "full"
        court = CourtDimensions(
            length=length,
            width=float(court_info.get("width", 15.0)),
            unit=str(court_info.get("unit", "m")),
            preset=court_info.get("preset"),
            layout=layout,
        )

        pixel_pts = np.array(
            [[float(p["pixel"]["x"]), float(p["pixel"]["y"])] for p in points],
            dtype=np.float64,
        )
        court_pts = np.array(
            [[float(p["court"]["x"]), float(p["court"]["y"])] for p in points],
            dtype=np.float64,
        )

        method = cv2.RANSAC if len(points) > 4 else 0
        matrix, _ = cv2.findHomography(
            pixel_pts, court_pts, method, config.ransac_reproj_threshold
        )
        if matrix is None:
            raise ValueError(f"Could not compute homography from {path}.")

        return cls(court=court, homography=matrix, config=config)

    def project(self, px: float, py: float) -> tuple[float, float] | None:
        """Project a normalized pixel point to court meters.

        Returns None when the point lands further than `court_margin` outside
        the court (typically a bench player, spectator, or bad detection).
        """
        src = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.matrix)
        x, y = float(dst[0, 0, 0]), float(dst[0, 0, 1])

        margin = self.config.court_margin
        if x < -margin or x > self.court.length + margin:
            return None
        if y < -margin or y > self.court.width + margin:
            return None

        if self.config.clamp_to_court:
            x = min(max(x, 0.0), self.court.length)
            y = min(max(y, 0.0), self.court.width)
        return x, y
