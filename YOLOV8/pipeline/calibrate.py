"""Stage 3: pixel → court-foot homography from a manual calibration file.

Automatic court-line / keypoint detection is unreliable on broadcast and
sideline video (occlusion, lens distortion, moving camera, painted logos).
This MVP loads `court-calibration-v1` JSON produced by yolo/court-marker.html
(≥4 landmark pairs) and computes OpenCV findHomography.

A future court-keypoint model (baseline, FT line, 3PT arc, corners) could
replace the manual clicker; until then, mark a new clip in court-marker.html.

Existing calibrations in this repo are in meters (FIBA). Output coordinates
are converted to feet.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.schemas import METERS_TO_FEET, Court

# Reject projections this far outside the court (feet).
DEFAULT_COURT_MARGIN_FT = 3.0


class CourtProjector:
    """Maps normalized pixel (0-1) coordinates to court feet."""

    def __init__(self, court: Court, matrix: np.ndarray, native_to_feet: float) -> None:
        self.court = court
        self.matrix = matrix
        self.native_to_feet = native_to_feet

    @classmethod
    def from_calibration_file(cls, path: str | Path) -> CourtProjector:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        points = data.get("points") or []
        if len(points) < 4:
            raise ValueError(
                f"Calibration {path} has {len(points)} points; at least 4 are required. "
                "Mark landmarks in yolo/court-marker.html and export court_calibration.json."
            )
        court_info = data.get("court") or {}
        length = float(court_info.get("length", 28.0))
        width = float(court_info.get("width", 15.0))
        unit = str(court_info.get("unit", "m"))
        layout = court_info.get("layout")
        if layout not in ("full", "half"):
            layout = "half" if length < 20 else "full"
        scale = METERS_TO_FEET if unit in {"m", "meter", "meters"} else 1.0

        pixel_pts = np.array(
            [[float(p["pixel"]["x"]), float(p["pixel"]["y"])] for p in points],
            dtype=np.float64,
        )
        court_pts = np.array(
            [[float(p["court"]["x"]), float(p["court"]["y"])] for p in points],
            dtype=np.float64,
        )
        method = cv2.RANSAC if len(points) > 4 else 0
        matrix, _ = cv2.findHomography(pixel_pts, court_pts, method, 3.0)
        if matrix is None:
            raise ValueError(f"Could not compute homography from {path}.")

        court = Court(
            layout=layout,
            length=round(length * scale, 3),
            width=round(width * scale, 3),
            unit="ft",
            preset=court_info.get("preset"),
        )
        return cls(court=court, matrix=matrix, native_to_feet=scale)

    def project(
        self,
        px: float,
        py: float,
        margin_ft: float = DEFAULT_COURT_MARGIN_FT,
        clamp: bool = True,
    ) -> tuple[float, float] | None:
        src = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.matrix)
        x = float(dst[0, 0, 0]) * self.native_to_feet
        y = float(dst[0, 0, 1]) * self.native_to_feet
        if (
            x < -margin_ft
            or x > self.court.length + margin_ft
            or y < -margin_ft
            or y > self.court.width + margin_ft
        ):
            return None
        if clamp:
            x = min(max(x, 0.0), self.court.length)
            y = min(max(y, 0.0), self.court.width)
        return round(x, 3), round(y, 3)


def _project_points(points: list[dict], projector: CourtProjector) -> list[dict]:
    out: list[dict] = []
    for pt in points:
        mapped = projector.project(float(pt["px"]), float(pt["py"]))
        row = dict(pt)
        if mapped is None:
            row["x"] = None
            row["y"] = None
        else:
            row["x"], row["y"] = mapped
        out.append(row)
    return out


def default_calibration_path(video_path: str | Path | None = None) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    local = root / "court_calibration.json"
    if local.exists():
        return local
    sibling = root.parent / "yolo" / "court_calibration.json"
    if sibling.exists():
        if video_path is None:
            return sibling
        try:
            data = json.loads(sibling.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return sibling
        clip_key = str(data.get("clipKey") or "")
        if clip_key and Path(video_path).name == clip_key:
            return sibling
        return sibling
    return None


def run_calibrate(
    output_dir: str | Path,
    calibration_path: str | Path | None = None,
    video_path: str | Path | None = None,
) -> dict:
    """Project tracks.json into court feet. Writes court_tracks.json."""
    output_dir = Path(output_dir)
    tracks_path = output_dir / "tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing {tracks_path}. Run --stage track first.")

    cal_path = Path(calibration_path) if calibration_path else default_calibration_path(video_path)
    if cal_path is None or not cal_path.exists():
        raise FileNotFoundError(
            "No court calibration file. Mark ≥4 landmarks in yolo/court-marker.html "
            "and pass --calibration court_calibration.json. Automatic court-line "
            "detection is not implemented (unreliable on broadcast/sideline video)."
        )

    projector = CourtProjector.from_calibration_file(cal_path)
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    players_out = []
    on_court = 0
    total = 0
    for player in tracks.get("players") or []:
        points = _project_points(player.get("points") or [], projector)
        total += len(points)
        on_court += sum(1 for p in points if p.get("x") is not None)
        players_out.append({**player, "points": points})

    ball = tracks.get("ball") or {}
    ball_points = _project_points(ball.get("points") or [], projector)

    artifact = {
        "format": tracks.get("format", "play-tracks-v1"),
        "video": tracks.get("video"),
        "sampleFps": tracks.get("sampleFps"),
        "frameCount": tracks.get("frameCount"),
        "calibration": str(cal_path.resolve()),
        "court": {
            "layout": projector.court.layout,
            "length": projector.court.length,
            "width": projector.court.width,
            "unit": "ft",
            "preset": projector.court.preset,
        },
        "players": players_out,
        "ball": {**ball, "points": ball_points},
        "onCourtFraction": round(on_court / total, 4) if total else 0.0,
    }
    out_json = output_dir / "court_tracks.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"Stage 3 calibrate: {on_court}/{total} player points on court "
        f"({artifact['court']['length']}×{artifact['court']['width']} ft) → {out_json}"
    )
    return artifact
