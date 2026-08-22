"""Tests for track merge and pixel-then-reproject gap fill."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent
WORKER_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(WORKER_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from court_overlay import PRESETS, paint_quads  # noqa: E402
from court_projection import PoseCourtProjector  # noqa: E402
from player_config import HomographyConfig, PipelineConfig  # noqa: E402
from track_players import fill_track_gaps, merge_tracks  # noqa: E402


def _player(
    track_id: int,
    t0: float,
    t1: float,
    *,
    team: str | None,
    color: list[int] | None = None,
    x0: float = 5.0,
    y0: float = 5.0,
) -> dict:
    return {
        "trackId": track_id,
        "team": team,
        "jerseyNumber": None,
        "jerseyConfidence": None,
        "jerseyColor": color,
        "mergedTrackIds": [],
        "points": [
            {"t": t0, "frame": int(t0 * 10), "px": 0.4, "py": 0.5, "x": x0, "y": y0, "conf": 0.9},
            {
                "t": t1,
                "frame": int(t1 * 10),
                "px": 0.41,
                "py": 0.51,
                "x": x0 + 0.2,
                "y": y0 + 0.1,
                "conf": 0.9,
            },
        ],
    }


def test_merge_allows_null_team_and_copies_label() -> None:
    config = PipelineConfig()
    unlabeled = _player(1, 0.0, 1.0, team=None, color=[200, 20, 20])
    labeled = _player(2, 2.0, 3.0, team="home", color=[210, 30, 30], x0=5.5, y0=5.2)
    merged = merge_tracks([unlabeled, labeled], config)
    assert len(merged) == 1
    assert merged[0]["team"] == "home"
    assert 2 in merged[0]["mergedTrackIds"]


def test_merge_rejects_conflicting_teams() -> None:
    config = PipelineConfig()
    a = _player(1, 0.0, 1.0, team="home", color=[200, 20, 20])
    b = _player(2, 2.0, 3.0, team="away", color=[20, 20, 200], x0=5.5, y0=5.2)
    merged = merge_tracks([a, b], config)
    assert len(merged) == 2


def test_gap_fill_reprojects_through_homography() -> None:
    paint = paint_quads(PRESETS["fibaHalf"])[0]
    ox, oy = 0.2, 0.3

    def kpts(scale: float) -> list[dict]:
        return [{"x": ox + scale * x, "y": oy + scale * y, "conf": 0.99} for x, y in paint]

    payload = {
        "preset": "fibaHalf",
        "keypointOrder": [0, 1, 2, 3],
        "frames": [
            {"index": 0, "t_sec": 0.0, "keypoints": kpts(0.02)},
            {"index": 1, "t_sec": 2.0, "keypoints": kpts(0.04)},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )
        cx, cy = float(np.mean(paint[:, 0])), float(np.mean(paint[:, 1]))
        px = ox + 0.02 * cx
        py = oy + 0.02 * cy
        at_start = projector.project(px, py, t_sec=0.0)
        at_end = projector.project(px, py, t_sec=2.0)
        assert at_start is not None and at_end is not None

        config = PipelineConfig()
        config.detection.sample_fps = 1.0
        players = [
            {
                "trackId": 1,
                "team": "home",
                "jerseyNumber": None,
                "jerseyConfidence": None,
                "jerseyColor": None,
                "mergedTrackIds": [],
                "points": [
                    {
                        "t": 0.0,
                        "frame": 0,
                        "px": px,
                        "py": py,
                        "x": at_start[0],
                        "y": at_start[1],
                        "conf": 0.9,
                    },
                    {
                        "t": 2.0,
                        "frame": 2,
                        "px": px,
                        "py": py,
                        "x": at_end[0],
                        "y": at_end[1],
                        "conf": 0.9,
                    },
                ],
            }
        ]
        inserted = fill_track_gaps(players, config, projector=projector)
        assert inserted >= 1
        mid = next(p for p in players[0]["points"] if p.get("interpolated"))
        assert mid["x"] is not None
        assert abs(mid["x"] - at_start[0]) < 0.2
        meter_mid = (at_start[0] + at_end[0]) / 2.0
        assert abs(mid["x"] - meter_mid) > 0.2
