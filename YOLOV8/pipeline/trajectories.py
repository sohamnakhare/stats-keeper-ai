"""Stage 4: interpolate detection gaps and smooth court-space trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.schemas import FORMAT_TRAJECTORIES

DEFAULT_MAX_GAP_SEC = 1.5
DEFAULT_MIN_POINTS = 8
DEFAULT_SMOOTH_WINDOW = 5


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def interpolate_points(points: list[dict], sample_fps: float, max_gap_sec: float) -> list[dict]:
    if len(points) < 2:
        return list(points)
    step = 1.0 / max(sample_fps, 0.1)
    ordered = sorted(
        [p for p in points if p.get("x") is not None and p.get("y") is not None],
        key=lambda p: p["t"],
    )
    if len(ordered) < 2:
        return ordered
    filled: list[dict] = []
    for prev, nxt in zip(ordered, ordered[1:]):
        filled.append({**prev, "interpolated": bool(prev.get("interpolated", False))})
        dt = float(nxt["t"]) - float(prev["t"])
        if dt <= step * 1.5 or dt > max_gap_sec:
            continue
        steps = int(round(dt / step))
        for i in range(1, steps):
            f = i / steps
            filled.append(
                {
                    "t": round(float(prev["t"]) + f * dt, 4),
                    "frame": int(round(prev.get("frame", 0) + f * (nxt.get("frame", 0) - prev.get("frame", 0)))),
                    "px": round(_lerp(float(prev["px"]), float(nxt["px"]), f), 5),
                    "py": round(_lerp(float(prev["py"]), float(nxt["py"]), f), 5),
                    "x": round(_lerp(float(prev["x"]), float(nxt["x"]), f), 3),
                    "y": round(_lerp(float(prev["y"]), float(nxt["y"]), f), 3),
                    "conf": None,
                    "missing": False,
                    "predicted": False,
                    "interpolated": True,
                }
            )
    filled.append({**ordered[-1], "interpolated": bool(ordered[-1].get("interpolated", False))})
    return filled


def smooth_xy(points: list[dict], window: int = DEFAULT_SMOOTH_WINDOW) -> list[dict]:
    n = len(points)
    if n < 3 or window < 3:
        return [{**p, "smoothed": False} for p in points]
    xs = np.array([float(p["x"]) for p in points], dtype=np.float64)
    ys = np.array([float(p["y"]) for p in points], dtype=np.float64)
    half = window // 2
    sx = xs.copy()
    sy = ys.copy()
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        sx[i] = xs[a:b].mean()
        sy[i] = ys[a:b].mean()
    out = []
    for i, p in enumerate(points):
        row = dict(p)
        row["x"] = round(float(sx[i]), 3)
        row["y"] = round(float(sy[i]), 3)
        row["smoothed"] = True
        out.append(row)
    return out


def _series_to_trajectory(track_id: int, label: str, points: list[dict]) -> dict:
    return {
        "player_id": track_id,
        "label": label,
        "points": [
            {
                "t": p["t"],
                "x": p["x"],
                "y": p["y"],
                "frame": p.get("frame"),
                "interpolated": bool(p.get("interpolated", False)),
                "smoothed": bool(p.get("smoothed", False)),
                "conf": p.get("conf"),
            }
            for p in points
        ],
    }


def run_trajectories(
    output_dir: str | Path,
    *,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    min_points: int = DEFAULT_MIN_POINTS,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> dict:
    """Write trajectories.json from court_tracks.json."""
    output_dir = Path(output_dir)
    src = output_dir / "court_tracks.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run --stage calibrate first.")
    data = json.loads(src.read_text(encoding="utf-8"))
    sample_fps = float(data.get("sampleFps") or 10.0)

    players_out: list[dict] = []
    dropped = 0
    for player in data.get("players") or []:
        if player.get("label") not in {None, "player"}:
            continue
        filled = interpolate_points(player.get("points") or [], sample_fps, max_gap_sec)
        if len(filled) < min_points:
            dropped += 1
            continue
        smoothed = smooth_xy(filled, window=smooth_window)
        players_out.append(
            _series_to_trajectory(int(player["track_id"]), "player", smoothed)
        )

    ball_filled = interpolate_points(
        [p for p in (data.get("ball") or {}).get("points") or [] if not p.get("missing")],
        sample_fps,
        max_gap_sec,
    )
    ball_traj = None
    if len(ball_filled) >= 3:
        ball_traj = _series_to_trajectory(0, "basketball", smooth_xy(ball_filled, window=max(3, smooth_window - 2)))

    artifact = {
        "format": FORMAT_TRAJECTORIES,
        "video": data.get("video"),
        "sampleFps": sample_fps,
        "frameCount": data.get("frameCount"),
        "court": data.get("court"),
        "players": players_out,
        "ball": ball_traj,
        "droppedShortTracks": dropped,
    }
    out_json = output_dir / "trajectories.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"Stage 4 trajectories: {len(players_out)} players"
        f"{'' if ball_traj is None else ' + ball'}, dropped {dropped} short tracks → {out_json}"
    )
    return artifact
