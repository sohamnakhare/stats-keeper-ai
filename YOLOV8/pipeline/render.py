"""Stage 6: top-down court diagram (PNG / SVG) with player movement arrows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from pipeline.schemas import METERS_TO_FEET

COURT_COLOR = "#d9783a"
LINE_COLOR = "#f4f0ea"
PAINT_COLOR = "#c45c28"
BALL_COLOR = "#111111"

# FIBA markings in feet (used when preset looks FIBA / metric).
FIBA = {
    "hoop_inset": 1.575 * METERS_TO_FEET,
    "paint_width": 4.9 * METERS_TO_FEET,
    "paint_length": 5.8 * METERS_TO_FEET,
    "ft_radius": 1.8 * METERS_TO_FEET,
    "three_radius": 6.75 * METERS_TO_FEET,
    "restricted": 1.25 * METERS_TO_FEET,
    "backboard_width": 1.80 * METERS_TO_FEET,
    "corner_three_from_sideline": 0.90 * METERS_TO_FEET,
}

NBA = {
    "hoop_inset": 5.25,
    "paint_width": 16.0,
    "paint_length": 19.0,
    "ft_radius": 6.0,
    "three_radius": 23.75,
    "restricted": 4.0,
    "backboard_width": 6.0,
    "corner_three_from_sideline": 3.0,
}

PLAYER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]


def _markings(court: dict) -> dict:
    preset = str(court.get("preset") or "").lower()
    if "nba" in preset:
        return NBA
    return FIBA


def _draw_hoop_end(ax, court: dict, marks: dict, baseline: str) -> None:
    """Draw paint, FT circle, 3PT, hoop. baseline='near' (y=0) or 'far' (y=width)."""
    length = float(court["length"])
    width = float(court["width"])
    cx = length / 2.0
    hoop_y = marks["hoop_inset"] if baseline == "near" else width - marks["hoop_inset"]
    sign = 1.0 if baseline == "near" else -1.0
    paint_y0 = 0.0 if baseline == "near" else width - marks["paint_length"]
    paint_x = cx - marks["paint_width"] / 2.0

    ax.add_patch(
        Rectangle(
            (paint_x, paint_y0),
            marks["paint_width"],
            marks["paint_length"],
            linewidth=1.6,
            edgecolor=LINE_COLOR,
            facecolor=PAINT_COLOR,
            zorder=1,
        )
    )
    ft_y = marks["paint_length"] if baseline == "near" else width - marks["paint_length"]
    ax.add_patch(
        Arc(
            (cx, ft_y),
            marks["ft_radius"] * 2,
            marks["ft_radius"] * 2,
            theta1=0 if baseline == "near" else 180,
            theta2=180 if baseline == "near" else 360,
            linewidth=1.6,
            color=LINE_COLOR,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle((cx, hoop_y), marks["restricted"], fill=False, linewidth=1.2, color=LINE_COLOR, zorder=2)
    )
    ax.add_patch(Circle((cx, hoop_y), 0.75, fill=False, linewidth=2.0, color=LINE_COLOR, zorder=3))
    bb_half = marks["backboard_width"] / 2.0
    bb_y = hoop_y - sign * 0.5
    ax.plot([cx - bb_half, cx + bb_half], [bb_y, bb_y], color=LINE_COLOR, lw=2.4, zorder=3)

    inset = marks["corner_three_from_sideline"]
    angles = np.linspace(-1.15, 1.15, 80)
    xs = cx + marks["three_radius"] * np.sin(angles)
    ys = hoop_y + sign * marks["three_radius"] * np.cos(angles)
    mask = (xs >= inset) & (xs <= length - inset) & (ys >= 0) & (ys <= width)
    if mask.any():
        ax.plot(xs[mask], ys[mask], color=LINE_COLOR, lw=1.6, zorder=2)
    left_x, right_x = inset, length - inset
    base_y = 0.0 if baseline == "near" else width
    ax.plot([left_x, left_x], [base_y, hoop_y], color=LINE_COLOR, lw=1.6, zorder=2)
    ax.plot([right_x, right_x], [base_y, hoop_y], color=LINE_COLOR, lw=1.6, zorder=2)


def draw_court(ax, court: dict) -> None:
    length = float(court["length"])
    width = float(court["width"])
    marks = _markings(court)
    ax.add_patch(
        Rectangle((0, 0), length, width, linewidth=2.2, edgecolor=LINE_COLOR, facecolor=COURT_COLOR, zorder=0)
    )
    _draw_hoop_end(ax, court, marks, "near")
    if court.get("layout") == "full":
        _draw_hoop_end(ax, court, marks, "far")
        ax.plot([0, length], [width / 2, width / 2], color=LINE_COLOR, lw=1.6, zorder=2)
        ax.add_patch(
            Circle((length / 2, width / 2), marks["ft_radius"], fill=False, linewidth=1.6, color=LINE_COLOR, zorder=2)
        )
    ax.set_xlim(-1.0, length + 1.0)
    ax.set_ylim(-1.0, width + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def _path_xy(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([p["x"] for p in points if p.get("x") is not None and p.get("y") is not None], dtype=float)
    ys = np.array([p["y"] for p in points if p.get("x") is not None and p.get("y") is not None], dtype=float)
    return xs, ys


def _draw_player_path(ax, xs: np.ndarray, ys: np.ndarray, color: str, label: str) -> None:
    if len(xs) < 2:
        if len(xs) == 1:
            ax.scatter(xs[0], ys[0], s=36, color=color, zorder=5, label=label)
        return
    ax.plot(xs, ys, color=color, lw=1.8, solid_capstyle="round", zorder=4, label=label)
    ax.scatter(xs[0], ys[0], s=42, color=color, zorder=6, edgecolors="white", linewidths=0.6)
    ax.annotate(
        label,
        (xs[0], ys[0]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=7,
        color="white",
        zorder=7,
    )
    # Arrow along the last segment that has some length.
    for i in range(len(xs) - 1, 0, -1):
        dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        if dx * dx + dy * dy > 0.05:
            arrow = FancyArrowPatch(
                (xs[i - 1], ys[i - 1]),
                (xs[i], ys[i]),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.8,
                color=color,
                zorder=5,
            )
            ax.add_patch(arrow)
            break


def run_render(
    output_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Render trajectories.json onto a top-down court. Writes PNG and SVG."""
    output_dir = Path(output_dir)
    traj_path = output_dir / "trajectories.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}. Run --stage trajectories first.")
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    court = data.get("court") or {"layout": "half", "length": 47.0, "width": 50.0, "unit": "ft"}

    fig, ax = plt.subplots(figsize=(8.5, 8.5 if court.get("layout") == "full" else 7.2))
    fig.patch.set_facecolor("#1b1b1b")
    draw_court(ax, court)

    for i, player in enumerate(data.get("players") or []):
        xs, ys = _path_xy(player.get("points") or [])
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        _draw_player_path(ax, xs, ys, color, str(player.get("player_id")))

    ball = data.get("ball")
    if ball:
        xs, ys = _path_xy(ball.get("points") or [])
        if len(xs) >= 2:
            ax.plot(xs, ys, color=BALL_COLOR, lw=1.1, alpha=0.85, zorder=5, label="ball")
            ax.scatter(xs[-1], ys[-1], s=28, color=BALL_COLOR, zorder=6)

    ax.set_title("Play diagram (movement paths)", color="white", fontsize=11, pad=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix(".svg")
    if output_path.suffix.lower() != ".svg":
        fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Stage 6 render: {output_path}")
    if output_path.suffix.lower() != ".svg":
        print(f"  svg: {svg_path}")
    return output_path
