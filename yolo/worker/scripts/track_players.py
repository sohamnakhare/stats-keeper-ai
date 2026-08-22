#!/usr/bin/env python3
"""Track player movement, team colors and jersey numbers -> player_movement.json.

Pipeline: YOLO player detection -> Deep OC-SORT -> homography to court meters
-> team color classification -> selective jersey OCR -> track merging.

Every stage is configurable via --config (JSON mirroring PipelineConfig);
individual CLI flags override values from the file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from court_projection import CourtProjector, PoseCourtProjector  # noqa: E402
from detector import resolve_inference_device  # noqa: E402
from jersey_ocr import JerseyOcrCollector  # noqa: E402
from player_config import PipelineConfig, load_pipeline_config  # noqa: E402
from player_pose import PlayerPoseEstimator, foot_xy_from_pose, match_pose  # noqa: E402
from player_tracker import PlayerTracker  # noqa: E402
from team_classifier import TeamClassifier  # noqa: E402


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """CLI flags (when provided) take precedence over the config file."""
    det = config.detection
    if args.model is not None:
        det.model_path = str(args.model)
    if args.device is not None:
        det.device = args.device
    if args.conf is not None:
        det.conf = args.conf
    if args.sample_fps is not None:
        det.sample_fps = args.sample_fps
    if args.max_width is not None:
        det.max_width = args.max_width
    if args.court_polygon is not None:
        det.court_polygon_file = str(args.court_polygon)
    if args.no_referees:
        det.detect_referees = False
    if args.referees:
        det.detect_referees = True

    if args.calibration is not None:
        config.homography.calibration_file = str(args.calibration)
    if getattr(args, "court_detect", None) is not None:
        config.homography.court_detect_file = str(args.court_detect)

    if args.min_track_points is not None:
        config.tracking.min_track_points = args.min_track_points
    if args.court_margin is not None:
        config.homography.court_margin = args.court_margin

    if args.no_team:
        config.team.enabled = False
    if getattr(args, "team_method", None):
        config.team.method = args.team_method
    if getattr(args, "kit_book", None) is not None:
        config.team.kit_book_file = str(args.kit_book)
    if getattr(args, "no_pose", False):
        config.team.use_pose = False
    if args.no_ocr:
        config.jersey_ocr.enabled = False
    if args.ocr_gpu:
        config.jersey_ocr.gpu = True
    if args.no_merge:
        config.merge.enabled = False
    if getattr(args, "no_fill_gaps", False):
        config.merge.fill_gaps = False
    if getattr(args, "max_gap_sec", None) is not None:
        config.merge.max_gap_sec = args.max_gap_sec
        config.merge.max_fill_gap_sec = max(config.merge.max_fill_gap_sec, args.max_gap_sec)
    if getattr(args, "track_buffer", None) is not None:
        config.tracking.track_buffer = args.track_buffer
    if getattr(args, "no_reid", False):
        config.tracking.with_reid = False

    return config


def filter_kept_tracks(
    track_points: dict[int, list[dict]],
    track_labels: dict[int, str],
    config: PipelineConfig,
    has_projector: bool,
) -> dict[int, list[dict]]:
    """Keep player tracks only; require enough in-court samples when calibrated."""
    min_points = config.tracking.min_track_points
    min_frac = config.detection.min_on_court_fraction
    kept: dict[int, list[dict]] = {}

    for track_id, points in track_points.items():
        if track_labels.get(track_id, "player") != "player":
            continue

        if has_projector:
            on_court = [p for p in points if p["x"] is not None]
            if len(on_court) < min_points:
                continue
            if len(on_court) / max(len(points), 1) < min_frac:
                continue
            kept[track_id] = on_court
        else:
            if len(points) < min_points:
                continue
            kept[track_id] = points

    return kept


def merge_tracks(players: list[dict], config: PipelineConfig) -> list[dict]:
    """Reconnect broken tracks so a player who leaves sight and returns stays one player.

    Strong signal: matching jersey numbers (same team). Without numbers,
    fragments merge on team + jersey color similarity + a plausible
    time/court-distance gap and realistic implied speed.
    """
    merge_cfg = config.merge
    if not merge_cfg.enabled:
        return players

    def color_distance(c1, c2) -> float | None:
        if not c1 or not c2 or len(c1) != 3 or len(c2) != 3:
            return None
        return math.dist(c1, c2)

    def merge_score(a: dict, b: dict) -> float | None:
        """Lower is better. None = cannot merge."""
        gap = b["points"][0]["t"] - a["points"][-1]["t"]
        if gap <= 0 or gap > merge_cfg.max_gap_sec:
            return None
        pa, pb = a["points"][-1], b["points"][0]
        dist = 0.0
        if pa.get("x") is not None and pb.get("x") is not None:
            dist = math.hypot(pb["x"] - pa["x"], pb["y"] - pa["y"])
            if dist > merge_cfg.max_court_distance:
                return None
            if gap > 0.05 and dist / gap > merge_cfg.max_speed_mps:
                return None

        ja, jb = a.get("jerseyNumber"), b.get("jerseyNumber")
        if ja is not None and jb is not None:
            if ja != jb or a.get("team") != b.get("team"):
                return None
            # Prefer short gaps and short travel when jersey matches.
            return gap + dist * 0.1

        if not merge_cfg.allow_no_jersey:
            return None
        team_a, team_b = a.get("team"), b.get("team")
        if team_a is not None and team_b is not None and team_a != team_b:
            return None
        # Conflicting jersey vs null is ok; both null needs color agreement when available.
        cdist = color_distance(a.get("jerseyColor"), b.get("jerseyColor"))
        if cdist is not None and cdist > merge_cfg.max_color_distance:
            return None
        # Heavier penalty without jersey match so numbered merges win first.
        color_pen = 0.0 if cdist is None else cdist / 100.0
        return 10.0 + gap + dist * 0.15 + color_pen

    players = sorted(players, key=lambda p: p["points"][0]["t"])
    merged: list[dict] = []
    for player in players:
        best: tuple[float, dict] | None = None
        for candidate in merged:
            score = merge_score(candidate, player)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is None:
            merged.append(player)
            continue
        target = best[1]
        target["points"].extend(player["points"])
        target["points"].sort(key=lambda p: p["t"])
        target["mergedTrackIds"].append(player["trackId"])
        target["mergedTrackIds"].extend(player["mergedTrackIds"])
        if target["jerseyNumber"] is None and player["jerseyNumber"] is not None:
            target["jerseyNumber"] = player["jerseyNumber"]
            target["jerseyConfidence"] = player["jerseyConfidence"]
        else:
            target["jerseyConfidence"] = max(
                target["jerseyConfidence"] or 0.0, player["jerseyConfidence"] or 0.0
            )
        if target.get("jerseyColor") is None and player.get("jerseyColor") is not None:
            target["jerseyColor"] = player["jerseyColor"]
        if target.get("team") is None and player.get("team") is not None:
            target["team"] = player["team"]
    return merged


def fill_track_gaps(
    players: list[dict],
    config: PipelineConfig,
    projector: CourtProjector | PoseCourtProjector | None = None,
) -> int:
    """Insert interpolated points across detection gaps so players stay in play.

    Covers both occlusions within a track and joins created by merging.
    Interpolated points carry `"interpolated": true` so the renderer can draw
    them as ghosts. Court meters are re-projected at each timestamp when a
    projector is available (lerping meters slides under a moving camera).
    Returns the number of points inserted.
    """
    merge_cfg = config.merge
    if not merge_cfg.fill_gaps:
        return 0

    step = 1.0 / max(config.detection.sample_fps, 0.1)
    inserted = 0

    for player in players:
        points = sorted(player["points"], key=lambda p: p["t"])
        filled: list[dict] = []
        for prev, nxt in zip(points, points[1:]):
            filled.append(prev)
            dt = nxt["t"] - prev["t"]
            if dt <= step * 1.5 or dt > merge_cfg.max_fill_gap_sec:
                continue
            steps = int(round(dt / step))
            for i in range(1, steps):
                f = i / steps
                t = round(prev["t"] + f * dt, 3)
                px = round(prev["px"] + f * (nxt["px"] - prev["px"]), 4)
                py = round(prev["py"] + f * (nxt["py"] - prev["py"]), 4)
                court_xy = None
                if projector is not None:
                    court_xy = projector.project(px, py, t_sec=t)
                elif prev["x"] is not None and nxt["x"] is not None:
                    court_xy = (
                        prev["x"] + f * (nxt["x"] - prev["x"]),
                        prev["y"] + f * (nxt["y"] - prev["y"]),
                    )
                filled.append(
                    {
                        "t": t,
                        "frame": int(round(prev["frame"] + f * (nxt["frame"] - prev["frame"]))),
                        "px": px,
                        "py": py,
                        "x": round(court_xy[0], 2) if court_xy else None,
                        "y": round(court_xy[1], 2) if court_xy else None,
                        "conf": None,
                        "interpolated": True,
                    }
                )
                inserted += 1
        if points:
            filled.append(points[-1])
        player["points"] = filled

    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track player movement, teams and jersey numbers from a game clip"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKER_ROOT / "output" / "player_movement.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pipeline config JSON (sections: detection, tracking, homography, team, jerseyOcr, merge)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="court_calibration.json from court-marker.html (static; ignores zoom)",
    )
    parser.add_argument(
        "--court-detect",
        type=Path,
        default=None,
        help="court_detect.json from detect_court.py (per-frame D keypoints, zoom-aware)",
    )
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights (E-BARD)")
    parser.add_argument("--device", default=None, help="Inference device (mps / cuda:0)")
    parser.add_argument("--conf", type=float, default=None, help="Min detection confidence")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument(
        "--court-polygon",
        type=Path,
        default=None,
        help="hoop_roi.json with courtPolygon (filters detections by foot position)",
    )
    parser.add_argument("--min-track-points", type=int, default=None)
    parser.add_argument(
        "--court-margin",
        type=float,
        default=None,
        help="Meters outside the court to still accept a foot projection",
    )
    parser.add_argument(
        "--no-referees",
        action="store_true",
        help="Skip referee class (default: already off)",
    )
    parser.add_argument(
        "--referees",
        action="store_true",
        help="Also detect the referee class",
    )
    parser.add_argument("--no-team", action="store_true", help="Skip team classification")
    parser.add_argument(
        "--team-method",
        choices=["dino", "color"],
        default=None,
        help="Team ID: jersey-color K-means (default) or DINOv2 torso embeddings",
    )
    parser.add_argument(
        "--kit-book",
        type=Path,
        default=None,
        help="kit_book.json with per-team example jersey crops (named team ids)",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip YOLO-pose torso crop; use the proportional chest rectangle",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Skip jersey number OCR")
    parser.add_argument("--ocr-gpu", action="store_true", help="Run PaddleOCR on GPU")
    parser.add_argument("--no-merge", action="store_true", help="Skip track merging")
    parser.add_argument(
        "--no-fill-gaps",
        action="store_true",
        help="Do not interpolate positions while a player is out of sight",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=None,
        help="Max seconds between track fragments to merge / fill (default 6)",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=None,
        help="Frames to keep a lost ID alive (default 90)",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable Deep OC-SORT appearance ReID (motion + GMC only)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    config = apply_cli_overrides(load_pipeline_config(args.config), args)

    if not Path(config.detection.model_path).exists():
        print(f"Model not found: {config.detection.model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        device = resolve_inference_device(config.detection.device)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print(f"Inference device: {device}", flush=True)

    projector: CourtProjector | PoseCourtProjector | None = None
    detect_file = config.homography.court_detect_file
    calib_file = config.homography.calibration_file
    if detect_file:
        projector = PoseCourtProjector.from_detect_file(detect_file, config.homography)
        print(
            f"Court detect (zoom-aware): {detect_file} "
            f"({projector.court.length} x {projector.court.width} {projector.court.unit}"
            f", preset={projector.court.preset})"
        )
        if calib_file:
            print(f"Ignoring static calibration {calib_file} (court-detect takes precedence)")
    elif calib_file:
        projector = CourtProjector.from_calibration_file(calib_file, config.homography)
        print(
            f"Court calibration: {calib_file} "
            f"({projector.court.length} x {projector.court.width} {projector.court.unit})"
        )
    else:
        print("No calibration file: output will contain pixel coordinates only")

    tracker = PlayerTracker(config.detection, config.tracking)
    team_classifier = TeamClassifier(config.team, device=device)
    ocr_collector = JerseyOcrCollector(config.jersey_ocr)
    pose_estimator: PlayerPoseEstimator | None = None
    if config.team.enabled and config.team.use_pose:
        try:
            pose_estimator = PlayerPoseEstimator(config.team, device=device)
            print(f"Player pose: {config.team.pose_model}", flush=True)
        except Exception as exc:
            print(f"Player pose unavailable ({exc}); using rectangle torso crop.", flush=True)
            pose_estimator = None

    track_points: dict[int, list[dict]] = defaultdict(list)
    track_labels: dict[int, str] = {}
    foot_ema: dict[int, tuple[float, float]] = {}
    foot_alpha = config.detection.foot_ema_alpha
    court_xy_ema: dict[int, tuple[float, float]] = {}
    court_alpha = config.homography.court_xy_ema_alpha
    frame_count = 0

    print("Tracking players...", flush=True)
    for sampled, observations in tracker.iter_observations(str(args.video)):
        frame_count = sampled.index + 1
        poses = pose_estimator.estimate(sampled.bgr) if pose_estimator else []
        frame_h, frame_w = sampled.bgr.shape[:2]
        for obs in observations:
            if obs.label != "player":
                continue
            track_labels[obs.track_id] = obs.label
            pose = match_pose(obs, poses) if poses else None
            foot = (
                foot_xy_from_pose(
                    obs, pose, (frame_w, frame_h), min_conf=config.team.pose_min_conf
                )
                if pose is not None
                else None
            )
            raw_x, raw_y = foot if foot is not None else (obs.foot_x, obs.foot_y)
            prev = foot_ema.get(obs.track_id)
            if prev is None or foot_alpha >= 1.0:
                foot_x, foot_y = raw_x, raw_y
            elif foot_alpha <= 0.0:
                foot_x, foot_y = prev
            else:
                foot_x = foot_alpha * raw_x + (1.0 - foot_alpha) * prev[0]
                foot_y = foot_alpha * raw_y + (1.0 - foot_alpha) * prev[1]
            foot_ema[obs.track_id] = (foot_x, foot_y)
            court_xy = (
                projector.project(foot_x, foot_y, t_sec=obs.t_sec)
                if projector
                else None
            )
            if court_xy is not None:
                prev_court = court_xy_ema.get(obs.track_id)
                if prev_court is None or court_alpha >= 1.0:
                    smoothed = court_xy
                elif court_alpha <= 0.0:
                    smoothed = prev_court
                else:
                    smoothed = (
                        court_alpha * court_xy[0] + (1.0 - court_alpha) * prev_court[0],
                        court_alpha * court_xy[1] + (1.0 - court_alpha) * prev_court[1],
                    )
                court_xy_ema[obs.track_id] = smoothed
                court_xy = smoothed
            # When calibrated, ignore detections that don't land on the court
            # (cameramen / crowd / bench along the sideline).
            if projector is not None and court_xy is None:
                continue
            track_points[obs.track_id].append(
                {
                    "t": round(obs.t_sec, 3),
                    "frame": obs.frame,
                    "px": round(foot_x, 4),
                    "py": round(foot_y, 4),
                    "x": round(court_xy[0], 2) if court_xy else None,
                    "y": round(court_xy[1], 2) if court_xy else None,
                    "conf": round(obs.conf, 3),
                }
            )
            team_classifier.observe(sampled.bgr, obs, pose=pose)
            ocr_collector.observe(sampled.bgr, obs)
        if sampled.index % 50 == 0:
            print(
                f"  frame {sampled.index} t={sampled.t_sec:.1f}s "
                f"tracks={len(track_points)}",
                flush=True,
            )

    # Drop short / off-court / non-player tracks (cameramen, refs, bench).
    kept = filter_kept_tracks(
        track_points, track_labels, config, has_projector=projector is not None
    )
    print(f"Tracks: {len(track_points)} raw, {len(kept)} kept (players on court)")

    print(f"Classifying teams ({config.team.method})...", flush=True)
    team_assignment = team_classifier.finalize()

    print("Running jersey OCR...", flush=True)
    jersey_results = ocr_collector.finalize()
    for track_id, result in sorted(jersey_results.items()):
        print(f"  track {track_id} -> #{result.number} (share={result.confidence}, votes={result.votes})")

    players: list[dict] = []
    for track_id, points in kept.items():
        jersey = jersey_results.get(track_id)
        players.append(
            {
                "trackId": track_id,
                "label": track_labels.get(track_id, "player"),
                "team": team_assignment.track_team.get(track_id),
                "jerseyNumber": jersey.number if jersey else None,
                "jerseyConfidence": jersey.confidence if jersey else None,
                "jerseyColor": list(team_assignment.track_colors.get(track_id) or []) or None,
                "mergedTrackIds": [],
                "points": points,
            }
        )

    before = len(players)
    players = merge_tracks(players, config)
    if config.merge.enabled and before != len(players):
        print(f"Merged {before - len(players)} broken track(s)")

    inserted = fill_track_gaps(players, config, projector=projector)
    if inserted:
        print(f"Gap fill: inserted {inserted} interpolated point(s)")

    artifact = {
        "video": args.video.name,
        "sampleFps": config.detection.sample_fps,
        "frameCount": frame_count,
        "court": (
            {
                "layout": projector.court.layout,
                "length": projector.court.length,
                "width": projector.court.width,
                "unit": projector.court.unit,
                "preset": projector.court.preset,
            }
            if projector
            else None
        ),
        "teams": [
            {"id": team, "color": list(color)}
            for team, color in sorted(team_assignment.team_colors.items())
        ],
        "config": config.model_dump(by_alias=True),
        "players": players,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")

    team_counts = Counter(player["team"] or "?" for player in players)
    print("Team split: " + ", ".join(f"{name}={n}" for name, n in sorted(team_counts.items())))
    if artifact["teams"]:
        print(
            "Team colors: "
            + ", ".join(f"{t['id']} rgb{tuple(t['color'])}" for t in artifact["teams"])
        )

    for player in players:
        team = player["team"] or "?"
        number = f"#{player['jerseyNumber']}" if player["jerseyNumber"] else "#?"
        merged_info = (
            f" (+{len(player['mergedTrackIds'])} merged)" if player["mergedTrackIds"] else ""
        )
        print(
            f"  track {player['trackId']} [{player['label']}] team={team} {number} "
            f"{len(player['points'])} points{merged_info}"
        )


if __name__ == "__main__":
    main()
