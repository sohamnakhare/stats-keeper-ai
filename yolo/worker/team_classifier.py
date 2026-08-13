"""Team classification from jersey color (torso crop + HSV K-means)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from player_config import TeamConfig
from player_tracker import PlayerObservation


@dataclass
class TeamAssignment:
    """Final result: per-track team labels and representative team colors."""

    track_team: dict[int, str | None]  # track_id -> "A" | "B" | "referee" | None
    team_colors: dict[str, tuple[int, int, int]]  # team -> RGB
    track_colors: dict[int, tuple[int, int, int]]  # track_id -> median jersey RGB


@dataclass
class _TrackColorState:
    label: str = "player"
    samples: list[np.ndarray] = field(default_factory=list)  # dominant BGR per observation
    seen: int = 0


def _torso_crop(
    frame_bgr: np.ndarray, obs: PlayerObservation, config: TeamConfig
) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    bx1 = obs.x1 * width
    bx2 = obs.x2 * width
    by1 = obs.y1 * height
    by2 = obs.y2 * height
    bw = bx2 - bx1
    bh = by2 - by1

    x1 = int(bx1 + bw * config.torso_x_inset)
    x2 = int(bx2 - bw * config.torso_x_inset)
    y1 = int(by1 + bh * config.torso_top)
    y2 = int(by1 + bh * config.torso_bottom)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame_bgr[y1:y2, x1:x2]


def dominant_torso_color(
    frame_bgr: np.ndarray, obs: PlayerObservation, config: TeamConfig
) -> np.ndarray | None:
    """Dominant torso color (BGR float array) via K-means in HSV space.

    The crop border approximates the background (court floor); the largest
    cluster clearly different from it is taken as the jersey color, so the
    court surface doesn't win just by pixel count.
    """
    crop = _torso_crop(frame_bgr, obs, config)
    if crop is None:
        return None

    # Downsample for speed; K-means on ~1k pixels is plenty.
    crop = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA)

    border = np.concatenate(
        [crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0
    ).astype(np.float32)
    background_bgr = np.median(border, axis=0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    k = max(2, config.kmeans_clusters)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        hsv, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)

    centers_bgr = cv2.cvtColor(
        centers.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2BGR
    ).reshape(-1, 3).astype(np.float64)

    background_dist = np.linalg.norm(centers_bgr - background_bgr, axis=1)
    non_background = background_dist > 40.0
    order = np.argsort(counts)[::-1]
    for idx in order:
        if non_background[idx]:
            return centers_bgr[idx]
    return centers_bgr[order[0]]


class TeamClassifier:
    """Accumulates per-track jersey colors, then clusters tracks into 2 teams."""

    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self._tracks: dict[int, _TrackColorState] = {}

    def observe(self, frame_bgr: np.ndarray, obs: PlayerObservation) -> None:
        if not self.config.enabled:
            return
        state = self._tracks.setdefault(obs.track_id, _TrackColorState(label=obs.label))
        state.label = obs.label
        state.seen += 1
        if (state.seen - 1) % max(1, self.config.sample_every) != 0:
            return

        height, _ = frame_bgr.shape[:2]
        if obs.bbox_height * height < self.config.min_crop_height:
            return

        color = dominant_torso_color(frame_bgr, obs, self.config)
        if color is not None:
            state.samples.append(color)

    @staticmethod
    def _bgr_to_rgb_tuple(bgr: np.ndarray) -> tuple[int, int, int]:
        b, g, r = (int(round(float(v))) for v in bgr)
        return (r, g, b)

    def finalize(self) -> TeamAssignment:
        track_team: dict[int, str | None] = {}
        track_colors: dict[int, tuple[int, int, int]] = {}

        eligible: list[int] = []
        for track_id, state in self._tracks.items():
            if state.label == "referee":
                track_team[track_id] = "referee"
            else:
                track_team[track_id] = None
            if state.samples:
                median = np.median(np.stack(state.samples), axis=0)
                track_colors[track_id] = self._bgr_to_rgb_tuple(median)
            if (
                state.label == "player"
                and len(state.samples) >= self.config.min_samples_per_track
            ):
                eligible.append(track_id)

        if not self.config.enabled or len(eligible) < 2:
            return TeamAssignment(track_team=track_team, team_colors={}, track_colors=track_colors)

        # Cluster per-track median colors into 2 teams in Lab space
        # (perceptually more uniform than BGR for color distance).
        medians_bgr = np.stack(
            [np.median(np.stack(self._tracks[tid].samples), axis=0) for tid in eligible]
        )
        lab = cv2.cvtColor(
            medians_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, cluster_labels, centroids = cv2.kmeans(
            lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        cluster_labels = cluster_labels.flatten()

        # Temporal smoothing: re-assign each track by majority vote of its
        # individual samples' nearest team centroid.
        for i, track_id in enumerate(eligible):
            samples_bgr = np.stack(self._tracks[track_id].samples)
            samples_lab = cv2.cvtColor(
                samples_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).reshape(-1, 3).astype(np.float32)
            d0 = np.linalg.norm(samples_lab - centroids[0], axis=1)
            d1 = np.linalg.norm(samples_lab - centroids[1], axis=1)
            votes_1 = int(np.sum(d1 < d0))
            majority = 1 if votes_1 * 2 > len(samples_lab) else 0
            cluster_labels[i] = majority
            track_team[track_id] = "A" if majority == 0 else "B"

        team_colors: dict[str, tuple[int, int, int]] = {}
        for team, cluster in (("A", 0), ("B", 1)):
            member_ids = [tid for i, tid in enumerate(eligible) if cluster_labels[i] == cluster]
            if member_ids:
                members = np.stack(
                    [np.median(np.stack(self._tracks[tid].samples), axis=0) for tid in member_ids]
                )
                team_colors[team] = self._bgr_to_rgb_tuple(np.median(members, axis=0))

        return TeamAssignment(
            track_team=track_team, team_colors=team_colors, track_colors=track_colors
        )
