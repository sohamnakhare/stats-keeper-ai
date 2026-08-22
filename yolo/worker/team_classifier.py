"""Team classification from torso crops (DINOv2 embeddings or jersey color)."""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dino_encoder import DinoEncoder, EmbeddingEncoder, l2_normalize
from player_config import TeamConfig
from player_pose import PoseInstance, crop_torso_from_pose
from player_tracker import PlayerObservation


@dataclass
class TeamAssignment:
    """Final result: per-track team labels and representative team colors."""

    track_team: dict[int, str | None]  # track_id -> team id | "referee" | None
    team_colors: dict[str, tuple[int, int, int]]  # team -> RGB
    track_colors: dict[int, tuple[int, int, int]]  # track_id -> median jersey RGB


@dataclass
class KitBook:
    """Named kit prototypes plus RGB for rendering and color fallback."""

    prototypes: dict[str, np.ndarray]
    colors: dict[str, tuple[int, int, int]]  # display RGB (JSON or from images)
    sample_colors: dict[str, tuple[int, int, int]]  # RGB from kit photos (Lab match)


@dataclass(order=True)
class _ScoredCrop:
    score: float
    seq: int
    crop: np.ndarray = field(compare=False)


@dataclass
class _TrackColorState:
    label: str = "player"
    samples: list[np.ndarray] = field(default_factory=list)  # dominant BGR per observation
    crops: list[_ScoredCrop] = field(default_factory=list)  # best torso crops for DINO
    sat_samples: list[float] = field(default_factory=list)
    v_std_samples: list[float] = field(default_factory=list)
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


def _chromatic_mask(
    hsv_img: np.ndarray,
    *,
    min_sat: int = 40,
    min_val: int = 40,
    include_dark: bool = False,
) -> np.ndarray:
    """True for jersey-like pixels; drops skin, white numbers, and floor shadow."""
    h = hsv_img[:, :, 0].reshape(-1)
    s = hsv_img[:, :, 1].reshape(-1)
    v = hsv_img[:, :, 2].reshape(-1)
    # OpenCV H is degrees/2. Skin ~10-25° → H 5-13; keep a slightly wider band.
    skin = (h >= 3) & (h <= 18) & (s >= 40) & (s <= 180) & (v >= 80)
    blown = (v > 245) & (s < 40)
    bright = (~skin) & (~blown) & (s >= min_sat) & (v >= min_val)
    if not include_dark:
        return bright
    dark_chroma = (~skin) & (~blown) & (s >= 20) & (v >= 15) & (v < min_val)
    return bright | dark_chroma


def _chromatic_pixels_bgr(
    bgr: np.ndarray,
    *,
    min_sat: int = 40,
    min_val: int = 40,
) -> np.ndarray:
    """BGR pixels that look like dyed fabric, not floor shadow or white numbers."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _chromatic_mask(hsv, min_sat=min_sat, min_val=min_val)
    return bgr.reshape(-1, 3)[mask]


def kit_image_color(bgr: np.ndarray) -> tuple[int, int, int]:
    """Median RGB of chromatic pixels so white numbers do not wash out the kit."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _chromatic_mask(hsv)
    use = bgr.reshape(-1, 3)[mask]
    if len(use) < 8:
        use = bgr.reshape(-1, 3)[_chromatic_mask(hsv, include_dark=True)]
    if len(use) < 8:
        use = bgr.reshape(-1, 3)
    b, g, r = (int(round(float(v))) for v in np.median(use.astype(np.float32), axis=0))
    return (r, g, b)


def dominant_color_from_crop(crop: np.ndarray, config: TeamConfig) -> np.ndarray | None:
    """Jersey-fabric color (BGR float array) from a torso crop.

    HSV-filter (drop skin / floor / numbers), then k-means k=2 on remaining
    pixels. Navy-safe dark-chroma mask if the bright-sat mask is empty.
    """
    if crop is None or crop.size == 0:
        return None

    small = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA)
    hsv_img = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hsv = hsv_img.reshape(-1, 3).astype(np.float32)
    bgr = small.reshape(-1, 3).astype(np.float32)

    mask = _chromatic_mask(hsv_img)
    if int(np.count_nonzero(mask)) < 12:
        mask = _chromatic_mask(hsv_img, include_dark=True)
    filtered = hsv[mask]
    filtered_bgr = bgr[mask]
    if len(filtered) < 8:
        return None

    if len(filtered) < 16:
        return np.median(filtered_bgr, axis=0)

    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(filtered, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    centers_bgr = cv2.cvtColor(
        centers.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_HSV2BGR
    ).reshape(-1, 3).astype(np.float64)

    best_idx = 0
    best_score = -1.0
    for idx in range(k):
        sat = float(centers[idx, 1])
        val = float(centers[idx, 2])
        if sat < 12 and val > 200:
            continue
        score = sat * (0.35 + np.sqrt(float(counts[idx])))
        if val < 80:
            score *= 1.15
        if score > best_score:
            best_score = score
            best_idx = idx
    return centers_bgr[best_idx]


def _crop_quality_score(crop: np.ndarray, bbox_h_px: float) -> float:
    """Prefer large boxes whose torso crop still has saturated fabric."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    return float(bbox_h_px) * (0.35 + sat / 255.0)


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
    return dominant_color_from_crop(crop, config)


def _keep_best_crop(heap: list[_ScoredCrop], item: _ScoredCrop, max_keep: int) -> None:
    if len(heap) < max_keep:
        heapq.heappush(heap, item)
    elif item.score > heap[0].score:
        heapq.heapreplace(heap, item)


def _parse_rgb(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        rgb = tuple(int(round(float(c))) for c in value)
    except (TypeError, ValueError):
        return None
    if any(c < 0 or c > 255 for c in rgb):
        return None
    return rgb  # type: ignore[return-value]


def load_kit_book(path: Path | str, encoder: EmbeddingEncoder | None = None) -> KitBook:
    """RGB (and optional embeddings) per team from kit_book.json images."""
    book_path = Path(path)
    data = json.loads(book_path.read_text(encoding="utf-8"))
    prototypes: dict[str, np.ndarray] = {}
    colors: dict[str, tuple[int, int, int]] = {}
    sample_colors: dict[str, tuple[int, int, int]] = {}
    for team in data.get("teams") or []:
        team_id = str(team.get("id") or "").strip()
        images = team.get("images") or []
        if not team_id or not images:
            continue
        crops: list[np.ndarray] = []
        for rel in images:
            image_path = (book_path.parent / rel).resolve()
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                print(f"Kit book image not found: {image_path}", file=sys.stderr)
                continue
            crops.append(bgr)
        if not crops:
            continue
        if encoder is not None:
            stacked = encoder.encode_bgr(crops)
            prototypes[team_id] = l2_normalize(stacked.mean(axis=0))
        per_image = [kit_image_color(crop) for crop in crops]
        extracted = tuple(int(round(float(v))) for v in np.median(np.array(per_image), axis=0))
        extracted_rgb = (extracted[0], extracted[1], extracted[2])
        display = _parse_rgb(team.get("color"))
        colors[team_id] = display or extracted_rgb
        sample_colors[team_id] = extracted_rgb
    return KitBook(prototypes=prototypes, colors=colors, sample_colors=sample_colors)


class TeamClassifier:
    """Accumulates per-track torso crops, then assigns teams via DINO or color."""

    def __init__(
        self,
        config: TeamConfig,
        device: str | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._encoder = encoder
        self._tracks: dict[int, _TrackColorState] = {}
        self._crop_seq = 0
        self._kit_book: KitBook | None = None

    def observe(
        self,
        frame_bgr: np.ndarray,
        obs: PlayerObservation,
        pose: PoseInstance | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        state = self._tracks.setdefault(obs.track_id, _TrackColorState(label=obs.label))
        state.label = obs.label
        state.seen += 1
        if (state.seen - 1) % max(1, self.config.sample_every) != 0:
            return

        height, _ = frame_bgr.shape[:2]
        bbox_h_px = obs.bbox_height * height
        if bbox_h_px < self.config.min_crop_height:
            return

        crop = None
        if pose is not None:
            crop = crop_torso_from_pose(frame_bgr, obs, pose, self.config)
        if crop is None:
            crop = _torso_crop(frame_bgr, obs, self.config)
        if crop is None:
            return

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        state.sat_samples.append(float(np.mean(hsv[:, :, 1])))
        state.v_std_samples.append(float(np.std(hsv[:, :, 2])))

        color = dominant_color_from_crop(crop, self.config)
        if color is not None:
            state.samples.append(color)

        if self.config.method == "dino":
            self._crop_seq += 1
            _keep_best_crop(
                state.crops,
                _ScoredCrop(
                    score=_crop_quality_score(crop, bbox_h_px),
                    seq=self._crop_seq,
                    crop=crop.copy(),
                ),
                self.config.max_crops_per_track,
            )

    @staticmethod
    def _bgr_to_rgb_tuple(bgr: np.ndarray) -> tuple[int, int, int]:
        b, g, r = (int(round(float(v))) for v in bgr)
        return (r, g, b)

    def _colors_from_samples(self) -> tuple[dict[int, str | None], dict[int, tuple[int, int, int]], list[int]]:
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
        return track_team, track_colors, eligible

    def _team_colors_from_members(
        self, track_team: dict[int, str | None]
    ) -> dict[str, tuple[int, int, int]]:
        grouped: dict[str, list[int]] = {}
        for track_id, team in track_team.items():
            if team in (None, "referee"):
                continue
            grouped.setdefault(team, []).append(track_id)
        team_colors: dict[str, tuple[int, int, int]] = {}
        for team, member_ids in grouped.items():
            if self._kit_book and team in self._kit_book.colors:
                team_colors[team] = self._kit_book.colors[team]
                continue
            stacks = [
                np.median(np.stack(self._tracks[tid].samples), axis=0)
                for tid in member_ids
                if self._tracks[tid].samples
            ]
            if stacks:
                team_colors[team] = self._bgr_to_rgb_tuple(np.median(np.stack(stacks), axis=0))
        return team_colors

    def _load_kit_colors(self) -> None:
        if self._kit_book is not None or not self.config.kit_book_file:
            return
        book = load_kit_book(self.config.kit_book_file, encoder=None)
        if book.sample_colors:
            self._kit_book = book

    @staticmethod
    def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
        arr = np.asarray(bgr, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr.reshape(1, 1, 3)
        elif arr.ndim == 2:
            arr = arr.reshape(-1, 1, 3)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    def _kit_lab(self, sample_colors: dict[str, tuple[int, int, int]]) -> tuple[list[str], np.ndarray]:
        names = list(sample_colors.keys())
        kit_bgr = np.array(
            [[sample_colors[n][2], sample_colors[n][1], sample_colors[n][0]] for n in names],
            dtype=np.uint8,
        )
        return names, self._bgr_to_lab(kit_bgr)

    def _exclude_appearance_referees(
        self, track_team: dict[int, str | None], eligible: list[int]
    ) -> list[int]:
        kit_lab = None
        if self._kit_book and self._kit_book.sample_colors:
            _names, kit_lab = self._kit_lab(self._kit_book.sample_colors)
        for track_id, state in self._tracks.items():
            if state.label != "player" or track_team.get(track_id) == "referee":
                continue
            if not state.sat_samples:
                continue
            sat = float(np.median(state.sat_samples))
            v_std = float(np.median(state.v_std_samples)) if state.v_std_samples else 0.0
            near_kit = False
            if kit_lab is not None and state.samples:
                median = np.median(np.stack(state.samples), axis=0)
                lab = self._bgr_to_lab(median)[0]
                dists = np.linalg.norm(kit_lab - lab, axis=1)
                near_kit = float(np.min(dists)) < self.config.close_centroid_lab
            striped = v_std >= 40.0 and sat < 50.0
            low_sat = sat < self.config.referee_max_sat
            if (low_sat or striped) and not near_kit:
                track_team[track_id] = "referee"
        return [tid for tid in eligible if track_team.get(tid) != "referee"]

    def _map_lab_clusters_to_kits(
        self, centroids_lab: np.ndarray, sample_colors: dict[str, tuple[int, int, int]]
    ) -> dict[int, str]:
        names, kit_lab = self._kit_lab(sample_colors)
        k, n = int(centroids_lab.shape[0]), len(names)
        dists = np.linalg.norm(centroids_lab[:, None, :] - kit_lab[None, :, :], axis=2)
        if k == 2 and n == 2:
            if float(dists[0, 0] + dists[1, 1]) <= float(dists[0, 1] + dists[1, 0]):
                return {0: names[0], 1: names[1]}
            return {0: names[1], 1: names[0]}
        mapping: dict[int, str] = {}
        used: set[int] = set()
        pairs = sorted(
            ((float(dists[c, t]), c, t) for c in range(k) for t in range(n)),
        )
        for _, cluster, team_idx in pairs:
            if cluster in mapping or team_idx in used:
                continue
            mapping[cluster] = names[team_idx]
            used.add(team_idx)
        for cluster in range(k):
            mapping.setdefault(cluster, names[int(np.argmin(dists[cluster]))])
        return mapping

    def _assign_nearest_kit(
        self,
        track_team: dict[int, str | None],
        eligible: list[int],
        sample_colors: dict[str, tuple[int, int, int]],
    ) -> None:
        names, kit_lab = self._kit_lab(sample_colors)
        margin = self.config.kit_lab_margin
        for track_id in eligible:
            if not self._tracks[track_id].samples:
                continue
            median = np.median(np.stack(self._tracks[track_id].samples), axis=0)
            lab = self._bgr_to_lab(median)[0]
            dists = np.linalg.norm(kit_lab - lab, axis=1)
            order = np.argsort(dists)
            best = int(order[0])
            second = float(dists[order[1]]) if len(order) > 1 else float("inf")
            if second - float(dists[best]) < margin:
                track_team[track_id] = None
            else:
                track_team[track_id] = names[best]

    def _cluster_k(self, n_eligible: int) -> int:
        """k-means k: capped by config, kit-book size, and eligible tracks."""
        n_kits = 0
        if self._kit_book and self._kit_book.sample_colors:
            n_kits = len(self._kit_book.sample_colors)
        n_kits = n_kits or 2
        k = min(self.config.kmeans_clusters, n_eligible, n_kits)
        if n_eligible >= 2:
            return max(2, k)
        return 1

    def _cluster_lab_or_value(self, eligible: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (track_cluster array aligned to eligible, centroids in Lab)."""
        medians_bgr = np.stack(
            [np.median(np.stack(self._tracks[tid].samples), axis=0) for tid in eligible]
        )
        lab = self._bgr_to_lab(medians_bgr)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        k = self._cluster_k(len(eligible))
        if k < 2:
            return np.zeros(len(eligible), dtype=np.int32), lab
        _, labels, centroids = cv2.kmeans(
            lab, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centroids = centroids.astype(np.float32)
        labels = labels.flatten().astype(np.int32)
        if k == 2 and float(np.linalg.norm(centroids[0] - centroids[1])) < self.config.close_centroid_lab:
            hsv = cv2.cvtColor(
                medians_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
            ).reshape(-1, 3).astype(np.float32)
            values = hsv[:, 2:3]
            _, labels, v_centers = cv2.kmeans(values, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            order = np.argsort(v_centers.flatten())
            remap = {int(order[0]): 0, int(order[1]): 1}
            labels = np.array([remap[int(v)] for v in labels.flatten()], dtype=np.int32)
            centroids = np.stack(
                [
                    lab[labels == 0].mean(axis=0) if np.any(labels == 0) else centroids[0],
                    lab[labels == 1].mean(axis=0) if np.any(labels == 1) else centroids[1],
                ]
            ).astype(np.float32)
            return labels, centroids
        return labels, centroids

    def _finalize_color(
        self,
        track_team: dict[int, str | None],
        track_colors: dict[int, tuple[int, int, int]],
        eligible: list[int],
    ) -> TeamAssignment:
        if not self.config.enabled:
            return TeamAssignment(
                track_team=track_team, team_colors={}, track_colors=track_colors
            )
        self._load_kit_colors()
        eligible = self._exclude_appearance_referees(track_team, eligible)
        sample_colors = self._kit_book.sample_colors if self._kit_book else {}

        if len(eligible) < 2:
            if sample_colors:
                self._assign_nearest_kit(track_team, eligible, sample_colors)
            return TeamAssignment(
                track_team=track_team,
                team_colors=self._team_colors_from_members(track_team),
                track_colors=track_colors,
            )

        _, centroids = self._cluster_lab_or_value(eligible)
        k = int(centroids.shape[0])
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        vote_floor = self.config.min_vote_share
        cluster_of: dict[int, int] = {}
        for track_id in eligible:
            samples_bgr = np.stack(self._tracks[track_id].samples)
            samples_lab = self._bgr_to_lab(samples_bgr)
            dists = np.linalg.norm(samples_lab[:, None, :] - centroids[None, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)
            counts = np.bincount(nearest, minlength=k)
            cluster = int(np.argmax(counts))
            share = float(counts[cluster]) / max(len(samples_lab), 1)
            if share < vote_floor:
                track_team[track_id] = None
                continue
            cluster_of[track_id] = cluster
            track_team[track_id] = letters[cluster] if cluster < len(letters) else str(cluster)

        named: dict[int, str] | None = None
        if len(sample_colors) >= 2:
            named = self._map_lab_clusters_to_kits(centroids, sample_colors)
            for track_id, cluster in cluster_of.items():
                if track_team.get(track_id) in (None, "referee"):
                    continue
                track_team[track_id] = named[cluster]

        counts = self._count_teams(track_team, eligible)
        if sample_colors and len(counts) < 2:
            self._assign_nearest_kit(track_team, eligible, sample_colors)

        return TeamAssignment(
            track_team=track_team,
            team_colors=self._team_colors_from_members(track_team),
            track_colors=track_colors,
        )

    def _get_encoder(self) -> EmbeddingEncoder:
        if self._encoder is not None:
            return self._encoder
        if not self.device:
            raise RuntimeError("DINOv2 team classification requires an inference device.")
        self._encoder = DinoEncoder(self.config.dino_model, self.device)
        return self._encoder

    def _encode_track_crops(
        self, encoder: EmbeddingEncoder, track_ids: list[int]
    ) -> dict[int, np.ndarray]:
        """Encode buffered crops; return per-track (N, D) embeddings."""
        all_crops: list[np.ndarray] = []
        spans: list[tuple[int, int, int]] = []  # track_id, start, count
        for track_id in track_ids:
            crops = [item.crop for item in self._tracks[track_id].crops]
            if not crops:
                continue
            start = len(all_crops)
            all_crops.extend(crops)
            spans.append((track_id, start, len(crops)))
        if not all_crops:
            return {}
        stacked = encoder.encode_bgr(all_crops)
        per_track: dict[int, np.ndarray] = {}
        for track_id, start, count in spans:
            per_track[track_id] = stacked[start : start + count]
        return per_track

    def _embed_cluster(
        self, usable: list[int], per_track: dict[int, np.ndarray], k: int
    ) -> tuple[dict[int, int], np.ndarray] | None:
        """Majority-vote each track onto k embedding centroids."""
        if len(usable) < 2 or k < 2:
            return None
        k = min(k, len(usable))
        means = l2_normalize(np.stack([per_track[tid].mean(axis=0) for tid in usable]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, _, centroids = cv2.kmeans(
            means.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centroids = l2_normalize(centroids.astype(np.float32))
        assigned: dict[int, int] = {}
        for track_id in usable:
            dists = 1.0 - per_track[track_id] @ centroids.T
            nearest = np.argmin(dists, axis=1)
            assigned[track_id] = int(np.bincount(nearest, minlength=k).argmax())
        return assigned, centroids

    @staticmethod
    def _map_clusters_to_names(
        centroids: np.ndarray, proto: np.ndarray, names: list[str]
    ) -> dict[int, str]:
        """Assign cluster indices to kit names by maximum cosine matching."""
        k, n = int(centroids.shape[0]), len(names)
        sims = centroids @ proto.T
        if k == 2 and n == 2:
            if float(sims[0, 0] + sims[1, 1]) >= float(sims[0, 1] + sims[1, 0]):
                return {0: names[0], 1: names[1]}
            return {0: names[1], 1: names[0]}
        mapping: dict[int, str] = {}
        used: set[int] = set()
        pairs = sorted(
            ((float(sims[c, t]), c, t) for c in range(k) for t in range(n)),
            reverse=True,
        )
        for _, cluster, team_idx in pairs:
            if cluster in mapping or team_idx in used:
                continue
            mapping[cluster] = names[team_idx]
            used.add(team_idx)
            if len(mapping) == k:
                break
        for cluster in range(k):
            mapping.setdefault(cluster, names[int(np.argmax(sims[cluster]))])
        return mapping

    def _assign_by_kit_color(
        self,
        track_team: dict[int, str | None],
        eligible: list[int],
        sample_colors: dict[str, tuple[int, int, int]],
    ) -> None:
        names = list(sample_colors.keys())
        kit_bgr = np.array(
            [[sample_colors[n][2], sample_colors[n][1], sample_colors[n][0]] for n in names],
            dtype=np.uint8,
        )
        kit_lab = cv2.cvtColor(kit_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(
            np.float32
        )
        for track_id in eligible:
            if not self._tracks[track_id].samples:
                continue
            median = np.median(np.stack(self._tracks[track_id].samples), axis=0).astype(np.uint8)
            lab = (
                cv2.cvtColor(median.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
                .reshape(3)
                .astype(np.float32)
            )
            dists = np.linalg.norm(kit_lab - lab, axis=1)
            track_team[track_id] = names[int(np.argmin(dists))]

    @staticmethod
    def _count_teams(track_team: dict[int, str | None], eligible: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for track_id in eligible:
            team = track_team.get(track_id)
            if team in (None, "referee"):
                continue
            counts[team] = counts.get(team, 0) + 1
        return counts

    def _assign_by_prototype(
        self,
        track_team: dict[int, str | None],
        usable: list[int],
        per_track: dict[int, np.ndarray],
        proto: np.ndarray,
        names: list[str],
    ) -> None:
        """Label each track by crop majority vote against kit prototypes."""
        min_cos = self.config.min_cosine
        for track_id in usable:
            samples = per_track.get(track_id)
            if samples is None or len(samples) == 0:
                continue
            sims = samples @ proto.T
            best_idx = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(len(samples)), best_idx]
            votes: dict[str, int] = {}
            for idx, sim in zip(best_idx, best_sim):
                if float(sim) < min_cos:
                    continue
                name = names[int(idx)]
                votes[name] = votes.get(name, 0) + 1
            if not votes:
                continue
            track_team[track_id] = max(votes, key=votes.get)

    def _assign_kit_book(
        self,
        encoder: EmbeddingEncoder,
        track_team: dict[int, str | None],
        eligible: list[int],
        per_track: dict[int, np.ndarray],
    ) -> bool:
        """Label tracks from the kit book. Returns False if the book could not be used."""
        if not self.config.kit_book_file:
            return False
        book = load_kit_book(self.config.kit_book_file, encoder)
        if len(book.prototypes) < 1:
            print("Kit book had no usable team images; clustering instead.", file=sys.stderr)
            return False
        self._kit_book = book
        names = list(book.prototypes.keys())
        proto = np.stack([book.prototypes[name] for name in names])
        usable = [tid for tid in eligible if tid in per_track and len(per_track[tid]) > 0]

        self._assign_by_prototype(track_team, usable, per_track, proto, names)
        counts = self._count_teams(track_team, eligible)
        if len(counts) >= 2:
            print(f"Kit book: nearest prototype {counts}", flush=True)
            return True

        clustered = False
        if len(names) >= 2 and len(usable) >= 2:
            result = self._embed_cluster(usable, per_track, k=len(names))
            if result is not None:
                assigned, centroids = result
                centroid_sim = (
                    float(centroids[0] @ centroids[1]) if len(centroids) >= 2 else 1.0
                )
                if centroid_sim < 0.9:
                    mapping = self._map_clusters_to_names(centroids, proto, names)
                    for track_id, cluster in assigned.items():
                        track_team[track_id] = mapping[cluster]
                    clustered = True

        counts = self._count_teams(track_team, eligible)
        if clustered and len(counts) >= 2:
            print(f"Kit book: clustered tracks {counts}", flush=True)
            return True

        if len(names) >= 2 and book.sample_colors:
            self._assign_by_kit_color(track_team, eligible, book.sample_colors)
            counts = self._count_teams(track_team, eligible)
            if len(counts) >= 2:
                print(f"Kit book: DINO kits too similar; assigned by jersey color {counts}", flush=True)
                return True

        print(f"Kit book: nearest prototype {self._count_teams(track_team, eligible)}", flush=True)
        return True

    def _cluster_embeddings(
        self,
        track_team: dict[int, str | None],
        eligible: list[int],
        per_track: dict[int, np.ndarray],
    ) -> None:
        usable = [tid for tid in eligible if tid in per_track and len(per_track[tid]) > 0]
        result = self._embed_cluster(usable, per_track, k=2)
        if result is None:
            return
        assigned, _centroids = result
        for track_id, cluster in assigned.items():
            track_team[track_id] = "A" if cluster == 0 else "B"

    def _finalize_dino(
        self,
        track_team: dict[int, str | None],
        track_colors: dict[int, tuple[int, int, int]],
        eligible: list[int],
    ) -> TeamAssignment:
        dino_eligible = [
            tid
            for tid in eligible
            if len(self._tracks[tid].crops) >= self.config.min_samples_per_track
        ]
        try:
            encoder = self._get_encoder()
            per_track = self._encode_track_crops(encoder, dino_eligible)
            used_book = self._assign_kit_book(encoder, track_team, dino_eligible, per_track)
            if not used_book:
                self._cluster_embeddings(track_team, dino_eligible, per_track)
        except Exception as exc:
            print(
                f"DINOv2 team model failed ({exc}); falling back to jersey color.",
                file=sys.stderr,
                flush=True,
            )
            return self._finalize_color(track_team, track_colors, eligible)

        return TeamAssignment(
            track_team=track_team,
            team_colors=self._team_colors_from_members(track_team),
            track_colors=track_colors,
        )

    def finalize(self) -> TeamAssignment:
        track_team, track_colors, eligible = self._colors_from_samples()
        if not self.config.enabled:
            return TeamAssignment(
                track_team=track_team, team_colors={}, track_colors=track_colors
            )
        if self.config.method == "dino":
            return self._finalize_dino(track_team, track_colors, eligible)
        return self._finalize_color(track_team, track_colors, eligible)
