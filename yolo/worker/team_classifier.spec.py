"""Unit tests for team classification (stub encoder; no Hugging Face download)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dino_encoder import l2_normalize
from player_config import TeamConfig
from player_tracker import PlayerObservation
from team_classifier import TeamClassifier, dominant_color_from_crop


def _obs(track_id: int, label: str = "player") -> PlayerObservation:
    return PlayerObservation(
        frame=0,
        t_sec=0.0,
        track_id=track_id,
        label=label,
        x1=0.1,
        y1=0.1,
        x2=0.9,
        y2=0.9,
        conf=0.9,
    )


def _frame(bgr: tuple[int, int, int], size: int = 80) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    return img


class StubEncoder:
    """Maps mean BGR to a 4-D embedding so red/blue crops separate cleanly."""

    def encode_bgr(self, crops: list[np.ndarray]) -> np.ndarray:
        rows = []
        for crop in crops:
            mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
            vec = np.array([mean[0], mean[1], mean[2], 1.0], dtype=np.float32)
            rows.append(l2_normalize(vec))
        return np.stack(rows)


def _config(**kwargs) -> TeamConfig:
    defaults = dict(
        method="dino",
        sample_every=1,
        min_samples_per_track=1,
        min_crop_height=10,
        max_crops_per_track=8,
        min_cosine=0.35,
    )
    defaults.update(kwargs)
    return TeamConfig(**defaults)


def _observe_color(classifier: TeamClassifier, track_id: int, bgr: tuple[int, int, int], n: int = 3) -> None:
    frame = _frame(bgr)
    obs = _obs(track_id)
    for _ in range(n):
        classifier.observe(frame, obs)


def test_dino_clusters_two_kits_into_a_and_b() -> None:
    clf = TeamClassifier(_config(), encoder=StubEncoder())
    _observe_color(clf, 1, (0, 0, 255))  # red
    _observe_color(clf, 2, (255, 0, 0))  # blue
    result = clf.finalize()
    assert result.track_team[1] in {"A", "B"}
    assert result.track_team[2] in {"A", "B"}
    assert result.track_team[1] != result.track_team[2]
    assert result.track_colors[1]
    assert result.track_colors[2]


def test_kit_book_assigns_named_team_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        red_path = tmp_path / "away.jpg"
        blue_path = tmp_path / "home.jpg"
        cv2.imwrite(str(red_path), _frame((0, 0, 255)))
        cv2.imwrite(str(blue_path), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "images": ["away.jpg"]},
                        {"id": "home", "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )

        clf = TeamClassifier(_config(kit_book_file=str(book)), encoder=StubEncoder())
        _observe_color(clf, 1, (0, 0, 255))
        _observe_color(clf, 2, (255, 0, 0))
        result = clf.finalize()
        assert result.track_team[1] == "away"
        assert result.track_team[2] == "home"


def test_kit_book_labels_outlier_by_prototype_not_cluster() -> None:
    """A single opposite-kit player must not inherit the majority cluster's team."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cv2.imwrite(str(tmp_path / "away.jpg"), _frame((0, 0, 255)))
        cv2.imwrite(str(tmp_path / "home.jpg"), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "color": [210, 45, 45], "images": ["away.jpg"]},
                        {"id": "home", "color": [40, 80, 210], "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )

        clf = TeamClassifier(_config(kit_book_file=str(book)), encoder=StubEncoder())
        _observe_color(clf, 1, (0, 0, 255))
        _observe_color(clf, 2, (0, 0, 255))
        _observe_color(clf, 3, (0, 0, 255))
        _observe_color(clf, 4, (255, 0, 0))
        result = clf.finalize()
        assert result.track_team[1] == "away"
        assert result.track_team[2] == "away"
        assert result.track_team[3] == "away"
        assert result.track_team[4] == "home"
        assert result.team_colors["away"] == (210, 45, 45)
        assert result.team_colors["home"] == (40, 80, 210)


def test_kit_book_uses_json_color_for_display() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        red_path = tmp_path / "away.jpg"
        blue_path = tmp_path / "home.jpg"
        cv2.imwrite(str(red_path), _frame((0, 0, 255)))
        cv2.imwrite(str(blue_path), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "color": [40, 80, 210], "images": ["away.jpg"]},
                        {"id": "home", "color": [210, 45, 45], "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )

        clf = TeamClassifier(_config(kit_book_file=str(book)), encoder=StubEncoder())
        _observe_color(clf, 1, (0, 0, 255))
        _observe_color(clf, 2, (255, 0, 0))
        result = clf.finalize()
        assert result.team_colors["away"] == (40, 80, 210)
        assert result.team_colors["home"] == (210, 45, 45)


class CollapsingEncoder:
    """Almost-constant embeddings so DINO cannot separate kits."""

    def encode_bgr(self, crops: list[np.ndarray]) -> np.ndarray:
        rows = []
        for crop in crops:
            mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
            vec = np.array([1.0, 0.02, 0.02, 0.02], dtype=np.float32)
            vec[1] += mean[0] * 1e-6
            rows.append(l2_normalize(vec))
        return np.stack(rows)


def test_kit_book_falls_back_to_color_when_dino_collapses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        red_path = tmp_path / "home.jpg"
        blue_path = tmp_path / "away.jpg"
        cv2.imwrite(str(red_path), _frame((0, 0, 255)))
        cv2.imwrite(str(blue_path), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "images": ["away.jpg"]},
                        {"id": "home", "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )

        clf = TeamClassifier(_config(kit_book_file=str(book)), encoder=CollapsingEncoder())
        _observe_color(clf, 1, (0, 0, 255))
        _observe_color(clf, 2, (255, 0, 0))
        result = clf.finalize()
        assert result.track_team[1] == "home"
        assert result.track_team[2] == "away"


def test_dominant_color_prefers_jersey_over_white_number() -> None:
    img = np.zeros((32, 24, 3), dtype=np.uint8)
    img[:] = (0, 0, 200)
    img[10:22, 8:16] = (255, 255, 255)
    color = dominant_color_from_crop(img, TeamConfig())
    assert color is not None
    _b, _g, r = (int(round(float(v))) for v in color)
    assert r > 140


def test_dominant_color_ignores_dark_floor() -> None:
    img = np.zeros((40, 30, 3), dtype=np.uint8)
    img[:] = (18, 22, 28)
    img[8:32, 6:24] = (0, 0, 200)
    color = dominant_color_from_crop(img, TeamConfig())
    assert color is not None
    b, g, r = (int(round(float(v))) for v in color)
    assert r > 140
    assert r > b and r > g


def test_crop_quality_prefers_saturated_jersey() -> None:
    from team_classifier import _crop_quality_score

    grey = np.full((20, 16, 3), 80, dtype=np.uint8)
    red = np.full((20, 16, 3), (0, 0, 200), dtype=np.uint8)
    assert _crop_quality_score(red, 80.0) > _crop_quality_score(grey, 80.0)


def test_kit_book_rejects_low_cosine() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        green_path = tmp_path / "home.jpg"
        cv2.imwrite(str(green_path), _frame((0, 255, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps({"teams": [{"id": "home", "images": ["home.jpg"]}]}),
            encoding="utf-8",
        )

        clf = TeamClassifier(
            _config(kit_book_file=str(book), min_cosine=0.35),
            encoder=StubEncoder(),
        )
        _observe_color(clf, 1, (0, 0, 255))  # red vs green prototype ~ orthogonal
        result = clf.finalize()
        assert result.track_team[1] is None


def test_color_method_still_splits_lab() -> None:
    clf = TeamClassifier(_config(method="color"), encoder=StubEncoder())
    _observe_color(clf, 1, (0, 0, 255))
    _observe_color(clf, 2, (255, 0, 0))
    result = clf.finalize()
    assert result.track_team[1] in {"A", "B"}
    assert result.track_team[2] in {"A", "B"}
    assert result.track_team[1] != result.track_team[2]


def test_color_path_kit_book_names_teams_without_encoder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cv2.imwrite(str(tmp_path / "away.jpg"), _frame((0, 0, 255)))
        cv2.imwrite(str(tmp_path / "home.jpg"), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "color": [210, 45, 45], "images": ["away.jpg"]},
                        {"id": "home", "color": [40, 80, 210], "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        clf = TeamClassifier(_config(method="color", kit_book_file=str(book)))
        _observe_color(clf, 1, (0, 0, 255))
        _observe_color(clf, 2, (255, 0, 0))
        result = clf.finalize()
        assert result.track_team[1] == "away"
        assert result.track_team[2] == "home"
        assert result.team_colors["away"] == (210, 45, 45)
        assert result.team_colors["home"] == (40, 80, 210)


def test_low_sat_track_marked_referee() -> None:
    clf = TeamClassifier(_config(method="color"))
    _observe_color(clf, 1, (0, 0, 255))
    _observe_color(clf, 2, (255, 0, 0))
    _observe_color(clf, 3, (40, 40, 40))
    result = clf.finalize()
    assert result.track_team[1] in {"A", "B"}
    assert result.track_team[2] in {"A", "B"}
    assert result.track_team[1] != result.track_team[2]
    assert result.track_team[3] == "referee"


def test_dominant_color_ignores_skin() -> None:
    img = np.zeros((40, 30, 3), dtype=np.uint8)
    img[:] = (90, 140, 200)
    img[8:32, 6:24] = (0, 0, 200)
    color = dominant_color_from_crop(img, TeamConfig())
    assert color is not None
    _b, _g, r = (int(round(float(v))) for v in color)
    assert r > 140


def test_close_kits_still_split_with_kit_book() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cv2.imwrite(str(tmp_path / "away.jpg"), _frame((40, 20, 20)))
        cv2.imwrite(str(tmp_path / "home.jpg"), _frame((20, 20, 40)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "images": ["away.jpg"]},
                        {"id": "home", "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        clf = TeamClassifier(_config(method="color", kit_book_file=str(book), close_centroid_lab=80.0))
        _observe_color(clf, 1, (40, 20, 20))
        _observe_color(clf, 2, (20, 20, 40))
        result = clf.finalize()
        assert result.track_team[1] == "away"
        assert result.track_team[2] == "home"


def test_torso_from_shoulders_hips() -> None:
    from player_pose import torso_xyxy_from_keypoints

    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[5] = [40, 20]
    kpts[6] = [80, 20]
    kpts[11] = [45, 80]
    kpts[12] = [75, 80]
    conf = np.ones(17, dtype=np.float32)
    box = torso_xyxy_from_keypoints(kpts, conf)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 < 40 and x2 > 80
    assert y1 < 20 and y2 > 80


def test_torso_missing_keypoints_falls_back() -> None:
    from player_pose import crop_torso_from_pose, PoseInstance
    from team_classifier import _torso_crop

    kpts = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    pose = PoseInstance(0.1, 0.1, 0.9, 0.9, kpts, conf)
    frame = _frame((0, 0, 200))
    obs = _obs(1)
    assert crop_torso_from_pose(frame, obs, pose, TeamConfig()) is None
    assert _torso_crop(frame, obs, TeamConfig()) is not None


def test_referee_is_not_assigned_a_kit() -> None:
    clf = TeamClassifier(_config(), encoder=StubEncoder())
    _observe_color(clf, 1, (0, 0, 255))
    _observe_color(clf, 2, (255, 0, 0))
    frame = _frame((128, 128, 128))
    clf.observe(frame, _obs(9, label="referee"))
    result = clf.finalize()
    assert result.track_team[9] == "referee"


def test_foot_xy_from_both_ankles() -> None:
    from player_pose import COCO_L_ANKLE, COCO_R_ANKLE, PoseInstance, foot_xy_from_pose

    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[COCO_L_ANKLE] = [32.0, 64.0]
    kpts[COCO_R_ANKLE] = [48.0, 64.0]
    conf = np.ones(17, dtype=np.float32)
    pose = PoseInstance(0.1, 0.1, 0.9, 0.9, kpts, conf)
    foot = foot_xy_from_pose(_obs(1), pose, (80, 80))
    assert foot is not None
    assert abs(foot[0] - 40.0 / 80.0) < 1e-6
    assert abs(foot[1] - 64.0 / 80.0) < 1e-6


def test_foot_xy_from_one_ankle() -> None:
    from player_pose import COCO_L_ANKLE, COCO_R_ANKLE, PoseInstance, foot_xy_from_pose

    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[COCO_L_ANKLE] = [40.0, 60.0]
    kpts[COCO_R_ANKLE] = [10.0, 10.0]
    conf = np.zeros(17, dtype=np.float32)
    conf[COCO_L_ANKLE] = 0.9
    pose = PoseInstance(0.1, 0.1, 0.9, 0.9, kpts, conf)
    assert foot_xy_from_pose(_obs(1), pose, (80, 80), min_conf=0.4) is None


def test_foot_xy_missing_ankles_returns_none() -> None:
    from player_pose import PoseInstance, foot_xy_from_pose

    kpts = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    pose = PoseInstance(0.1, 0.1, 0.9, 0.9, kpts, conf)
    assert foot_xy_from_pose(_obs(1), pose, (80, 80)) is None


def test_cluster_k_follows_kit_book_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cv2.imwrite(str(tmp_path / "away.jpg"), _frame((0, 0, 255)))
        cv2.imwrite(str(tmp_path / "home.jpg"), _frame((255, 0, 0)))
        book = tmp_path / "kit_book.json"
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "images": ["away.jpg"]},
                        {"id": "home", "images": ["home.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        clf = TeamClassifier(
            _config(method="color", kit_book_file=str(book), kmeans_clusters=3)
        )
        clf._load_kit_colors()
        assert clf._cluster_k(6) == 2

        cv2.imwrite(str(tmp_path / "alt.jpg"), _frame((0, 255, 0)))
        book.write_text(
            json.dumps(
                {
                    "teams": [
                        {"id": "away", "images": ["away.jpg"]},
                        {"id": "home", "images": ["home.jpg"]},
                        {"id": "alt", "images": ["alt.jpg"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        clf3 = TeamClassifier(
            _config(method="color", kit_book_file=str(book), kmeans_clusters=3)
        )
        clf3._load_kit_colors()
        assert clf3._cluster_k(6) == 3
