"""Tests for prepare_court_dataset pose label handling."""

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_court_dataset import (  # noqa: E402
    _copy_label_file,
    normalize_pose_line,
    write_data_yaml,
)

SAMPLE_POSE = (
    "0 0.56431390625 0.6451012962962963 0.43307578125 0.1704799074074074 "
    "0.5817856770833333 0.5743991666666667 2 "
    "0.76567078125 0.6091353703703705 1 "
    "0.5440167187499999 0.7029249074074074 2 "
    "0.36003999999999997 0.6389872222222222 2"
)


def test_normalize_pose_line_accepts_four_keypoints() -> None:
    result = normalize_pose_line(SAMPLE_POSE)
    assert result is not None
    parts = result.split()
    assert len(parts) == 17
    assert parts[0] == "0"
    assert parts[7] == "2"
    assert parts[10] == "1"


def test_normalize_pose_line_rejects_bbox_only() -> None:
    assert normalize_pose_line("0 0.5 0.5 0.4 0.2") is None


def test_normalize_pose_line_rejects_non_court_class() -> None:
    line = SAMPLE_POSE.replace("0 ", "1 ", 1)
    assert normalize_pose_line(line) is None


def test_normalize_pose_line_rejects_bad_visibility() -> None:
    parts = SAMPLE_POSE.split()
    parts[7] = "3"
    assert normalize_pose_line(" ".join(parts)) is None


def test_copy_label_file_keeps_empty_negative(tmp_path) -> None:
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")
    dest = tmp_path / "out.txt"
    assert _copy_label_file(src, dest) == "negative"
    assert dest.read_text(encoding="utf-8") == ""


def test_copy_label_file_copies_pose(tmp_path) -> None:
    src = tmp_path / "court.txt"
    src.write_text(SAMPLE_POSE + "\n", encoding="utf-8")
    dest = tmp_path / "out.txt"
    assert _copy_label_file(src, dest) == "positive"
    assert dest.read_text(encoding="utf-8").startswith("0 0.564314 0.645101")


def test_write_data_yaml(tmp_path) -> None:
    write_data_yaml(tmp_path, tmp_path / "data.local.yaml")
    text = (tmp_path / "data.local.yaml").read_text(encoding="utf-8")
    assert "names: ['court']" in text
    assert "kpt_shape: [4, 3]" in text
    assert "flip_idx: [1, 0, 3, 2]" in text
    assert f"path: {tmp_path.resolve()}" in text
