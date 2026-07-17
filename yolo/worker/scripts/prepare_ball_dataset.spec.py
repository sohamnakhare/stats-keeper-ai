"""Tests for prepare_ball_dataset label handling."""

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_ball_dataset import (  # noqa: E402
    _copy_label_file,
    normalize_detection_line,
    write_data_yaml,
)


def test_normalize_detection_line_accepts_bbox() -> None:
    line = "0 0.822265625 0.5 0.0205078125 0.03125"
    result = normalize_detection_line(line)
    assert result == "0 0.822266 0.500000 0.020508 0.031250"


def test_normalize_detection_line_rejects_polygon() -> None:
    line = "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8"
    assert normalize_detection_line(line) is None


def test_normalize_detection_line_rejects_non_ball_class() -> None:
    assert normalize_detection_line("1 0.5 0.5 0.1 0.1") is None


def test_copy_label_file_keeps_empty_negative(tmp_path) -> None:
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")
    dest = tmp_path / "out.txt"
    assert _copy_label_file(src, dest) == "negative"
    assert dest.read_text(encoding="utf-8") == ""


def test_copy_label_file_copies_bbox(tmp_path) -> None:
    src = tmp_path / "ball.txt"
    src.write_text("0 0.5 0.5 0.02 0.03\n", encoding="utf-8")
    dest = tmp_path / "out.txt"
    assert _copy_label_file(src, dest) == "positive"
    assert dest.read_text(encoding="utf-8").startswith("0 0.500000 0.500000")


def test_write_data_yaml(tmp_path) -> None:
    write_data_yaml(tmp_path, tmp_path / "data.local.yaml")
    text = (tmp_path / "data.local.yaml").read_text(encoding="utf-8")
    assert "names: ['basketball']" in text
    assert f"path: {tmp_path.resolve()}" in text
