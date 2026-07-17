#!/usr/bin/env python3
"""Prepare Roboflow ball export for YOLOv8 detection training."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Literal

WORKER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = WORKER_ROOT / "models" / "Finetune E-BARD.v1i.yolov8"
DEFAULT_OUTPUT = WORKER_ROOT / "models" / "ball_dataset_prepared"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_detection_line(line: str) -> str | None:
    """Validate a YOLO detection label line: class cx cy w h (normalized)."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    class_id = parts[0]
    if class_id != "0":
        return None

    try:
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None

    if w <= 0.0 or h <= 0.0:
        return None
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _copy_label_file(src_label: Path, dest_label: Path) -> Literal["positive", "negative", "skip"]:
    """Copy ball detection labels; empty files are kept as hard negatives."""
    raw = src_label.read_text(encoding="utf-8").strip()
    if not raw:
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_label.write_text("", encoding="utf-8")
        return "negative"

    lines_out: list[str] = []
    for line in raw.splitlines():
        normalized = normalize_detection_line(line)
        if normalized is not None:
            lines_out.append(normalized)

    if not lines_out:
        return "skip"

    dest_label.parent.mkdir(parents=True, exist_ok=True)
    dest_label.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return "positive"


def _collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image: {image_path.name}")
        pairs.append((image_path, label_path))
    return pairs


def _split_pairs(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if len(pairs) < 2:
        return pairs, []

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    val_set = set(id(p) for p in shuffled[:val_count])

    train_pairs = [p for p in shuffled if id(p) not in val_set]
    val_pairs = [p for p in shuffled if id(p) in val_set]
    return train_pairs, val_pairs


def _copy_split(
    pairs: list[tuple[Path, Path]],
    images_out: Path,
    labels_out: Path,
) -> dict[str, int]:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    counts = {"positive": 0, "negative": 0, "skipped": 0}

    for image_path, label_path in pairs:
        dest_image = images_out / image_path.name
        dest_label = labels_out / f"{image_path.stem}.txt"
        shutil.copy2(image_path, dest_image)
        result = _copy_label_file(label_path, dest_label)
        counts[result] += 1
        if result == "skip":
            dest_image.unlink(missing_ok=True)
            dest_label.unlink(missing_ok=True)

    return counts


def write_data_yaml(dataset_root: Path, yaml_path: Path) -> None:
    content = (
        f"path: {dataset_root.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names: ['basketball']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def _prepare_from_train_only(
    source_dir: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> dict[str, int | str]:
    source_train_images = source_dir / "train" / "images"
    source_train_labels = source_dir / "train" / "labels"
    if not source_train_images.is_dir():
        raise FileNotFoundError(f"Expected train images at {source_train_images}")

    pairs = _collect_pairs(source_train_images, source_train_labels)
    if not pairs:
        raise ValueError(f"No image/label pairs found under {source_train_images}")

    train_pairs, val_pairs = _split_pairs(pairs, val_fraction=val_fraction, seed=seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    train_counts = _copy_split(
        train_pairs,
        output_dir / "train" / "images",
        output_dir / "train" / "labels",
    )
    val_counts = {"positive": 0, "negative": 0, "skipped": 0}
    if val_pairs:
        val_counts = _copy_split(
            val_pairs,
            output_dir / "valid" / "images",
            output_dir / "valid" / "labels",
        )

    yaml_path = output_dir / "data.local.yaml"
    write_data_yaml(output_dir, yaml_path)

    return {
        "source": str(source_dir.resolve()),
        "output": str(output_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "total_source": len(pairs),
        "train_positive": train_counts["positive"],
        "train_negative": train_counts["negative"],
        "val_positive": val_counts["positive"],
        "val_negative": val_counts["negative"],
        "skipped_invalid_labels": train_counts["skipped"] + val_counts["skipped"],
    }


def _has_split(source_dir: Path, split: str) -> bool:
    images = source_dir / split / "images"
    labels = source_dir / split / "labels"
    return images.is_dir() and labels.is_dir() and any(images.iterdir())


def _prepare_from_roboflow_splits(source_dir: Path, output_dir: Path) -> dict[str, int | str]:
    """Copy existing train/valid/test splits when Roboflow exported them."""
    split_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    totals = {"positive": 0, "negative": 0, "skipped": 0, "images": 0}
    copied_splits: list[str] = []

    for src_split, dest_split in split_map.items():
        if not _has_split(source_dir, src_split):
            continue
        if dest_split in copied_splits:
            continue

        pairs = _collect_pairs(
            source_dir / src_split / "images",
            source_dir / src_split / "labels",
        )
        counts = _copy_split(
            pairs,
            output_dir / dest_split / "images",
            output_dir / dest_split / "labels",
        )
        copied_splits.append(dest_split)
        totals["positive"] += counts["positive"]
        totals["negative"] += counts["negative"]
        totals["skipped"] += counts["skipped"]
        totals["images"] += counts["positive"] + counts["negative"]

    if "train" not in copied_splits:
        raise FileNotFoundError(f"No train split found under {source_dir}")

    yaml_path = output_dir / "data.local.yaml"
    write_data_yaml(output_dir, yaml_path)

    return {
        "source": str(source_dir.resolve()),
        "output": str(output_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "total_source": totals["images"],
        "train_positive": totals["positive"],
        "train_negative": totals["negative"],
        "val_positive": 0,
        "val_negative": 0,
        "skipped_invalid_labels": totals["skipped"],
        "splits": ", ".join(copied_splits),
    }


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, int | str]:
    if _has_split(source_dir, "valid") or _has_split(source_dir, "val"):
        return _prepare_from_roboflow_splits(source_dir, output_dir)
    return _prepare_from_train_only(source_dir, output_dir, val_fraction, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ball dataset for YOLOv8 training")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Roboflow YOLOv8 export directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Prepared dataset output directory",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source dataset not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    try:
        summary = prepare_dataset(
            source_dir=args.source,
            output_dir=args.output,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Prepared dataset at {summary['output']}")
    print(f"data.yaml: {summary['data_yaml']}")
    print(f"Source images: {summary['total_source']}")
    if "splits" in summary:
        print(f"Roboflow splits: {summary['splits']}")
    else:
        print(
            f"Train: {summary['train_positive']} with ball, "
            f"{summary['train_negative']} negatives"
        )
        print(
            f"Val: {summary['val_positive']} with ball, "
            f"{summary['val_negative']} negatives"
        )
    if summary["skipped_invalid_labels"]:
        print(f"Skipped (invalid labels): {summary['skipped_invalid_labels']}")


if __name__ == "__main__":
    main()
