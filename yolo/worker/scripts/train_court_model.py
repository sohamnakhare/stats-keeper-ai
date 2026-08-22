#!/usr/bin/env python3
"""Fine-tune YOLOv8-pose on prepared court keypoint dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = WORKER_ROOT / "models" / "court_dataset_prepared" / "data.local.yaml"
DEFAULT_PRETRAINED = Path("yolov8n-pose.pt")
DEFAULT_OUTPUT = WORKER_ROOT / "models" / "court_pose_v1.pt"
DEFAULT_PROJECT = WORKER_ROOT / "models" / "runs"
DEFAULT_NAME = "court_pose"


def _default_device() -> str:
    import sys as _sys

    return "mps" if _sys.platform == "darwin" else "cuda:0"


def _pretrained_arg(pretrained: Path) -> str:
    if pretrained.exists():
        return str(pretrained.resolve())
    return str(pretrained)


def train_court_model(
    data_yaml: Path,
    pretrained: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    device: str,
    output_weights: Path,
) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(_pretrained_arg(pretrained))
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=str(project.resolve()),
        name=name,
        exist_ok=True,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"Training finished but best.pt missing at {best_src}")

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, output_weights)
    return output_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-pose court detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data.local.yaml path")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
        help="Pose checkpoint (Ultralytics downloads yolov8n-pose.pt if missing)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Stable output weights path")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()

    try:
        output = train_court_model(
            data_yaml=args.data,
            pretrained=args.pretrained,
            project=args.project,
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            output_weights=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Training complete. Best weights: {output}")


if __name__ == "__main__":
    main()
