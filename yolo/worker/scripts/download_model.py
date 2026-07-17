#!/usr/bin/env python3
"""Download E-BARD YOLOv8n weights from HuggingFace."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "GabrieleGiudici/E-BARD-detection-models"
FILENAME = "BODD_yolov8n_0001.pt"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"


def download_model(dest_dir: Path = DEFAULT_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(dest_dir),
    )
    path = Path(downloaded)
    print(f"Model saved to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download E-BARD basketball detection model")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory to save model weights",
    )
    args = parser.parse_args()
    download_model(args.dest)


if __name__ == "__main__":
    main()
