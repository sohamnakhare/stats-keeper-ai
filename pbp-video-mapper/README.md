# PBP Video Mapper

Map basketball play-by-play game clock to video timestamps by reading the broadcast scoreboard with OCR.

## Python Environment Setup

```bash
cd pbp-video-mapper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate when done:

```bash
deactivate
```

## Setup

```bash
cd pbp-video-mapper
pip install -r requirements.txt
```

## 1. Mark scoreboard regions

1. Open `scoreboard-marker.html` in a browser
2. Load a game video and seek to a clear scoreboard frame
3. Draw boxes for **Game Clock**, **Home**, **Away** (required); Shot Clock / Quarter optional
4. Download `scoreboard_roi.json`

## 2. Run OCR

```bash
python worker/scripts/detect_scoreboard.py \
  --video /path/to/game.mp4 \
  --roi scoreboard_roi.json \
  --output worker/output/scoreboard_track.json
```

Flags:

- `--fps 1.0` — sample rate (default 1 fps)
- `--no-gpu` — force CPU for EasyOCR
- `--debug` — include raw OCR text in output
- `--no-smooth` — skip outlier smoothing

## Output

`scoreboard_track.json` includes:

- `readings` — video time → score, `gameClock` (`M:SS` / `MM:SS`), shot clock, quarter
- `gameClockToVideo` — each unique canonical `gameClock` → first video timestamp
