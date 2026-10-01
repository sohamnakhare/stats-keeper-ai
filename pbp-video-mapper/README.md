# PBP Video Mapper

Map basketball play-by-play game clock to video timestamps by reading the broadcast scoreboard with OCR.

## Python Environment Setup

`requirements.txt` installs `paddlepaddle-gpu==3.3.0` for CUDA 12.6. That wheel is Linux x86_64 only and needs an NVIDIA driver >= 550.54.14, so `pip install -r requirements.txt` does not work on macOS.

On Linux:

```bash
cd pbp-video-mapper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On a Mac, install the same packages with the CPU build of Paddle:

```bash
cd pbp-video-mapper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python-headless "numpy>=1.26.0" "pydantic>=2.9.0" "paddleocr>=3.0.0,<4" "yt-dlp>=2024.8.6"
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Deactivate when done:

```bash
deactivate
```

## 1. Mark scoreboard regions

1. Open `scoreboard-marker.html` in a browser
2. Load a game video and seek to a clear scoreboard frame
3. Draw a box around the whole **scorebug**, then **Game Clock**, **Home**, **Away** inside it (required); Shot Clock / Quarter optional
4. Download `scoreboard_roi.json`

## 2. Run OCR

YouTube or a direct video link:

```bash
python worker/scripts/detect_scoreboard.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --roi scoreboard_roi.json \
  --output worker/output/scoreboard_track.json
```

Or a local file:

```bash
python worker/scripts/detect_scoreboard.py \
  --video /path/to/game.mp4 \
  --roi scoreboard_roi.json \
  --output worker/output/scoreboard_track.json
```

Or a video id from the app (`GET $SCOREBUG_API_BASE/api/scorebug-videos/{id}`, default `http://localhost:3000`). The response supplies the video URL and the scorebug markings. Field boxes are already relative to the scorebug crop.

```bash
python worker/scripts/detect_scoreboard.py \
  --id VIDEO_ID \
  --output worker/output/scoreboard_track.json
```

`--url` accepts a YouTube link (`youtube.com`, `youtu.be`, Shorts) or a direct `.mp4`, `.webm`, `.mkv`, or `.mov` link. Other http(s) links are accepted when the response is `video/*`. A URL is read at 720p, video only, while OCR runs (`SCOREBUG_MAX_HEIGHT` overrides the cap). A local `--video` file is sampled from disk. Pass one of `--url`, `--video`, or `--id`. `--roi` is required unless `--id` is set. `ffmpeg` must be on `PATH`.

Flags:

- `--id` — scorebug video id; loads the URL and markings from `$SCOREBUG_API_BASE` (default `http://localhost:3000`)
- `--url` — YouTube or direct video URL (instead of `--video`)
- `--fps 1.0` — sample rate (default 1 fps)
- `--no-gpu` — force CPU for EasyOCR
- `--debug` — include raw OCR text in output
- `--no-smooth` — skip outlier smoothing

## Output

`scoreboard_track.json` includes:

- `readings` — video time → score, `gameClock` (`M:SS` / `MM:SS`), shot clock, quarter
- `gameClockToVideo` — each unique canonical `gameClock` → first video timestamp

## Droplet

GitHub Actions builds this image on the default branch and pushes it to GitHub Container Registry as `ghcr.io/<owner>/<repo>/pbp-video-mapper` (`:<short-sha>` and `:latest`). Set `SCOREBUG_API_BASE` as an Actions repository variable before the first build. That value is baked into the image, and `docker run -e SCOREBUG_API_BASE=...` can still override it.

On the GPU droplet (NVIDIA driver >= 550.54.14), with Docker and the NVIDIA container toolkit:

```bash
docker login ghcr.io
export PBP_IMAGE=ghcr.io/<owner>/<repo>/pbp-video-mapper
docker compose pull
docker compose up -d
curl -X POST http://localhost:8000/detect -H 'Content-Type: application/json' -d '{"id":"VIDEO_ID"}'
```

`docker compose up` serves the API on port 8000. `POST /detect` runs one OCR job, POSTs the track JSON to `{SCOREBUG_API_BASE}/api/scorebug-videos/{videoId}/score-timeline`, and returns that same JSON. A one-off CLI run overrides that entrypoint:

```bash
docker compose run --rm --entrypoint python worker worker/scripts/detect_scoreboard.py --id VIDEO_ID
```

## Mac Vision API

On a Mac, a separate service reads the scorebug with Apple Vision instead of Paddle. `POST /detect` posts the track JSON to `{SCOREBUG_API_BASE}/api/scorebug-videos/{videoId}/score-timeline` and returns that same JSON.

```bash
cd pbp-video-mapper
python3 -m venv .venv-mac
source .venv-mac/bin/activate
python -m pip install -r requirements-mac.txt
export SCOREBUG_API_BASE=http://localhost:3000
# ffmpeg is required. Optional: SCOREBUG_MAX_HEIGHT=480
python -m uvicorn mac.api:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/detect -H 'Content-Type: application/json' -d '{"id":"VIDEO_ID"}'
```
