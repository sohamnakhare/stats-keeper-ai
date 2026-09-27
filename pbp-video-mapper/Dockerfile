FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
        python3.11 \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG SCOREBUG_API_BASE
RUN test -n "$SCOREBUG_API_BASE"
ENV SCOREBUG_API_BASE=${SCOREBUG_API_BASE}

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', device='cpu')"

COPY worker ./worker
COPY scoreboard_roi.json ./scoreboard_roi.json

ENTRYPOINT ["python", "worker/scripts/detect_scoreboard.py"]
