"""Download a YouTube video or a direct video file."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

WORKER_ROOT = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_DIR = WORKER_ROOT / "downloads"

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov"}

_YOUTUBE_HOSTS = {
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "youtu.be",
}

_CONTENT_TYPE_EXT = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
}

_DEFAULT_MAX_HEIGHT = 720
_DEFAULT_CONCURRENT_FRAGMENTS = 16


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_video_height() -> int:
    """Tallest YouTube frame to fetch. ``SCOREBUG_MAX_HEIGHT`` overrides 720."""
    return _positive_int_env("SCOREBUG_MAX_HEIGHT", _DEFAULT_MAX_HEIGHT)


def concurrent_fragment_count() -> int:
    """Parallel fragment or HTTP connections. Defaults to 16."""
    return _positive_int_env(
        "SCOREBUG_CONCURRENT_FRAGMENTS",
        _DEFAULT_CONCURRENT_FRAGMENTS,
    )


def youtube_format(max_height: int | None = None) -> str:
    """Video-only YouTube format at or below ``max_height``. No audio."""
    height = max_video_height() if max_height is None else max_height
    cap = f"height<={height}"
    return f"bv*[{cap}][ext=mp4]/b[{cap}][ext=mp4]/bv*[{cap}]/b[{cap}]"


def ytdlp_argv() -> list[str]:
    """Prefer the yt-dlp executable, otherwise the installed module."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def youtube_stream_command(url: str) -> list[str]:
    """yt-dlp args that write a capped video stream to stdout."""
    return [
        *ytdlp_argv(),
        "-f",
        youtube_format(),
        "--concurrent-fragments",
        str(concurrent_fragment_count()),
        "-o",
        "-",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]


def aria2c_command(url: str, dest: Path) -> list[str]:
    """Download one file with many HTTP connections."""
    connections = str(concurrent_fragment_count())
    return [
        "aria2c",
        "-x",
        connections,
        "-s",
        connections,
        "--file-allocation=none",
        "-o",
        dest.name,
        "-d",
        str(dest.parent),
        url,
    ]


class UnsupportedVideoUrl(ValueError):
    """The URL is not a YouTube link or a direct video file."""


def classify_video_url(url: str) -> str:
    """Return ``youtube``, ``direct``, or ``probe``.

    ``probe`` is an http(s) URL with no video extension. The caller must
    accept it only when the response Content-Type is ``video/*``.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UnsupportedVideoUrl(
            "Unsupported video URL. Use a YouTube link or a direct "
            ".mp4, .webm, .mkv, or .mov link."
        )

    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    if host in _YOUTUBE_HOSTS or host.endswith(".youtube.com"):
        return "youtube"

    suffix = Path(parsed.path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "direct"
    return "probe"


def extension_for_content_type(content_type: str | None) -> str | None:
    """Map a Content-Type to a video extension, or None when it is not video."""
    if not content_type:
        return None
    media = content_type.split(";", 1)[0].strip().lower()
    if media in _CONTENT_TYPE_EXT:
        return _CONTENT_TYPE_EXT[media]
    if media.startswith("video/"):
        return ".mp4"
    return None


def download_video(url: str, dest_dir: Path | str | None = None) -> Path:
    """Download ``url`` and return the local video path.

    Reuses a file already saved for the same URL.
    """
    kind = classify_video_url(url)
    directory = Path(dest_dir) if dest_dir is not None else DEFAULT_DOWNLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(url.strip().encode()).hexdigest()[:16]
    existing = _existing_download(directory, digest)
    if existing is not None:
        return existing

    if kind == "youtube":
        return _download_youtube(url.strip(), directory, digest)
    if kind == "direct":
        ext = Path(urlparse(url.strip()).path).suffix.lower()
        return _stream_download(url.strip(), directory / f"{digest}{ext}")
    return _stream_probed(url.strip(), directory, digest)


def _existing_download(directory: Path, digest: str) -> Path | None:
    matches = [
        path
        for path in directory.glob(f"{digest}.*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.endswith(".part")
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _download_youtube(url: str, directory: Path, digest: str) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required to download YouTube videos. "
            "Install with: pip install yt-dlp"
        ) from exc

    outtmpl = str(directory / f"{digest}.%(ext)s")
    options = {
        "format": youtube_format(),
        "concurrent_fragment_downloads": concurrent_fragment_count(),
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"Could not download video: {url}")
        downloaded = Path(ydl.prepare_filename(info))

    if downloaded.is_file() and downloaded.suffix.lower() in VIDEO_EXTENSIONS:
        return downloaded

    existing = _existing_download(directory, digest)
    if existing is not None:
        return existing
    raise RuntimeError(f"Download finished but no video file was saved for {url}")


def _aria2c_to(url: str, dest: Path) -> bool:
    if shutil.which("aria2c") is None:
        return False
    result = subprocess.run(
        aria2c_command(url, dest),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def download_for_fallback(url: str) -> Path:
    """Download into a fresh temp directory when a stream cannot start.

    The caller deletes ``path.parent`` after sampling. A plain file uses
    aria2c when it is installed; YouTube uses the capped yt-dlp format.
    """
    directory = Path(tempfile.mkdtemp(prefix="scorebug-"))
    try:
        if classify_video_url(url) != "youtube":
            dest = directory / "video.mp4"
            if _aria2c_to(url, dest):
                return dest
            dest.unlink(missing_ok=True)
        return download_video(url, directory)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _stream_download(url: str, dest: Path) -> Path:
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        partial.replace(dest)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest


def _stream_probed(url: str, directory: Path, digest: str) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    partial: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type")
            ext = extension_for_content_type(content_type)
            if ext is None:
                raise UnsupportedVideoUrl(
                    "Unsupported video URL. Use a YouTube link or a direct "
                    ".mp4, .webm, .mkv, or .mov link."
                )
            dest = directory / f"{digest}{ext}"
            partial = dest.with_name(dest.name + ".part")
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        partial.replace(dest)
    except urllib.error.HTTPError as exc:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
    except Exception:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise
    return dest
