"""Download a YouTube video or a direct video file."""

from __future__ import annotations

import hashlib
import shutil
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

# Single file OpenCV can open. Avoids a separate audio merge.
_YTDLP_FORMAT = "bv*[ext=mp4]/b[ext=mp4]/bv*/b"


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
        "format": _YTDLP_FORMAT,
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
