"""Deterministic frame sampler for video and image inputs (Phase 6).

Extracts a configurable number of evenly-spaced keyframes from video files
to temporary JPEG images, or handles single image inputs directly.
Uses PyAV (av) which is already bundled and requires no system FFmpeg.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import av
from PIL import Image

logger = logging.getLogger("analyzer.processor.frames")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


def sample_frames(
    media_path: str | Path,
    max_frames: int = 6,
) -> Tuple[List[Path], float, float]:
    """Sample up to *max_frames* keyframes from *media_path*.

    Parameters
    ----------
    media_path:
        Path to image or video file.
    max_frames:
        Maximum number of frames to sample (default: 6).

    Returns
    -------
    Tuple[List[Path], float, float]
        (list_of_frame_file_paths, media_duration_seconds, extraction_time_seconds)

    Raises
    ------
    FrameExtractionError
        If media does not exist, format is unsupported, or decoding fails.
    """
    t0 = time.perf_counter()
    media_path = Path(media_path)

    if not media_path.exists():
        raise FrameExtractionError("MEDIA_NOT_FOUND", f"File does not exist: {media_path}")

    ext = media_path.suffix.lower()

    # Case 1: Single image input
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        try:
            with Image.open(media_path) as img:
                img.verify()
            extraction_time = time.perf_counter() - t0
            return [media_path], 0.0, extraction_time
        except Exception as exc:
            raise FrameExtractionError("FRAME_EXTRACTION_FAILED", f"Corrupt image file: {exc}")

    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise FrameExtractionError("UNSUPPORTED_MEDIA", f"Unsupported extension for vision: {ext}")

    # Case 2: Video input
    try:
        container = av.open(str(media_path))
    except Exception as exc:
        raise FrameExtractionError("FRAME_EXTRACTION_FAILED", f"Cannot open video container: {exc}")

    try:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            container.close()
            raise FrameExtractionError("UNSUPPORTED_MEDIA", f"No video stream found in {media_path}")

        duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration and video_stream.time_base else 0.0
        if duration <= 0.0 and container.duration:
            duration = float(container.duration / av.TIME_BASE)

        max_frames = max(1, max_frames)
        frame_paths: List[Path] = []

        if duration <= 0.0:
            # Fallback: iterate and sample sequentially if duration unknown
            count = 0
            for i, frame in enumerate(container.decode(video_stream)):
                if i % 30 == 0 and len(frame_paths) < max_frames:
                    img = frame.to_image()
                    temp_file = Path(tempfile.gettempdir()) / f"reelfox_frame_{time.time_ns()}_{count}.jpg"
                    img.save(temp_file, format="JPEG", quality=85)
                    frame_paths.append(temp_file)
                    count += 1
                if len(frame_paths) >= max_frames:
                    break
        else:
            # Calculate target timestamps in seconds
            if max_frames == 1:
                target_times = [duration / 2.0]
            else:
                step = duration / (max_frames - 1) if max_frames > 1 else 0
                # Clamp to slightly inside boundaries to avoid end-of-file decode issues
                target_times = [min(i * step, max(0.0, duration - 0.1)) for i in range(max_frames)]

            for idx, target_sec in enumerate(target_times):
                try:
                    # Seek to target time in time_base units
                    pts = int(target_sec / float(video_stream.time_base))
                    container.seek(pts, stream=video_stream)
                    for frame in container.decode(video_stream):
                        img = frame.to_image()
                        temp_file = Path(tempfile.gettempdir()) / f"reelfox_frame_{time.time_ns()}_{idx}.jpg"
                        img.save(temp_file, format="JPEG", quality=85)
                        frame_paths.append(temp_file)
                        break
                except Exception as exc:
                    logger.warning("Failed to extract frame at t=%.2fs: %s", target_sec, exc)

        container.close()

        if not frame_paths:
            raise FrameExtractionError("FRAME_EXTRACTION_FAILED", "Failed to extract any valid frames from video.")

        extraction_time = time.perf_counter() - t0
        logger.debug("Extracted %d frames from video in %.3fs", len(frame_paths), extraction_time)
        return frame_paths, duration, extraction_time

    except FrameExtractionError:
        raise
    except Exception as exc:
        container.close()
        raise FrameExtractionError("FRAME_EXTRACTION_FAILED", f"Unexpected frame extraction error: {exc}")
