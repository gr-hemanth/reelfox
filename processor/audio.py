"""Audio extraction from media files (Phase 5).

Responsible for:
1. Detecting whether a media file contains an audio stream.
2. Extracting the audio to a temporary mono 16 kHz WAV suitable for ASR.

Uses the ``av`` library (PyAV), which is already a dependency of
``faster-whisper`` and bundles its own FFmpeg libraries.  No system FFmpeg
installation is required.

The caller owns the returned temporary WAV path and must delete it when done.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("analyzer.processor.audio")

# Target sample rate for Whisper
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1  # mono


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from a media file."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def has_audio_stream(media_path: str | Path) -> bool:
    """Return True if the media file contains at least one audio stream.

    This is a quick probe — it does not decode audio data.
    """
    try:
        import av
    except ImportError:
        logger.warning("PyAV (av) is not installed; cannot probe audio streams")
        return False

    try:
        with av.open(str(media_path)) as container:
            return len(container.streams.audio) > 0
    except Exception as exc:
        logger.debug("Could not probe audio in %s: %s", media_path, exc)
        return False


def extract_audio_to_wav(
    media_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, float, float]:
    """Extract audio from *media_path* to a temporary mono 16 kHz WAV.

    Parameters
    ----------
    media_path:
        Path to the source video/audio file.
    output_dir:
        Optional directory for the temp WAV.  Defaults to the system temp dir.

    Returns
    -------
    tuple of (wav_path, duration_seconds, extraction_seconds)
        *wav_path* is the path to the newly created WAV file.
        *duration_seconds* is the audio duration in seconds.
        *extraction_seconds* is wall-clock time taken for extraction.

    Raises
    ------
    AudioExtractionError
        If the file has no audio or extraction fails.
    """
    try:
        import av
    except ImportError as exc:
        raise AudioExtractionError(
            "FFMPEG_NOT_AVAILABLE",
            "PyAV (av) library is not installed. Install it with: "
            "pip install av",
        ) from exc

    media_path = Path(media_path)
    if not media_path.exists():
        raise AudioExtractionError(
            "AUDIO_NOT_FOUND",
            f"Media file does not exist: {media_path}",
        )

    t0 = time.perf_counter()
    try:
        input_container = av.open(str(media_path))
    except Exception as exc:
        raise AudioExtractionError(
            "AUDIO_EXTRACTION_FAILED",
            f"Cannot open media file: {exc}",
        ) from exc

    if len(input_container.streams.audio) == 0:
        input_container.close()
        raise AudioExtractionError(
            "AUDIO_NOT_FOUND",
            "Media file has no audio stream.",
        )

    audio_stream = input_container.streams.audio[0]

    # Calculate duration from the stream or container.
    if audio_stream.duration and audio_stream.time_base:
        duration = float(audio_stream.duration * audio_stream.time_base)
    elif input_container.duration:
        duration = input_container.duration / av.time_base
    else:
        duration = 0.0

    # Create temp WAV.
    suffix = ".wav"
    fd, wav_path_str = tempfile.mkstemp(
        suffix=suffix,
        prefix="reelfox_audio_",
        dir=str(output_dir) if output_dir else None,
    )
    import os
    os.close(fd)
    wav_path = Path(wav_path_str)

    try:
        # Set up the resampler for mono 16 kHz s16.
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )

        output_container = av.open(str(wav_path), mode="w")
        output_stream = output_container.add_stream("pcm_s16le", rate=TARGET_SAMPLE_RATE)
        output_stream.layout = "mono"

        for frame in input_container.decode(audio=0):
            resampled = resampler.resample(frame)
            for resampled_frame in resampled:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

        # Flush encoder.
        for packet in output_stream.encode():
            output_container.mux(packet)

        output_container.close()
    except Exception as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            "AUDIO_EXTRACTION_FAILED",
            f"Failed to extract audio: {exc}",
        ) from exc
    finally:
        input_container.close()

    extraction_seconds = time.perf_counter() - t0
    logger.info(
        "Audio extracted: %s -> %s (%.1fs, %.1fs duration)",
        media_path.name,
        wav_path.name,
        extraction_seconds,
        duration,
    )
    return wav_path, duration, extraction_seconds
