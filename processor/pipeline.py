"""End-to-end audio/speech processing pipeline (Phase 5).

Chains together:
1. Audio stream detection.
2. Audio extraction to temporary mono 16 kHz WAV.
3. Speech classification and transcription via ASR.
4. Temporary audio cleanup.

The caller supplies a path to an already-extracted media file (from the
extractor layer). This module does NOT download Instagram content.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from processor.audio import AudioExtractionError, extract_audio_to_wav, has_audio_stream
from processor.models import (
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
)
from processor.speech import BaseSpeechRecognizer, FasterWhisperRecognizer

logger = logging.getLogger("analyzer.processor.pipeline")


def process_speech(
    media_path: str | Path,
    recognizer: BaseSpeechRecognizer | None = None,
    keep_audio: bool = False,
) -> SpeechResult:
    """Run the full audio/speech pipeline on *media_path*.

    Parameters
    ----------
    media_path:
        Path to the extracted video/audio file.
    recognizer:
        ASR backend to use. If ``None``, a default
        :class:`FasterWhisperRecognizer` is created from environment config.
    keep_audio:
        If True, the temporary WAV is *not* deleted after processing.
        Intended for debugging only.

    Returns
    -------
    SpeechResult
        A fully populated result. Never raises — errors are captured in the
        result's failure fields.
    """
    total_t0 = time.perf_counter()
    media_path = Path(media_path)

    # Validate input.
    if not media_path.exists():
        return SpeechResult(
            success=False,
            failure_category=SpeechFailureCategory.AUDIO_NOT_FOUND.value,
            failure_message=f"Media file does not exist: {media_path}",
        )

    # Check for supported media types.
    suffix = media_path.suffix.lower()
    supported = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    if suffix not in supported:
        return SpeechResult(
            success=False,
            failure_category=SpeechFailureCategory.UNSUPPORTED_MEDIA.value,
            failure_message=f"Unsupported media extension: {suffix}",
        )

    # Step 1: probe for audio stream.
    if not has_audio_stream(media_path):
        return SpeechResult(
            success=True,
            audio_present=False,
            speech_present=False,
            classification=SpeechClassification.NO_AUDIO.value,
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    # Step 2: extract audio to temporary WAV.
    wav_path: Optional[Path] = None
    try:
        wav_path, duration, extraction_seconds = extract_audio_to_wav(media_path)
    except AudioExtractionError as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=exc.category,
            failure_message=exc.message,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    except Exception as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=SpeechFailureCategory.AUDIO_EXTRACTION_FAILED.value,
            failure_message=str(exc),
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    # Step 3: transcribe.
    try:
        if recognizer is None:
            recognizer = FasterWhisperRecognizer()

        result = recognizer.transcribe(wav_path)

        # Enrich the result with pipeline-level timing.
        result.audio_present = True
        result.audio_extraction_seconds = extraction_seconds
        result.duration_seconds = duration or result.duration_seconds
        result.total_processing_seconds = time.perf_counter() - total_t0

        return result
    except Exception as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=SpeechFailureCategory.ASR_INFERENCE_FAILED.value,
            failure_message=str(exc),
            audio_extraction_seconds=extraction_seconds,
            duration_seconds=duration,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    finally:
        # Step 4: clean up temporary audio.
        if wav_path and wav_path.exists() and not keep_audio:
            try:
                wav_path.unlink()
                logger.debug("Cleaned up temporary audio: %s", wav_path)
            except OSError as exc:
                logger.warning("Could not delete temporary audio %s: %s", wav_path, exc)
