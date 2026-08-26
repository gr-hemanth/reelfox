"""Data models for the audio / speech understanding layer (Phase 5).

These structures hold *speech-processing* facts only — what audio was found,
whether it contains speech, and what was transcribed.  They deliberately
contain nothing about extraction (that lives in ``extractor.models``), OCR,
vision, summaries, or any LLM output; those belong to later phases.

Everything here is a plain dataclass that serialises cleanly to a dict (and
therefore to JSON) via :meth:`SpeechResult.as_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SpeechClassification(str, Enum):
    """Speech-presence determination for a media file.

    These are the five states the spec requires us to distinguish.
    ``MIXED_OR_UNCERTAIN`` is *not* a failure — it is an honest "we cannot
    tell confidently" which the downstream consumer should treat differently
    from ``SPEECH`` or ``AUDIO_NO_SPEECH``.
    """

    NO_AUDIO = "no_audio"
    AUDIO_NO_SPEECH = "audio_no_speech"
    SPEECH = "speech"
    MIXED_OR_UNCERTAIN = "mixed_or_uncertain"


class SpeechFailureCategory(str, Enum):
    """Machine-readable failure categories for speech processing.

    These are distinct from extraction failures.  ``NO_SPEECH_DETECTED`` is
    *not* the same as ``ASR_INFERENCE_FAILED`` — the former is a legitimate
    experimental outcome, the latter is a processing error.
    """

    AUDIO_NOT_FOUND = "AUDIO_NOT_FOUND"
    AUDIO_EXTRACTION_FAILED = "AUDIO_EXTRACTION_FAILED"
    FFMPEG_NOT_AVAILABLE = "FFMPEG_NOT_AVAILABLE"
    ASR_MODEL_LOAD_FAILED = "ASR_MODEL_LOAD_FAILED"
    ASR_INFERENCE_FAILED = "ASR_INFERENCE_FAILED"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
    UNKNOWN = "UNKNOWN"


@dataclass
class SpeechSegment:
    """One timed fragment of transcribed speech."""

    start: float
    end: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class SpeechResult:
    """The structured outcome of one audio/speech processing attempt.

    Mirrors the ExtractionResult philosophy: success/failure is explicit,
    partial information is retained, and the result serialises cleanly to JSON.
    """

    success: bool

    # Audio presence ---------------------------------------------------------
    audio_present: bool = False

    # Speech presence --------------------------------------------------------
    speech_present: bool = False
    classification: str = SpeechClassification.NO_AUDIO.value

    # Transcript -------------------------------------------------------------
    transcript: Optional[str] = None
    segments: list[SpeechSegment] = field(default_factory=list)

    # Language ---------------------------------------------------------------
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None

    # Timing -----------------------------------------------------------------
    duration_seconds: Optional[float] = None
    audio_extraction_seconds: Optional[float] = None
    model_load_seconds: Optional[float] = None
    transcription_time_seconds: Optional[float] = None
    total_processing_seconds: Optional[float] = None

    # Model info -------------------------------------------------------------
    model_name: Optional[str] = None

    # Failure ----------------------------------------------------------------
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary of the whole result."""
        return {
            "success": self.success,
            "audio_present": self.audio_present,
            "speech_present": self.speech_present,
            "classification": self.classification,
            "transcript": self.transcript,
            "segments": [s.as_dict() for s in self.segments],
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "audio_extraction_seconds": self.audio_extraction_seconds,
            "model_load_seconds": self.model_load_seconds,
            "transcription_time_seconds": self.transcription_time_seconds,
            "total_processing_seconds": self.total_processing_seconds,
            "model_name": self.model_name,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
        }

    def summary_line(self) -> str:
        """A single compact line for logs."""
        if self.success:
            lang = self.detected_language or "?"
            return (
                f"OK classification={self.classification} lang={lang} "
                f"transcript_len={len(self.transcript or '')} "
                f"t={self.total_processing_seconds}"
            )
        return (
            f"FAIL category={self.failure_category} "
            f"message={self.failure_message}"
        )
