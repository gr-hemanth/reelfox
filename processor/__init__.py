"""Multimodal processing — Phase 5: Audio / Speech Understanding.

This package provides:

- ``processor.models``: structured results (``SpeechResult``, ``SpeechSegment``,
  ``SpeechClassification``, ``SpeechFailureCategory``).
- ``processor.audio``: audio stream detection and extraction to temporary WAV.
- ``processor.speech``: ASR abstraction (``BaseSpeechRecognizer``) and the
  ``FasterWhisperRecognizer`` backend.
- ``processor.pipeline``: the end-to-end ``process_speech`` function that
  chains audio extraction → speech detection → transcription.

Later phases will add vision, OCR, multimodal synthesis and structured output.
"""

from processor.models import (
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    SpeechSegment,
)
from processor.speech import BaseSpeechRecognizer, FasterWhisperRecognizer

__all__ = [
    # Models
    "SpeechClassification",
    "SpeechFailureCategory",
    "SpeechResult",
    "SpeechSegment",
    # Speech
    "BaseSpeechRecognizer",
    "FasterWhisperRecognizer",
]
