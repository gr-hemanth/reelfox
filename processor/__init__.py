"""Multimodal processing — Phase 5 (Audio/Speech) and Phase 6 (Vision Understanding).

This package provides:

- ``processor.models``: structured results (``SpeechResult``, ``SpeechSegment``,
  ``SpeechClassification``, ``SpeechFailureCategory``, ``VisionResult``,
  ``FrameObservation``, ``VisionFailureCategory``).
- ``processor.audio``: audio stream detection and extraction to temporary WAV.
- ``processor.speech``: ASR abstraction (``BaseSpeechRecognizer``) and the
  ``FasterWhisperRecognizer`` backend.
- ``processor.frames``: keyframe sampling for images and videos.
- ``processor.vision``: Vision abstraction (``BaseVisionAnalyzer``) and the
  ``LocalVisionAnalyzer`` VLM backend.
- ``processor.pipeline``: end-to-end ``process_speech`` and ``process_vision`` functions.

Later phases will add OCR, multimodal synthesis and structured output.
"""

from processor.frames import sample_frames
from processor.models import (
    FrameObservation,
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    SpeechSegment,
    VisionFailureCategory,
    VisionResult,
)
from processor.pipeline import process_speech, process_vision
from processor.speech import BaseSpeechRecognizer, FasterWhisperRecognizer
from processor.vision import BaseVisionAnalyzer, LocalVisionAnalyzer

__all__ = [
    # Models - Speech
    "SpeechClassification",
    "SpeechFailureCategory",
    "SpeechResult",
    "SpeechSegment",
    # Models - Vision
    "VisionResult",
    "FrameObservation",
    "VisionFailureCategory",
    # Speech
    "BaseSpeechRecognizer",
    "FasterWhisperRecognizer",
    # Vision
    "BaseVisionAnalyzer",
    "LocalVisionAnalyzer",
    "sample_frames",
    # Pipeline
    "process_speech",
    "process_vision",
]
