"""Speech recognition abstraction and faster-whisper backend (Phase 5).

Provides:
- ``BaseSpeechRecognizer``: an abstract interface so the rest of the app
  never depends directly on a specific ASR library.
- ``FasterWhisperRecognizer``: the concrete backend using ``faster-whisper``
  with CTranslate2 for local, free, CPU-based transcription.

Configuration is environment-driven:
    ASR_MODEL_SIZE   = tiny | base | small        (default: base)
    ASR_DEVICE       = cpu | cuda | auto          (default: cpu)
    ASR_COMPUTE_TYPE = int8 | float16 | float32   (default: int8)

The model is downloaded automatically on first use through HuggingFace Hub.
After download, inference is entirely local with zero API calls.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from processor.models import (
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    SpeechSegment,
)

logger = logging.getLogger("analyzer.processor.speech")

# ---------------------------------------------------------------------------
# Environment-based configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"

VALID_MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
VALID_DEVICES = ("cpu", "cuda", "auto")
VALID_COMPUTE_TYPES = ("int8", "float16", "float32", "int8_float16")


def _asr_config() -> tuple[str, str, str]:
    """Read ASR configuration from environment variables."""
    model = os.environ.get("ASR_MODEL_SIZE", "").strip().lower() or DEFAULT_MODEL_SIZE
    if model not in VALID_MODEL_SIZES:
        logger.warning(
            "Invalid ASR_MODEL_SIZE=%r, falling back to %r", model, DEFAULT_MODEL_SIZE
        )
        model = DEFAULT_MODEL_SIZE

    device = os.environ.get("ASR_DEVICE", "").strip().lower() or DEFAULT_DEVICE
    if device not in VALID_DEVICES:
        logger.warning(
            "Invalid ASR_DEVICE=%r, falling back to %r", device, DEFAULT_DEVICE
        )
        device = DEFAULT_DEVICE

    compute = (
        os.environ.get("ASR_COMPUTE_TYPE", "").strip().lower() or DEFAULT_COMPUTE_TYPE
    )
    if compute not in VALID_COMPUTE_TYPES:
        logger.warning(
            "Invalid ASR_COMPUTE_TYPE=%r, falling back to %r",
            compute,
            DEFAULT_COMPUTE_TYPE,
        )
        compute = DEFAULT_COMPUTE_TYPE

    return model, device, compute


# ---------------------------------------------------------------------------
# Abstract recognizer
# ---------------------------------------------------------------------------


class BaseSpeechRecognizer(ABC):
    """Interface that the rest of the application depends on.

    Subclasses implement one method: :meth:`transcribe`, which takes a path
    to a pre-processed audio file and returns a :class:`SpeechResult`.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> SpeechResult:
        """Transcribe the audio at *audio_path* and return a structured result.

        *audio_path* should point to a mono 16 kHz WAV file (the output of
        :func:`processor.audio.extract_audio_to_wav`).
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for the loaded model."""


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

# Speech-presence thresholds derived from Whisper's no-speech probability.
# Segments with ``no_speech_prob`` above this are considered non-speech.
NO_SPEECH_PROB_THRESHOLD = 0.6

# If fewer than this fraction of total segments are speech, classify as
# AUDIO_NO_SPEECH. If between this and the upper bound, MIXED_OR_UNCERTAIN.
SPEECH_RATIO_LOW = 0.10
SPEECH_RATIO_HIGH = 0.50

# Minimum number of words in a transcript to consider it valid speech.
MIN_WORD_COUNT = 2


class FasterWhisperRecognizer(BaseSpeechRecognizer):
    """ASR backend using ``faster-whisper`` (CTranslate2).

    The model is lazily loaded on the first call to :meth:`transcribe` and
    cached for the lifetime of this object. Model download happens
    automatically via HuggingFace Hub on first use.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        env_model, env_device, env_compute = _asr_config()
        self._model_size = model_size or env_model
        self._device = device or env_device
        self._compute_type = compute_type or env_compute
        self._model = None
        self._model_load_seconds: Optional[float] = None

    @property
    def model_name(self) -> str:
        return f"faster-whisper-{self._model_size}"

    def _ensure_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self._model is not None:
            return

        logger.info(
            "Loading faster-whisper model: size=%s device=%s compute=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )
        t0 = time.perf_counter()
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:
            self._model_load_seconds = time.perf_counter() - t0
            logger.error("Failed to load ASR model: %s", exc)
            raise
        self._model_load_seconds = time.perf_counter() - t0
        logger.info("Model loaded in %.2fs", self._model_load_seconds)

    def transcribe(self, audio_path: str | Path) -> SpeechResult:
        """Transcribe the audio file and return a structured result."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return SpeechResult(
                success=False,
                failure_category=SpeechFailureCategory.AUDIO_NOT_FOUND.value,
                failure_message=f"Audio file does not exist: {audio_path}",
                model_name=self.model_name,
            )

        # Load model.
        try:
            self._ensure_model()
        except Exception as exc:
            return SpeechResult(
                success=False,
                failure_category=SpeechFailureCategory.ASR_MODEL_LOAD_FAILED.value,
                failure_message=str(exc),
                model_name=self.model_name,
                model_load_seconds=self._model_load_seconds,
            )

        # Run inference.
        t0 = time.perf_counter()
        try:
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
            )
            # Materialise segments — the generator is lazy.
            raw_segments = list(segments_gen)
        except Exception as exc:
            return SpeechResult(
                success=False,
                failure_category=SpeechFailureCategory.ASR_INFERENCE_FAILED.value,
                failure_message=str(exc),
                model_name=self.model_name,
                model_load_seconds=self._model_load_seconds,
            )

        transcription_time = time.perf_counter() - t0

        # Classify speech presence based on segments.
        return self._build_result(
            raw_segments=raw_segments,
            info=info,
            transcription_time=transcription_time,
        )

    def _build_result(
        self,
        raw_segments: list,
        info,
        transcription_time: float,
    ) -> SpeechResult:
        """Assemble a SpeechResult from faster-whisper output."""
        # Extract language info.
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)
        duration = getattr(info, "duration", None)

        # Build segment list and classify speech.
        segments: list[SpeechSegment] = []
        speech_segments = 0
        non_speech_segments = 0

        for seg in raw_segments:
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
            text = seg.text.strip()
            segment = SpeechSegment(
                start=seg.start,
                end=seg.end,
                text=text,
            )
            segments.append(segment)
            if no_speech_prob < NO_SPEECH_PROB_THRESHOLD and text:
                speech_segments += 1
            else:
                non_speech_segments += 1

        total_segments = speech_segments + non_speech_segments
        full_transcript = " ".join(s.text for s in segments if s.text).strip()
        word_count = len(full_transcript.split()) if full_transcript else 0

        # Determine speech classification.
        if total_segments == 0 or word_count < MIN_WORD_COUNT:
            classification = SpeechClassification.AUDIO_NO_SPEECH
            speech_present = False
        else:
            speech_ratio = speech_segments / total_segments
            if speech_ratio >= SPEECH_RATIO_HIGH:
                classification = SpeechClassification.SPEECH
                speech_present = True
            elif speech_ratio >= SPEECH_RATIO_LOW:
                classification = SpeechClassification.MIXED_OR_UNCERTAIN
                speech_present = True
            else:
                classification = SpeechClassification.AUDIO_NO_SPEECH
                speech_present = False

        # If classification says no speech, don't return a transcript.
        if not speech_present:
            full_transcript = None
            segments = []

        return SpeechResult(
            success=True,
            audio_present=True,
            speech_present=speech_present,
            classification=classification.value,
            transcript=full_transcript or None,
            segments=segments,
            detected_language=detected_language,
            language_probability=language_probability,
            duration_seconds=duration,
            model_load_seconds=self._model_load_seconds,
            transcription_time_seconds=transcription_time,
            model_name=self.model_name,
        )
