"""Offline unit tests for Phase 5 audio/speech processing.

These tests never load the real Whisper model, never download anything, and
never touch the network. All ASR components are mocked. The real integration
test lives in ``scripts/test_audio.py``.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processor.models import (  # noqa: E402
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    SpeechSegment,
)
from processor.speech import (  # noqa: E402
    BaseSpeechRecognizer,
    FasterWhisperRecognizer,
    NO_SPEECH_PROB_THRESHOLD,
)
from processor.audio import (  # noqa: E402
    AudioExtractionError,
    TARGET_SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_seconds: float = 1.0, silence: bool = True) -> Path:
    """Create a minimal valid WAV file (mono 16 kHz, 16-bit PCM).

    If *silence* is True the samples are all zeros (silence).
    Otherwise a simple 440 Hz tone is generated.
    """
    import math

    num_samples = int(TARGET_SAMPLE_RATE * duration_seconds)
    samples = []
    for i in range(num_samples):
        if silence:
            samples.append(0)
        else:
            value = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / TARGET_SAMPLE_RATE))
            samples.append(value)

    data = struct.pack(f"<{num_samples}h", *samples)
    # WAV header (44 bytes) + data
    data_size = len(data)
    file_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", file_size, b"WAVE",
        b"fmt ", 16,  # chunk size
        1,  # PCM format
        1,  # mono
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLE_RATE * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data", data_size,
    )
    path.write_bytes(header + data)
    return path


class FakeSegment:
    """Mimics a faster-whisper segment namedtuple."""

    def __init__(self, start: float, end: float, text: str, no_speech_prob: float = 0.1):
        self.start = start
        self.end = end
        self.text = text
        self.no_speech_prob = no_speech_prob


class FakeTranscriptionInfo:
    """Mimics faster-whisper TranscriptionInfo."""

    def __init__(
        self,
        language: str = "en",
        language_probability: float = 0.95,
        duration: float = 10.0,
    ):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class MockRecognizer(BaseSpeechRecognizer):
    """A test double that returns a preconfigured SpeechResult."""

    def __init__(self, result: SpeechResult):
        self._result = result

    @property
    def model_name(self) -> str:
        return "mock-model"

    def transcribe(self, audio_path):
        return self._result


# ---------------------------------------------------------------------------
# 1. No audio
# ---------------------------------------------------------------------------


class TestNoAudio:
    def test_no_audio_classification(self):
        result = SpeechResult(
            success=True,
            audio_present=False,
            speech_present=False,
            classification=SpeechClassification.NO_AUDIO.value,
        )
        assert result.classification == "no_audio"
        assert not result.audio_present
        assert not result.speech_present

    def test_pipeline_no_audio_stream(self, tmp_path):
        """Pipeline returns NO_AUDIO when media has no audio stream."""
        from processor.pipeline import process_speech

        video = tmp_path / "no_audio.mp4"
        video.write_bytes(b"\x00" * 1024)

        with patch("processor.pipeline.has_audio_stream", return_value=False):
            result = process_speech(video)

        assert result.success is True
        assert result.classification == SpeechClassification.NO_AUDIO.value
        assert not result.audio_present
        assert not result.speech_present


# ---------------------------------------------------------------------------
# 2. Audio without speech
# ---------------------------------------------------------------------------


class TestAudioNoSpeech:
    def test_audio_no_speech_result(self):
        result = SpeechResult(
            success=True,
            audio_present=True,
            speech_present=False,
            classification=SpeechClassification.AUDIO_NO_SPEECH.value,
        )
        assert result.classification == "audio_no_speech"
        assert result.audio_present
        assert not result.speech_present

    def test_recognizer_no_speech_segments(self, tmp_path):
        """When all segments have high no_speech_prob, classify as no speech."""
        wav = _make_wav(tmp_path / "silence.wav", duration_seconds=1.0)

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1

        segments = [
            FakeSegment(0.0, 2.0, "", no_speech_prob=0.9),
            FakeSegment(2.0, 4.0, "", no_speech_prob=0.95),
        ]
        info = FakeTranscriptionInfo()
        recognizer._model.transcribe.return_value = (iter(segments), info)

        result = recognizer.transcribe(wav)
        assert result.success is True
        assert result.classification == SpeechClassification.AUDIO_NO_SPEECH.value
        assert not result.speech_present
        assert result.transcript is None


# ---------------------------------------------------------------------------
# 3. Speech detected
# ---------------------------------------------------------------------------


class TestSpeechDetected:
    def test_speech_classification(self, tmp_path):
        """When most segments have speech, classify as SPEECH."""
        wav = _make_wav(tmp_path / "speech.wav")

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1

        segments = [
            FakeSegment(0.0, 3.0, "Hello world this is a test", no_speech_prob=0.1),
            FakeSegment(3.0, 6.0, "Another segment with real speech", no_speech_prob=0.15),
            FakeSegment(6.0, 9.0, "More speech content here", no_speech_prob=0.2),
        ]
        info = FakeTranscriptionInfo(language="en", language_probability=0.98, duration=9.0)
        recognizer._model.transcribe.return_value = (iter(segments), info)

        result = recognizer.transcribe(wav)
        assert result.success is True
        assert result.classification == SpeechClassification.SPEECH.value
        assert result.speech_present is True
        assert "Hello world" in result.transcript
        assert result.detected_language == "en"
        assert result.language_probability == 0.98
        assert len(result.segments) == 3


# ---------------------------------------------------------------------------
# 4. Mixed / uncertain classification
# ---------------------------------------------------------------------------


class TestMixedClassification:
    def test_mixed_when_some_speech_some_not(self, tmp_path):
        """When speech ratio is between low and high, classify as MIXED."""
        wav = _make_wav(tmp_path / "mixed.wav")

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1

        # 2 speech + 3 non-speech = 40% speech ratio -> MIXED_OR_UNCERTAIN
        segments = [
            FakeSegment(0.0, 2.0, "Some speech here today", no_speech_prob=0.1),
            FakeSegment(2.0, 4.0, "", no_speech_prob=0.9),
            FakeSegment(4.0, 6.0, "", no_speech_prob=0.85),
            FakeSegment(6.0, 8.0, "Another speech bit coming now", no_speech_prob=0.2),
            FakeSegment(8.0, 10.0, "", no_speech_prob=0.95),
        ]
        info = FakeTranscriptionInfo(duration=10.0)
        recognizer._model.transcribe.return_value = (iter(segments), info)

        result = recognizer.transcribe(wav)
        assert result.success is True
        assert result.classification == SpeechClassification.MIXED_OR_UNCERTAIN.value
        assert result.speech_present is True


# ---------------------------------------------------------------------------
# 5. Transcription success (full pipeline with mock)
# ---------------------------------------------------------------------------


class TestTranscriptionSuccess:
    def test_pipeline_with_mock_recognizer(self, tmp_path):
        from processor.pipeline import process_speech

        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 2048)

        mock_result = SpeechResult(
            success=True,
            audio_present=True,
            speech_present=True,
            classification=SpeechClassification.SPEECH.value,
            transcript="Hello world from the test",
            detected_language="en",
            language_probability=0.95,
            segments=[SpeechSegment(0.0, 3.0, "Hello world from the test")],
            model_name="mock-model",
        )
        recognizer = MockRecognizer(mock_result)

        wav = _make_wav(tmp_path / "extracted.wav")

        with (
            patch("processor.pipeline.has_audio_stream", return_value=True),
            patch("processor.pipeline.extract_audio_to_wav", return_value=(wav, 3.0, 0.5)),
        ):
            result = process_speech(video, recognizer=recognizer)

        assert result.success is True
        assert result.speech_present is True
        assert result.transcript == "Hello world from the test"


# ---------------------------------------------------------------------------
# 6. Model load failure
# ---------------------------------------------------------------------------


class TestModelLoadFailure:
    def test_model_load_error_captured(self, tmp_path):
        wav = _make_wav(tmp_path / "test.wav")

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = None
        recognizer._model_load_seconds = None

        with patch(
            "processor.speech.WhisperModel",
            side_effect=RuntimeError("CUDA not available"),
            create=True,
        ):
            # Patch the import inside _ensure_model
            with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
                fake_module = sys.modules["faster_whisper"]
                fake_module.WhisperModel = MagicMock(
                    side_effect=RuntimeError("CUDA not available")
                )
                result = recognizer.transcribe(wav)

        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.ASR_MODEL_LOAD_FAILED.value
        assert "CUDA not available" in result.failure_message


# ---------------------------------------------------------------------------
# 7. Inference failure
# ---------------------------------------------------------------------------


class TestInferenceFailure:
    def test_inference_error_captured(self, tmp_path):
        wav = _make_wav(tmp_path / "test.wav")

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1
        recognizer._model.transcribe.side_effect = RuntimeError("inference crashed")

        result = recognizer.transcribe(wav)
        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.ASR_INFERENCE_FAILED.value
        assert "inference crashed" in result.failure_message


# ---------------------------------------------------------------------------
# 8. FFmpeg / PyAV missing
# ---------------------------------------------------------------------------


class TestFfmpegMissing:
    def test_audio_extraction_error_ffmpeg(self, tmp_path):
        from processor.pipeline import process_speech

        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 2048)

        with (
            patch("processor.pipeline.has_audio_stream", return_value=True),
            patch(
                "processor.pipeline.extract_audio_to_wav",
                side_effect=AudioExtractionError(
                    "FFMPEG_NOT_AVAILABLE",
                    "PyAV (av) library is not installed.",
                ),
            ),
        ):
            result = process_speech(video)

        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.FFMPEG_NOT_AVAILABLE.value


# ---------------------------------------------------------------------------
# 9. Temporary audio cleanup
# ---------------------------------------------------------------------------


class TestTempAudioCleanup:
    def test_temp_wav_deleted_after_processing(self, tmp_path):
        from processor.pipeline import process_speech

        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 2048)
        wav = _make_wav(tmp_path / "temp_audio.wav")
        assert wav.exists()

        mock_result = SpeechResult(
            success=True,
            audio_present=True,
            speech_present=False,
            classification=SpeechClassification.AUDIO_NO_SPEECH.value,
            model_name="mock-model",
        )
        recognizer = MockRecognizer(mock_result)

        with (
            patch("processor.pipeline.has_audio_stream", return_value=True),
            patch("processor.pipeline.extract_audio_to_wav", return_value=(wav, 1.0, 0.1)),
        ):
            result = process_speech(video, recognizer=recognizer)

        assert not wav.exists(), "Temporary WAV should be deleted after processing"

    def test_temp_wav_kept_when_debug(self, tmp_path):
        from processor.pipeline import process_speech

        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 2048)
        wav = _make_wav(tmp_path / "keep_audio.wav")

        mock_result = SpeechResult(
            success=True,
            audio_present=True,
            speech_present=False,
            classification=SpeechClassification.AUDIO_NO_SPEECH.value,
            model_name="mock-model",
        )
        recognizer = MockRecognizer(mock_result)

        with (
            patch("processor.pipeline.has_audio_stream", return_value=True),
            patch("processor.pipeline.extract_audio_to_wav", return_value=(wav, 1.0, 0.1)),
        ):
            result = process_speech(video, recognizer=recognizer, keep_audio=True)

        assert wav.exists(), "Temporary WAV should be kept when keep_audio=True"


# ---------------------------------------------------------------------------
# 10. Transcript serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_speech_result_to_dict(self):
        result = SpeechResult(
            success=True,
            audio_present=True,
            speech_present=True,
            classification=SpeechClassification.SPEECH.value,
            transcript="Hello world",
            segments=[SpeechSegment(0.0, 2.0, "Hello world")],
            detected_language="en",
            language_probability=0.95,
            duration_seconds=5.0,
            model_name="faster-whisper-base",
        )
        d = result.as_dict()
        text = json.dumps(d)
        assert '"success": true' in text
        assert '"transcript": "Hello world"' in text
        assert '"detected_language": "en"' in text
        assert len(d["segments"]) == 1
        assert d["segments"][0]["start"] == 0.0
        assert d["segments"][0]["text"] == "Hello world"

    def test_speech_result_serialisation_all_fields(self):
        result = SpeechResult(success=False)
        d = result.as_dict()
        expected_keys = {
            "success", "audio_present", "speech_present", "classification",
            "transcript", "segments", "detected_language", "language_probability",
            "duration_seconds", "audio_extraction_seconds", "model_load_seconds",
            "transcription_time_seconds", "total_processing_seconds",
            "model_name", "failure_category", "failure_message",
        }
        assert set(d.keys()) == expected_keys

    def test_speech_segment_as_dict(self):
        seg = SpeechSegment(start=1.5, end=3.7, text="test segment")
        d = seg.as_dict()
        assert d == {"start": 1.5, "end": 3.7, "text": "test segment"}

    def test_summary_line_success(self):
        result = SpeechResult(
            success=True,
            classification=SpeechClassification.SPEECH.value,
            detected_language="en",
            transcript="Hello",
            total_processing_seconds=1.5,
        )
        line = result.summary_line()
        assert "OK" in line
        assert "en" in line

    def test_summary_line_failure(self):
        result = SpeechResult(
            success=False,
            failure_category="ASR_INFERENCE_FAILED",
            failure_message="boom",
        )
        line = result.summary_line()
        assert "FAIL" in line
        assert "boom" in line


# ---------------------------------------------------------------------------
# 11. Language handling
# ---------------------------------------------------------------------------


class TestLanguage:
    def test_language_detection_passthrough(self, tmp_path):
        wav = _make_wav(tmp_path / "hindi.wav")

        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1

        segments = [
            FakeSegment(0.0, 5.0, "यह एक परीक्षण है नमस्ते", no_speech_prob=0.05),
        ]
        info = FakeTranscriptionInfo(language="hi", language_probability=0.88, duration=5.0)
        recognizer._model.transcribe.return_value = (iter(segments), info)

        result = recognizer.transcribe(wav)
        assert result.detected_language == "hi"
        assert result.language_probability == 0.88
        assert "परीक्षण" in result.transcript

    def test_no_hardcoded_english(self):
        """Ensure the speech module doesn't hard-code English."""
        source = (PROJECT_ROOT / "processor" / "speech.py").read_text(encoding="utf-8")
        # Check there's no language="en" forced in transcribe calls.
        assert 'language="en"' not in source
        assert "language='en'" not in source


# ---------------------------------------------------------------------------
# 12. No secret leakage
# ---------------------------------------------------------------------------


class TestNoSecretLeakage:
    def test_speech_result_has_no_path_fields(self):
        """SpeechResult should not contain file paths that might leak info."""
        result = SpeechResult(success=True)
        d = result.as_dict()
        for key in d:
            assert "path" not in key.lower() or key == "media_path"

    def test_no_api_key_imports_in_processor(self):
        """Processor modules must not import API-key-related libraries."""
        for filename in ("audio.py", "speech.py", "pipeline.py", "models.py"):
            source = (PROJECT_ROOT / "processor" / filename).read_text(encoding="utf-8")
            for line in source.splitlines():
                if line.startswith(("import ", "from ")):
                    for lib in ("openai", "anthropic", "requests", "httpx"):
                        assert lib not in line, (
                            f"{filename} imports {lib} — Phase 5 must not use paid APIs"
                        )


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------


class TestEnums:
    def test_speech_classification_values(self):
        assert SpeechClassification.NO_AUDIO.value == "no_audio"
        assert SpeechClassification.AUDIO_NO_SPEECH.value == "audio_no_speech"
        assert SpeechClassification.SPEECH.value == "speech"
        assert SpeechClassification.MIXED_OR_UNCERTAIN.value == "mixed_or_uncertain"

    def test_failure_category_values(self):
        assert SpeechFailureCategory.AUDIO_NOT_FOUND.value == "AUDIO_NOT_FOUND"
        assert SpeechFailureCategory.ASR_MODEL_LOAD_FAILED.value == "ASR_MODEL_LOAD_FAILED"
        assert SpeechFailureCategory.FFMPEG_NOT_AVAILABLE.value == "FFMPEG_NOT_AVAILABLE"
        assert SpeechFailureCategory.NO_SPEECH_DETECTED.value == "NO_SPEECH_DETECTED"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_media_file(self, tmp_path):
        from processor.pipeline import process_speech

        result = process_speech(tmp_path / "nonexistent.mp4")
        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.AUDIO_NOT_FOUND.value

    def test_unsupported_extension(self, tmp_path):
        from processor.pipeline import process_speech

        txt = tmp_path / "file.txt"
        txt.write_text("not media")
        result = process_speech(txt)
        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.UNSUPPORTED_MEDIA.value

    def test_recognizer_missing_audio_file(self, tmp_path):
        recognizer = FasterWhisperRecognizer.__new__(FasterWhisperRecognizer)
        recognizer._model_size = "base"
        recognizer._device = "cpu"
        recognizer._compute_type = "int8"
        recognizer._model = MagicMock()
        recognizer._model_load_seconds = 0.1

        result = recognizer.transcribe(tmp_path / "ghost.wav")
        assert result.success is False
        assert result.failure_category == SpeechFailureCategory.AUDIO_NOT_FOUND.value


# ---------------------------------------------------------------------------
# ASR config from environment
# ---------------------------------------------------------------------------


class TestASRConfig:
    def test_default_config(self, monkeypatch):
        from processor.speech import _asr_config

        monkeypatch.delenv("ASR_MODEL_SIZE", raising=False)
        monkeypatch.delenv("ASR_DEVICE", raising=False)
        monkeypatch.delenv("ASR_COMPUTE_TYPE", raising=False)

        model, device, compute = _asr_config()
        assert model == "base"
        assert device == "cpu"
        assert compute == "int8"

    def test_custom_config(self, monkeypatch):
        from processor.speech import _asr_config

        monkeypatch.setenv("ASR_MODEL_SIZE", "small")
        monkeypatch.setenv("ASR_DEVICE", "cuda")
        monkeypatch.setenv("ASR_COMPUTE_TYPE", "float16")

        model, device, compute = _asr_config()
        assert model == "small"
        assert device == "cuda"
        assert compute == "float16"

    def test_invalid_config_falls_back(self, monkeypatch):
        from processor.speech import _asr_config

        monkeypatch.setenv("ASR_MODEL_SIZE", "xlarge")
        monkeypatch.setenv("ASR_DEVICE", "tpu")
        monkeypatch.setenv("ASR_COMPUTE_TYPE", "bfloat16")

        model, device, compute = _asr_config()
        assert model == "base"
        assert device == "cpu"
        assert compute == "int8"
