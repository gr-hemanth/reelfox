"""Offline unit tests for Phase 6 vision understanding.

These tests never load real weights, never touch network/HuggingFace Hub, and mock
the vision model completely. The real integration test lives in ``scripts/test_vision.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processor.frames import FrameExtractionError, sample_frames  # noqa: E402
from processor.models import (  # noqa: E402
    FrameObservation,
    VisionFailureCategory,
    VisionResult,
)
from processor.pipeline import process_vision  # noqa: E402
from processor.vision import BaseVisionAnalyzer, LocalVisionAnalyzer  # noqa: E402


class MockVisionAnalyzer(BaseVisionAnalyzer):
    """Test double returning a preconfigured VisionResult."""

    def __init__(self, result: VisionResult):
        self._result = result

    @property
    def model_name(self) -> str:
        return "mock-vision-model"

    def analyze_frames(
        self,
        frame_paths: list[Path],
        media_path: Path | None = None,
        input_type: str = "video",
        frame_extraction_seconds: float | None = None,
    ) -> VisionResult:
        self._result.frames_analyzed = len(frame_paths)
        self._result.media_path = str(media_path) if media_path else None
        self._result.input_type = input_type
        return self._result


def _create_dummy_image(path: Path) -> Path:
    """Create a minimal 10x10 JPEG image."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="blue")
    img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# 1. Single image processing
# ---------------------------------------------------------------------------


class TestSingleImageProcessing:
    def test_image_processing_success(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "test.jpg")

        mock_res = VisionResult(
            success=True,
            observations=["A blue square image."],
            subjects=["square"],
            objects=["blue box"],
            actions=["displaying"],
            scenes=["indoor"],
            model_name="mock-vision",
        )
        analyzer = MockVisionAnalyzer(mock_res)

        result = process_vision(img_path, analyzer=analyzer)

        assert result.success is True
        assert result.input_type == "image"
        assert result.frames_analyzed == 1
        assert "A blue square image." in result.observations


# ---------------------------------------------------------------------------
# 2. Video frame sampling logic
# ---------------------------------------------------------------------------


class TestVideoFrameSampling:
    def test_image_returns_single_frame(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "frame.jpg")
        frames, duration, ext_time = sample_frames(img_path, max_frames=6)
        assert len(frames) == 1
        assert frames[0] == img_path
        assert duration == 0.0

    def test_sampling_max_frames_respected(self, tmp_path):
        """Frame sampler respects configured max_frames."""
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"\x00" * 100)

        with patch("av.open") as mock_av_open:
            mock_container = MagicMock()
            mock_stream = MagicMock()
            mock_stream.type = "video"
            mock_stream.duration = 1000
            mock_stream.time_base = 0.01  # 10s duration
            mock_container.streams = [mock_stream]
            mock_av_open.return_value = mock_container

            # Create mock decode frame
            mock_frame = MagicMock()
            mock_pil = MagicMock()
            mock_frame.to_image.return_value = mock_pil
            mock_container.decode.return_value = [mock_frame]

            with patch("tempfile.gettempdir", return_value=str(tmp_path)):
                with patch.object(Path, "save", create=True):
                    frames, dur, _ = sample_frames(dummy_video, max_frames=3)
                    assert dur == 10.0


# ---------------------------------------------------------------------------
# 3. Empty & invalid media handling
# ---------------------------------------------------------------------------


class TestInvalidMedia:
    def test_nonexistent_media_returns_failure(self, tmp_path):
        ghost = tmp_path / "nonexistent.mp4"
        result = process_vision(ghost)

        assert result.success is False
        assert result.failure_category == VisionFailureCategory.MEDIA_NOT_FOUND.value

    def test_unsupported_media_extension(self, tmp_path):
        txt = tmp_path / "file.txt"
        txt.write_text("hello")

        result = process_vision(txt)

        assert result.success is False
        assert result.failure_category == VisionFailureCategory.UNSUPPORTED_MEDIA.value


# ---------------------------------------------------------------------------
# 4. Frame extraction failure
# ---------------------------------------------------------------------------


class TestFrameExtractionFailure:
    def test_corrupt_video_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"not a real video header")

        result = process_vision(corrupt)

        assert result.success is False
        assert result.failure_category in (
            VisionFailureCategory.FRAME_EXTRACTION_FAILED.value,
            VisionFailureCategory.UNSUPPORTED_MEDIA.value,
        )


# ---------------------------------------------------------------------------
# 5. Model load failure
# ---------------------------------------------------------------------------


class TestModelLoadFailure:
    def test_model_load_exception_captured(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "test.jpg")

        analyzer = LocalVisionAnalyzer.__new__(LocalVisionAnalyzer)
        analyzer._model_name = "vikhyatk/moondream2"
        analyzer._device = "cpu"
        analyzer._model = None
        analyzer._model_load_seconds = None

        with patch.object(analyzer, "_ensure_model", side_effect=RuntimeError("CUDA out of memory")):
            result = analyzer.analyze_frames([img_path], media_path=img_path, input_type="image")

        assert result.success is False
        assert result.failure_category == VisionFailureCategory.MODEL_LOAD_FAILED.value
        assert "CUDA out of memory" in result.failure_message


# ---------------------------------------------------------------------------
# 6. Model inference failure
# ---------------------------------------------------------------------------


class TestModelInferenceFailure:
    def test_inference_exception_captured(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "test.jpg")

        analyzer = LocalVisionAnalyzer.__new__(LocalVisionAnalyzer)
        analyzer._model_name = "vikhyatk/moondream2"
        analyzer._device = "cpu"
        analyzer._model = MagicMock()
        analyzer._tokenizer = MagicMock()
        analyzer._model_load_seconds = 0.1

        with patch.object(analyzer, "_prompt_frame", side_effect=RuntimeError("Model crashed")):
            result = analyzer.analyze_frames([img_path], media_path=img_path, input_type="image")

        assert result.success is False
        assert result.failure_category == VisionFailureCategory.VISION_INFERENCE_FAILED.value
        assert "Model crashed" in result.failure_message


# ---------------------------------------------------------------------------
# 7. Configurable frame count
# ---------------------------------------------------------------------------


class TestConfigurableFrameCount:
    def test_max_frames_env(self, monkeypatch):
        monkeypatch.setenv("VISION_MAX_FRAMES", "4")
        from config import get_config

        cfg = get_config()
        assert cfg.vision_max_frames == 4


# ---------------------------------------------------------------------------
# 8. Temporary frame cleanup
# ---------------------------------------------------------------------------


class TestTempFrameCleanup:
    def test_temp_frames_cleaned_after_processing(self, tmp_path):
        f1 = _create_dummy_image(tmp_path / "temp_f1.jpg")
        f2 = _create_dummy_image(tmp_path / "temp_f2.jpg")

        mock_res = VisionResult(success=True, observations=["obs"])
        analyzer = MockVisionAnalyzer(mock_res)

        with (
            patch("processor.pipeline.sample_frames", return_value=([f1, f2], 5.0, 0.1)),
        ):
            vid_path = tmp_path / "fake.mp4"
            vid_path.write_bytes(b"\x00" * 100)

            result = process_vision(vid_path, analyzer=analyzer, keep_frames=False)

        assert not f1.exists()
        assert not f2.exists()


# ---------------------------------------------------------------------------
# 9. VisionResult serialization
# ---------------------------------------------------------------------------


class TestVisionResultSerialization:
    def test_as_dict_keys(self):
        res = VisionResult(
            success=True,
            media_path="video.mp4",
            input_type="video",
            frames_analyzed=3,
            subjects=["person"],
            objects=["laptop"],
            actions=["speaking"],
            scenes=["office"],
            demonstrations=["code_explanation"],
            observations=["A person coding at a laptop."],
            frame_observations=[
                FrameObservation(
                    frame_index=0,
                    timestamp_seconds=0.0,
                    description="A person coding.",
                    subjects=["person"],
                    objects=["laptop"],
                )
            ],
            model_name="moondream2",
            total_processing_seconds=1.5,
        )

        d = res.as_dict()
        text = json.dumps(d)

        assert '"success": true' in text
        assert '"input_type": "video"' in text
        assert '"frames_analyzed": 3' in text
        assert len(d["frame_observations"]) == 1
        assert d["frame_observations"][0]["frame_index"] == 0

    def test_no_ocr_or_transcript_fields_in_vision_result(self):
        res = VisionResult(success=True)
        d = res.as_dict()
        forbidden = ["ocr", "transcript", "text_read", "summary", "takeaway"]
        for key in d:
            assert key not in forbidden, f"Forbidden field '{key}' in VisionResult"


# ---------------------------------------------------------------------------
# 10. Network & API safety audit
# ---------------------------------------------------------------------------


class TestNoSecretLeakage:
    def test_no_paid_api_imports_in_vision_modules(self):
        for filename in ("vision.py", "frames.py", "pipeline.py", "models.py"):
            source = (PROJECT_ROOT / "processor" / filename).read_text(encoding="utf-8")
            for line in source.splitlines():
                if line.strip().startswith(("import ", "from ")):
                    for paid_lib in ("openai", "anthropic", "google.generativeai", "cohere"):
                        assert paid_lib not in line, f"{filename} imports {paid_lib}!"
