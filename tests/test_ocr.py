"""Offline unit tests for Phase 7 OCR / On-Screen Text extraction.

These tests run completely offline and network-free.
The OCR engine/model is mocked for predictable and fast unit testing.
The real integration test on real media lives in ``scripts/test_ocr.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processor.models import (  # noqa: E402
    OCRFailureCategory,
    OCRFrameResult,
    OCRResult,
    OCRTextBlock,
)
from processor.ocr import (  # noqa: E402
    BaseOCRAnalyzer,
    LocalOCRAnalyzer,
    deduplicate_text_blocks,
    normalize_text,
)
from processor.pipeline import process_ocr  # noqa: E402


class MockOCRAnalyzer(BaseOCRAnalyzer):
    """Test double returning preconfigured OCR results."""

    def __init__(self, raw_blocks_by_frame: dict[int, list[OCRTextBlock]] | None = None):
        self._raw_blocks_by_frame = raw_blocks_by_frame or {}

    @property
    def engine_name(self) -> str:
        return "mock-ocr-engine"

    def analyze_image(
        self,
        image_or_path: Image.Image | Path | str,
        frame_index: int = 0,
        timestamp_seconds: float | None = None,
    ) -> list[OCRTextBlock]:
        return self._raw_blocks_by_frame.get(frame_index, [])

    def analyze_frames(
        self,
        frame_items: list[tuple[Path, int, float]],
        media_type: str = "video",
        media_path: Path | None = None,
        frame_extraction_seconds: float | None = None,
    ) -> OCRResult:
        all_blocks: list[OCRTextBlock] = []
        per_frame_results: list[OCRFrameResult] = []

        for fpath, idx, ts in frame_items:
            blocks = self.analyze_image(fpath, frame_index=idx, timestamp_seconds=ts)
            all_blocks.extend(blocks)
            per_frame_results.append(
                OCRFrameResult(
                    frame_index=idx,
                    timestamp_seconds=ts,
                    text_blocks=blocks,
                    frame_text=" ".join(b.text for b in blocks),
                )
            )

        deduped = deduplicate_text_blocks(all_blocks)
        has_text = len(deduped) > 0

        return OCRResult(
            success=True,
            media_type=media_type,
            media_path=str(media_path) if media_path else None,
            frames_analyzed=len(frame_items),
            text_detected=has_text,
            text_blocks=deduped,
            combined_text="\n".join(b.text for b in deduped),
            per_frame_results=per_frame_results,
            model_name_or_engine=self.engine_name,
            frame_extraction_seconds=frame_extraction_seconds,
            processing_time_seconds=0.05,
        )


def _create_dummy_image(path: Path) -> Path:
    """Create a minimal 10x10 dummy JPEG image."""
    img = Image.new("RGB", (10, 10), color="white")
    img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# 1. Single-image OCR
# ---------------------------------------------------------------------------


class TestSingleImageOCR:
    def test_single_image_ocr_success(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "test.jpg")

        mock_block = OCRTextBlock(
            text="Hello World",
            confidence=0.95,
            frame_index=0,
            timestamp_seconds=0.0,
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        analyzer = MockOCRAnalyzer({0: [mock_block]})

        result = process_ocr(img_path, analyzer=analyzer)

        assert result.success is True
        assert result.media_type == "image"
        assert result.frames_analyzed == 1
        assert result.text_detected is True
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].text == "Hello World"
        assert result.text_blocks[0].confidence == 0.95
        assert result.combined_text == "Hello World"


# ---------------------------------------------------------------------------
# 2. Video frame OCR
# ---------------------------------------------------------------------------


class TestVideoFrameOCR:
    def test_video_ocr_sampling_and_timestamps(self, tmp_path):
        f0 = _create_dummy_image(tmp_path / "f0.jpg")
        f1 = _create_dummy_image(tmp_path / "f1.jpg")

        blocks = {
            0: [OCRTextBlock(text="Introduction to AI", confidence=0.98, frame_index=0, timestamp_seconds=0.0)],
            1: [OCRTextBlock(text="Chapter 1: Neural Nets", confidence=0.91, frame_index=1, timestamp_seconds=5.0)],
        }
        analyzer = MockOCRAnalyzer(blocks)

        fake_video = tmp_path / "reel.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        with patch("processor.pipeline.sample_frames", return_value=([f0, f1], 10.0, 0.05)):
            result = process_ocr(fake_video, analyzer=analyzer)

        assert result.success is True
        assert result.media_type == "video"
        assert result.frames_analyzed == 2
        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].timestamp_seconds == 0.0
        assert result.text_blocks[1].timestamp_seconds == 5.0
        assert "Introduction to AI" in result.combined_text
        assert "Chapter 1: Neural Nets" in result.combined_text


# ---------------------------------------------------------------------------
# 3. Carousel OCR
# ---------------------------------------------------------------------------


class TestCarouselOCR:
    def test_carousel_multiple_images(self, tmp_path):
        slide1 = _create_dummy_image(tmp_path / "slide1.jpg")
        slide2 = _create_dummy_image(tmp_path / "slide2.jpg")
        slide3 = _create_dummy_image(tmp_path / "slide3.jpg")

        blocks = {
            0: [OCRTextBlock(text="Slide 1: Problem", confidence=0.99, frame_index=0)],
            1: [OCRTextBlock(text="Slide 2: Solution", confidence=0.97, frame_index=1)],
            2: [OCRTextBlock(text="Slide 3: Impact", confidence=0.96, frame_index=2)],
        }
        analyzer = MockOCRAnalyzer(blocks)

        result = process_ocr([slide1, slide2, slide3], analyzer=analyzer)

        assert result.success is True
        assert result.media_type == "carousel"
        assert result.frames_analyzed == 3
        assert len(result.text_blocks) == 3
        assert [b.text for b in result.text_blocks] == [
            "Slide 1: Problem",
            "Slide 2: Solution",
            "Slide 3: Impact",
        ]


# ---------------------------------------------------------------------------
# 4. No text detected
# ---------------------------------------------------------------------------


class TestNoTextDetected:
    def test_clean_empty_text_result(self, tmp_path):
        img_path = _create_dummy_image(tmp_path / "blank.jpg")
        analyzer = MockOCRAnalyzer({})  # no text blocks detected

        result = process_ocr(img_path, analyzer=analyzer)

        assert result.success is True
        assert result.text_detected is False
        assert result.text_blocks == []
        assert result.combined_text == ""
        assert result.failure_category is None


# ---------------------------------------------------------------------------
# 5. Invalid media handling
# ---------------------------------------------------------------------------


class TestInvalidMediaHandling:
    def test_nonexistent_file(self, tmp_path):
        ghost = tmp_path / "missing.mp4"
        result = process_ocr(ghost)

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.INVALID_MEDIA.value
        assert "does not exist" in result.failure_message

    def test_empty_carousel_list(self):
        result = process_ocr([])

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.INVALID_MEDIA.value

    def test_carousel_with_missing_item(self, tmp_path):
        img1 = _create_dummy_image(tmp_path / "img1.jpg")
        img2 = tmp_path / "img2.jpg"  # not created

        result = process_ocr([img1, img2])

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.INVALID_MEDIA.value

    def test_unsupported_media_extension(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("plain text")

        result = process_ocr(txt_file)

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.UNSUPPORTED_MEDIA.value


# ---------------------------------------------------------------------------
# 6. OCR Engine unavailable
# ---------------------------------------------------------------------------


class TestOCREngineUnavailable:
    def test_engine_package_not_installed(self, tmp_path):
        img = _create_dummy_image(tmp_path / "frame.jpg")
        analyzer = LocalOCRAnalyzer(engine=None)

        with patch.object(
            analyzer,
            "_ensure_engine",
            side_effect=RuntimeError("OCR engine package not available: No module named 'rapidocr_onnxruntime'"),
        ):
            result = analyzer.analyze_frames([(img, 0, 0.0)], media_type="image")

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.OCR_ENGINE_NOT_AVAILABLE.value


# ---------------------------------------------------------------------------
# 7. Model load failure
# ---------------------------------------------------------------------------


class TestModelLoadFailure:
    def test_engine_load_exception(self, tmp_path):
        img = _create_dummy_image(tmp_path / "frame.jpg")
        analyzer = LocalOCRAnalyzer(engine=None)

        with patch.object(
            analyzer,
            "_ensure_engine",
            side_effect=RuntimeError("OCR model load failed: ONNX model corrupted"),
        ):
            result = analyzer.analyze_frames([(img, 0, 0.0)], media_type="image")

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.OCR_MODEL_LOAD_FAILED.value
        assert "ONNX model corrupted" in result.failure_message


# ---------------------------------------------------------------------------
# 8. Inference failure
# ---------------------------------------------------------------------------


class TestInferenceFailure:
    def test_inference_exception(self, tmp_path):
        img = _create_dummy_image(tmp_path / "frame.jpg")
        analyzer = LocalOCRAnalyzer(engine=MagicMock())

        with patch.object(analyzer, "analyze_image", side_effect=RuntimeError("Inference crash")):
            result = analyzer.analyze_frames([(img, 0, 0.0)], media_type="image")

        assert result.success is False
        assert result.failure_category == OCRFailureCategory.OCR_INFERENCE_FAILED.value
        assert "Inference crash" in result.failure_message


# ---------------------------------------------------------------------------
# 9. Multi-frame deduplication
# ---------------------------------------------------------------------------


class TestMultiFrameDeduplication:
    def test_duplicate_text_across_frames_deduplicated(self):
        blocks = [
            OCRTextBlock(text="Machine Learning", confidence=0.95, frame_index=0, timestamp_seconds=0.0),
            OCRTextBlock(text="Machine Learning", confidence=0.96, frame_index=1, timestamp_seconds=2.5),
            OCRTextBlock(text="Machine Learning", confidence=0.94, frame_index=2, timestamp_seconds=5.0),
            OCRTextBlock(text="Serving Cost 4x", confidence=0.92, frame_index=3, timestamp_seconds=7.5),
        ]

        deduped = deduplicate_text_blocks(blocks)

        assert len(deduped) == 2
        assert deduped[0].text == "Machine Learning"
        assert deduped[0].frame_index == 0  # earliest frame preserved
        assert deduped[1].text == "Serving Cost 4x"

    def test_distinct_blocks_in_same_frame_preserved(self):
        blocks = [
            OCRTextBlock(text="Headline Title", confidence=0.98, frame_index=0),
            OCRTextBlock(text="Subtitle Details", confidence=0.92, frame_index=0),
        ]
        deduped = deduplicate_text_blocks(blocks)
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# 10. Confidence handling
# ---------------------------------------------------------------------------


class TestConfidenceHandling:
    def test_confidence_preserved_and_not_fabricated(self):
        block_with_conf = OCRTextBlock(text="With conf", confidence=0.8876)
        assert block_with_conf.confidence == 0.8876

        block_without_conf = OCRTextBlock(text="No conf", confidence=None)
        assert block_without_conf.confidence is None

    def test_local_ocr_confidence_parsing(self):
        mock_engine = MagicMock()
        # RapidOCR returns [bbox, text, score]
        mock_engine.return_value = (
            [
                [[[0, 0], [10, 0], [10, 10], [0, 10]], "Valid Score", "0.9412"],
                [[[0, 0], [10, 0], [10, 10], [0, 10]], "Invalid Score", "not_a_number"],
            ],
            [0.1, 0.01, 0.05],
        )

        analyzer = LocalOCRAnalyzer(engine=mock_engine)
        blocks = analyzer.analyze_image("dummy_path")

        assert len(blocks) == 2
        assert blocks[0].confidence == 0.9412
        assert blocks[1].confidence is None  # no fabricated number


# ---------------------------------------------------------------------------
# 11. Frame and timestamp preservation
# ---------------------------------------------------------------------------


class TestFrameTimestampPreservation:
    def test_frame_index_and_timestamp_preserved(self):
        block = OCRTextBlock(
            text="Timestamped",
            confidence=0.9,
            frame_index=4,
            timestamp_seconds=12.5,
            bbox=[[1, 2], [3, 4]],
        )

        d = block.as_dict()
        assert d["frame_index"] == 4
        assert d["timestamp_seconds"] == 12.5
        assert d["bbox"] == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# 12. Temporary frame cleanup
# ---------------------------------------------------------------------------


class TestTempFrameCleanup:
    def test_video_temp_frames_cleaned_after_processing(self, tmp_path):
        f0 = _create_dummy_image(tmp_path / "ocr_temp_0.jpg")
        f1 = _create_dummy_image(tmp_path / "ocr_temp_1.jpg")

        analyzer = MockOCRAnalyzer({})

        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        with patch("processor.pipeline.sample_frames", return_value=([f0, f1], 5.0, 0.01)):
            result = process_ocr(fake_video, analyzer=analyzer, keep_frames=False)

        assert result.success is True
        assert not f0.exists()
        assert not f1.exists()


# ---------------------------------------------------------------------------
# 13. Structured OCRResult serialization
# ---------------------------------------------------------------------------


class TestOCRResultSerialization:
    def test_serialization_fields(self):
        result = OCRResult(
            success=True,
            media_type="video",
            media_path="/path/to/video.mp4",
            frames_analyzed=2,
            text_detected=True,
            text_blocks=[
                OCRTextBlock(
                    text="Sample Text",
                    confidence=0.95,
                    frame_index=0,
                    timestamp_seconds=1.5,
                    bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
                )
            ],
            combined_text="Sample Text",
            per_frame_results=[
                OCRFrameResult(
                    frame_index=0,
                    timestamp_seconds=1.5,
                    text_blocks=[
                        OCRTextBlock(text="Sample Text", confidence=0.95, frame_index=0, timestamp_seconds=1.5)
                    ],
                    frame_text="Sample Text",
                )
            ],
            model_name_or_engine="rapidocr",
            processing_time_seconds=0.42,
        )

        data = result.as_dict()
        serialized = json.dumps(data)

        assert '"success": true' in serialized
        assert '"media_type": "video"' in serialized
        assert '"frames_analyzed": 2' in serialized
        assert '"text_detected": true' in serialized
        assert '"Sample Text"' in serialized
        assert '"processing_time_seconds": 0.42' in serialized

    def test_no_forbidden_summary_or_transcript_fields(self):
        result = OCRResult(success=True)
        data = result.as_dict()

        forbidden_fields = [
            "summary",
            "core_takeaway",
            "transcript",
            "speech",
            "multimodal_summary",
            "llm_output",
        ]
        for field in forbidden_fields:
            assert field not in data, f"Forbidden field '{field}' found in OCRResult!"


# ---------------------------------------------------------------------------
# 14. Network & API safety audit
# ---------------------------------------------------------------------------


class TestNoSecretLeakage:
    def test_no_paid_api_imports_in_ocr_modules(self):
        for filename in ("ocr.py", "pipeline.py", "models.py"):
            source = (PROJECT_ROOT / "processor" / filename).read_text(encoding="utf-8")
            for line in source.splitlines():
                if line.strip().startswith(("import ", "from ")):
                    for paid_lib in ("openai", "anthropic", "google.generativeai", "cohere"):
                        assert paid_lib not in line, f"{filename} imports paid library {paid_lib}!"


# ---------------------------------------------------------------------------
# 15. Text normalization
# ---------------------------------------------------------------------------


class TestTextNormalization:
    def test_whitespace_collapsing(self):
        raw = "  Line   with    irregular   spaces.  "
        assert normalize_text(raw) == "Line with irregular spaces."

    def test_empty_string(self):
        assert normalize_text("") == ""
        assert normalize_text("   \t  ") == ""
