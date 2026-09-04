"""OCR / on-screen text abstraction & local OCR backend (Phase 7).

This module defines:
- ``BaseOCRAnalyzer``: abstract interface for OCR backends.
- ``LocalOCRAnalyzer``: local OCR backend using RapidOCR (ONNXRuntime + PP-OCRv4).
- Text normalization and conservative multi-frame deduplication utilities.

Features:
- 100% local, free inference (₹0 cost, no API keys).
- Configurable frame sampling & engine via environment variables.
- Structured outcome: :class:`processor.models.OCRResult`.
- Distinguishes "no visible text found" from "OCR failed".
"""

from __future__ import annotations

import abc
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

from processor.models import (
    OCRFailureCategory,
    OCRFrameResult,
    OCRResult,
    OCRTextBlock,
)

logger = logging.getLogger("analyzer.processor.ocr")

DEFAULT_OCR_ENGINE = "rapidocr"


def _ocr_config() -> str:
    """Read OCR engine name from environment."""
    return os.environ.get("OCR_ENGINE", DEFAULT_OCR_ENGINE).strip().lower()


def normalize_text(text: str) -> str:
    """Lightly normalize raw OCR text.

    - Strips leading and trailing whitespace.
    - Collapses consecutive internal spaces/tabs into a single space.
    - Preserves meaningful punctuation, casing, numbers, and symbols.
    """
    if not text:
        return ""
    # Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    return cleaned


def deduplicate_text_blocks(blocks: Sequence[OCRTextBlock]) -> List[OCRTextBlock]:
    """Conservatively deduplicate text blocks across video frames.

    In video/Reel playback, subtitles and on-screen overlays persist across
    multiple sampled frames. This function preserves distinct text while
    removing redundant duplicates.

    Two text blocks are considered duplicates if their normalized text
    (lowercased, whitespace-collapsed) matches an already recorded block.
    The earlier frame's detection is retained.
    """
    unique_blocks: List[OCRTextBlock] = []
    seen_normalized: set[str] = set()

    for block in blocks:
        norm = re.sub(r"\s+", " ", block.text).strip().lower()
        if not norm:
            continue
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            unique_blocks.append(block)

    return unique_blocks


class BaseOCRAnalyzer(abc.ABC):
    """Abstract OCR analyzer interface."""

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """Return human-readable identifier of the underlying OCR engine."""
        pass

    @abc.abstractmethod
    def analyze_image(
        self,
        image_or_path: Image.Image | Path | str,
        frame_index: int = 0,
        timestamp_seconds: Optional[float] = None,
    ) -> List[OCRTextBlock]:
        """Perform OCR on a single image and return detected text blocks."""
        pass

    @abc.abstractmethod
    def analyze_frames(
        self,
        frame_items: List[Tuple[Path, int, float]],
        media_type: str = "video",
        media_path: Optional[Path] = None,
        frame_extraction_seconds: Optional[float] = None,
    ) -> OCRResult:
        """Analyze a list of (frame_path, frame_index, timestamp_seconds) items."""
        pass


class LocalOCRAnalyzer(BaseOCRAnalyzer):
    """Local OCR analyzer backend using RapidOCR (Phase 7).

    RapidOCR uses ONNXRuntime with PP-OCRv4 models for fast, accurate local text
    detection and recognition. It provides bounding boxes, text strings, and
    confidence scores without external system binaries.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        engine_name: Optional[str] = None,
    ):
        self._engine = engine
        self._engine_name = engine_name or _ocr_config()
        self._model_load_seconds: Optional[float] = None

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def _ensure_engine(self) -> None:
        """Lazily initialize the OCR engine on first use."""
        if self._engine is not None:
            return

        t0 = time.perf_counter()
        logger.info("Initializing local OCR engine: %s", self._engine_name)

        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            self._model_load_seconds = time.perf_counter() - t0
            logger.info("Local OCR engine initialized in %.2fs", self._model_load_seconds)
        except ImportError as exc:
            logger.error("RapidOCR package not available: %s", exc)
            raise RuntimeError(f"OCR engine package not available: {exc}") from exc
        except Exception as exc:
            logger.error("Failed to initialize OCR engine: %s", exc)
            raise RuntimeError(f"OCR model load failed: {exc}") from exc

    def analyze_image(
        self,
        image_or_path: Image.Image | Path | str,
        frame_index: int = 0,
        timestamp_seconds: Optional[float] = None,
    ) -> List[OCRTextBlock]:
        """Perform OCR on an image and return raw OCRTextBlock instances."""
        self._ensure_engine()

        # Handle path or PIL Image input
        if isinstance(image_or_path, (str, Path)):
            input_data = str(image_or_path)
        elif isinstance(image_or_path, Image.Image):
            import numpy as np
            input_data = np.array(image_or_path)
        else:
            input_data = image_or_path

        raw_output, _elapses = self._engine(input_data)

        blocks: List[OCRTextBlock] = []
        if not raw_output:
            return blocks

        for item in raw_output:
            # RapidOCR item format: [bbox, text, score]
            if not item or len(item) < 2:
                continue

            bbox = item[0] if len(item) > 0 else None
            text = str(item[1]) if len(item) > 1 else ""
            score_raw = item[2] if len(item) > 2 else None

            confidence: Optional[float] = None
            if score_raw is not None:
                try:
                    confidence = round(float(score_raw), 4)
                except (ValueError, TypeError):
                    confidence = None

            cleaned_text = normalize_text(text)
            if cleaned_text:
                blocks.append(
                    OCRTextBlock(
                        text=cleaned_text,
                        confidence=confidence,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        bbox=bbox,
                    )
                )

        return blocks

    def analyze_frames(
        self,
        frame_items: List[Tuple[Path, int, float]],
        media_type: str = "video",
        media_path: Optional[Path] = None,
        frame_extraction_seconds: Optional[float] = None,
    ) -> OCRResult:
        """Analyze sampled frames or images and return structured OCRResult."""
        total_t0 = time.perf_counter()

        if not frame_items:
            return OCRResult(
                success=False,
                media_type=media_type,
                media_path=str(media_path) if media_path else None,
                frames_analyzed=0,
                failure_category=OCRFailureCategory.FRAME_EXTRACTION_FAILED.value,
                failure_message="No frames provided for OCR analysis.",
                processing_time_seconds=time.perf_counter() - total_t0,
            )

        # Step 1: Ensure engine is ready
        try:
            self._ensure_engine()
        except RuntimeError as exc:
            msg = str(exc)
            category = (
                OCRFailureCategory.OCR_ENGINE_NOT_AVAILABLE.value
                if "not available" in msg.lower()
                else OCRFailureCategory.OCR_MODEL_LOAD_FAILED.value
            )
            return OCRResult(
                success=False,
                media_type=media_type,
                media_path=str(media_path) if media_path else None,
                frames_analyzed=0,
                failure_category=category,
                failure_message=msg,
                processing_time_seconds=time.perf_counter() - total_t0,
            )


        # Step 2: Run OCR on each frame
        inf_t0 = time.perf_counter()
        per_frame_results: List[OCRFrameResult] = []
        all_raw_blocks: List[OCRTextBlock] = []

        try:
            for fpath, f_idx, timestamp in frame_items:
                if not Path(fpath).exists():
                    continue

                blocks = self.analyze_image(
                    fpath,
                    frame_index=f_idx,
                    timestamp_seconds=timestamp,
                )
                all_raw_blocks.extend(blocks)

                frame_text = " ".join(b.text for b in blocks)
                per_frame_results.append(
                    OCRFrameResult(
                        frame_index=f_idx,
                        timestamp_seconds=timestamp,
                        text_blocks=blocks,
                        frame_text=frame_text,
                    )
                )

            inference_time = time.perf_counter() - inf_t0

            # Step 3: Conservative multi-frame deduplication
            deduped_blocks = deduplicate_text_blocks(all_raw_blocks)

            # Construct combined text from distinct unique blocks
            combined_text = "\n".join(b.text for b in deduped_blocks)
            has_text = len(deduped_blocks) > 0

            total_time = time.perf_counter() - total_t0

            return OCRResult(
                success=True,
                media_type=media_type,
                media_path=str(media_path) if media_path else None,
                frames_analyzed=len(per_frame_results),
                text_detected=has_text,
                text_blocks=deduped_blocks,
                combined_text=combined_text,
                per_frame_results=per_frame_results,
                model_name_or_engine=self._engine_name,
                frame_extraction_seconds=frame_extraction_seconds,
                model_load_seconds=self._model_load_seconds,
                inference_seconds=inference_time,
                processing_time_seconds=total_time,
            )

        except Exception as exc:
            logger.error("OCR inference failure: %s", exc)
            return OCRResult(
                success=False,
                media_type=media_type,
                media_path=str(media_path) if media_path else None,
                frames_analyzed=len(per_frame_results),
                failure_category=OCRFailureCategory.OCR_INFERENCE_FAILED.value,
                failure_message=str(exc),
                frame_extraction_seconds=frame_extraction_seconds,
                model_load_seconds=self._model_load_seconds,
                inference_seconds=time.perf_counter() - inf_t0,
                processing_time_seconds=time.perf_counter() - total_t0,
            )
