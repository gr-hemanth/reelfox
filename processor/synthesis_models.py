"""Data models for multimodal synthesis (Phase 8).

Defines:
- :class:`SynthesisFailureCategory`: machine-readable failure taxonomy.
- :class:`MultimodalEvidence`: structured input evidence container.
- :class:`MultimodalAnalysisResult`: structured multimodal interpretation output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SynthesisFailureCategory(str, Enum):
    """Machine-readable failure categories for multimodal synthesis."""

    API_KEY_MISSING = "API_KEY_MISSING"
    API_AUTHENTICATION_FAILED = "API_AUTHENTICATION_FAILED"
    API_REQUEST_FAILED = "API_REQUEST_FAILED"
    API_TIMEOUT = "API_TIMEOUT"
    API_RATE_LIMITED = "API_RATE_LIMITED"
    INVALID_MODEL_RESPONSE = "INVALID_MODEL_RESPONSE"
    JSON_PARSE_FAILED = "JSON_PARSE_FAILED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
    SYNTHESIS_DISABLED = "SYNTHESIS_DISABLED"
    UNKNOWN = "UNKNOWN"


def _truncate_text(text: Optional[str], max_chars: int) -> Optional[str]:
    """Truncate text cleanly with a visible notice if exceeding *max_chars*."""
    if not text:
        return text
    text_clean = text.strip()
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars] + "\n[... truncated for input limit ...]"


@dataclass
class MultimodalEvidence:
    """Structured evidence aggregated across extraction, speech, vision, and OCR.

    The synthesis model receives ONLY this structured textual evidence.
    Raw image and video bytes are NEVER passed to the synthesis model.
    """

    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    ocr: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "metadata": self.metadata,
            "speech": self.speech,
            "vision": self.vision,
            "ocr": self.ocr,
        }

    @classmethod
    def from_results(
        cls,
        extraction_result: Any,
        speech_result: Optional[Any] = None,
        vision_result: Optional[Any] = None,
        ocr_result: Optional[Any] = None,
        max_caption_chars: int = 1000,
        max_transcript_chars: int = 4000,
        max_ocr_chars: int = 4000,
        max_vision_observations: int = 12,
    ) -> "MultimodalEvidence":
        """Build a clean MultimodalEvidence container from pipeline stage results.

        Explicitly marks unavailable evidence rather than inventing missing data.
        Enforces conservative input size bounds.
        """
        source_url = getattr(extraction_result, "source_url", "") or ""

        # 1. Metadata / Caption
        caption_raw = getattr(extraction_result, "caption", None)
        caption_truncated = _truncate_text(caption_raw, max_caption_chars)
        metadata = {
            "media_type": getattr(extraction_result, "media_type", "unknown"),
            "caption": caption_truncated,
            "caption_present": bool(caption_truncated),
            "hashtags": list(getattr(extraction_result, "hashtags", []) or []),
        }

        # 2. Speech / ASR
        if speech_result is not None and getattr(speech_result, "success", False):
            transcript_raw = getattr(speech_result, "transcript", None)
            speech_info = {
                "available": True,
                "speech_present": getattr(speech_result, "speech_present", False),
                "classification": getattr(speech_result, "classification", "no_audio"),
                "detected_language": getattr(speech_result, "detected_language", None),
                "transcript": _truncate_text(transcript_raw, max_transcript_chars),
            }
        else:
            reason = "not available"
            if speech_result is not None and not getattr(speech_result, "success", False):
                reason = f"failed: {getattr(speech_result, 'failure_category', 'UNKNOWN')}"
            speech_info = {
                "available": False,
                "reason": reason,
            }

        # 3. Vision
        if vision_result is not None and getattr(vision_result, "success", False):
            obs_raw = list(getattr(vision_result, "observations", []) or [])
            if len(obs_raw) > max_vision_observations:
                obs_bounded = obs_raw[:max_vision_observations] + [
                    f"[... {len(obs_raw) - max_vision_observations} more frame observations truncated ...]"
                ]
            else:
                obs_bounded = obs_raw

            vision_info = {
                "available": True,
                "frames_analyzed": getattr(vision_result, "frames_analyzed", 0),
                "subjects": list(getattr(vision_result, "subjects", []) or []),
                "objects": list(getattr(vision_result, "objects", []) or []),
                "actions": list(getattr(vision_result, "actions", []) or []),
                "scenes": list(getattr(vision_result, "scenes", []) or []),
                "demonstrations": list(getattr(vision_result, "demonstrations", []) or []),
                "observations": obs_bounded,
            }
        else:
            reason = "not available"
            if vision_result is not None and not getattr(vision_result, "success", False):
                reason = f"failed: {getattr(vision_result, 'failure_category', 'UNKNOWN')}"
            vision_info = {
                "available": False,
                "reason": reason,
            }

        # 4. OCR
        if ocr_result is not None and getattr(ocr_result, "success", False):
            ocr_text_raw = getattr(ocr_result, "combined_text", "") or ""
            text_blocks_raw = getattr(ocr_result, "text_blocks", []) or []
            ocr_info = {
                "available": True,
                "text_detected": getattr(ocr_result, "text_detected", False),
                "frames_analyzed": getattr(ocr_result, "frames_analyzed", 0),
                "combined_text": _truncate_text(ocr_text_raw, max_ocr_chars),
            }
        else:
            reason = "not available"
            if ocr_result is not None and not getattr(ocr_result, "success", False):
                reason = f"failed: {getattr(ocr_result, 'failure_category', 'UNKNOWN')}"
            ocr_info = {
                "available": False,
                "reason": reason,
            }

        return cls(
            source_url=source_url,
            metadata=metadata,
            speech=speech_info,
            vision=vision_info,
            ocr=ocr_info,
        )


@dataclass
class MultimodalAnalysisResult:
    """The structured outcome of Phase 8 Multimodal Synthesis."""

    success: bool
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    core_takeaway: str = ""
    relevant_context: str = ""
    confidence: float = 0.0
    evidence_used: dict[str, bool] = field(
        default_factory=lambda: {
            "caption": False,
            "speech": False,
            "vision": False,
            "ocr": False,
        }
    )
    model_name: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    request_latency_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    raw_response: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary of the multimodal result."""
        return {
            "success": self.success,
            "summary": self.summary,
            "key_points": list(self.key_points),
            "core_takeaway": self.core_takeaway,
            "relevant_context": self.relevant_context,
            "confidence": self.confidence,
            "evidence_used": dict(self.evidence_used),
            "model_name": self.model_name,
            "processing_time_seconds": self.processing_time_seconds,
            "request_latency_seconds": self.request_latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
        }

    def summary_line(self) -> str:
        """A single compact line for logs."""
        if self.success:
            pts = len(self.key_points)
            return (
                f"OK model={self.model_name} conf={self.confidence:.2f} "
                f"key_points={pts} tokens={self.total_tokens} "
                f"t={self.processing_time_seconds}s"
            )
        return f"FAIL category={self.failure_category} message={self.failure_message}"
