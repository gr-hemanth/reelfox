from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("analyzer.processor.run_result")

# Standard model identifiers used in baseline
DEFAULT_MODEL_VERSIONS = {
    "asr": "faster-whisper base (CPU int8)",
    "vision": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "ocr": "RapidOCR PP-OCRv4 ONNX",
    "synthesis": "Qwen/Qwen2.5-3B-Instruct (CPU bfloat16)",
}


def _iso_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunResult:
    """Structured, run-level container for a complete end-to-end pipeline execution.

    Preserves individual stage results and measurements without flattening them away.
    Never contains raw media, passwords, session tokens, or API credentials.
    """

    run_id: str
    source_url: str
    started_at: str
    completed_at: str
    overall_success: bool
    content_type: str = "unknown"
    extraction_mode: str = "public"

    # High-level failure tracking
    failure_stage: Optional[str] = None  # e.g., "url_validation", "extraction", "speech", "vision", "ocr", "synthesis", "cleanup"
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None

    # Model versions
    model_versions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_VERSIONS))

    # Stage metrics
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    total_execution_seconds: float = 0.0

    # Cleanup tracking
    cleanup_attempted: bool = False
    cleanup_success: bool = True
    leftover_artifact_count: int = 0

    # Stage-level evidence & results (JSON-serializable dicts)
    validation: Optional[Dict[str, Any]] = None
    extraction: Optional[Dict[str, Any]] = None
    speech: Optional[Dict[str, Any]] = None
    vision: Optional[Dict[str, Any]] = None
    ocr: Optional[Dict[str, Any]] = None
    synthesis: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a clean, serializable dictionary without secrets or raw media."""
        return {
            "run_id": self.run_id,
            "source_url": self.source_url,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "overall_success": self.overall_success,
            "content_type": self.content_type,
            "extraction_mode": self.extraction_mode,
            "failure_stage": self.failure_stage,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "model_versions": dict(self.model_versions),
            "stage_metrics": dict(self.stage_metrics),
            "total_execution_seconds": round(self.total_execution_seconds, 3),
            "cleanup": {
                "cleanup_attempted": self.cleanup_attempted,
                "cleanup_success": self.cleanup_success,
                "leftover_artifact_count": self.leftover_artifact_count,
            },
            "stages": {
                "validation": self.validation,
                "extraction": self.extraction,
                "speech": self.speech,
                "vision": self.vision,
                "ocr": self.ocr,
                "synthesis": self.synthesis,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize run result to JSON string."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


def build_stage_metrics(
    validation_res: Any = None,
    extraction_res: Any = None,
    speech_res: Any = None,
    vision_res: Any = None,
    ocr_res: Any = None,
    synthesis_res: Any = None,
    cleanup_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile stage-level metrics into a structured dictionary."""
    metrics: Dict[str, Any] = {}

    # 1. Extraction Metrics
    if extraction_res is not None:
        metrics["extraction"] = {
            "success": bool(getattr(extraction_res, "success", False)),
            "media_downloaded": bool(getattr(extraction_res, "media_downloaded", False)),
            "caption_extracted": bool(getattr(extraction_res, "caption_extracted", False)),
            "media_type": getattr(extraction_res, "media_type", "unknown"),
            "download_seconds": getattr(extraction_res, "download_seconds", None),
            "failure_category": getattr(extraction_res, "failure_category", None),
        }
    else:
        metrics["extraction"] = {"success": False, "failure_category": "NOT_ATTEMPTED"}

    # 2. Speech Metrics
    if speech_res is not None:
        metrics["speech"] = {
            "success": bool(getattr(speech_res, "success", False)),
            "audio_present": bool(getattr(speech_res, "audio_present", False)),
            "speech_present": bool(getattr(speech_res, "speech_present", False)),
            "classification": getattr(speech_res, "classification", None),
            "detected_language": getattr(speech_res, "detected_language", None),
            "transcription_time_seconds": getattr(speech_res, "transcription_time_seconds", None),
            "transcript_available": bool(getattr(speech_res, "transcript", None)),
            "failure_category": getattr(speech_res, "failure_category", None),
        }
    else:
        metrics["speech"] = {
            "success": False,
            "audio_present": False,
            "speech_present": False,
            "transcript_available": False,
            "failure_category": None,
        }

    # 3. Vision Metrics
    if vision_res is not None:
        metrics["vision"] = {
            "success": bool(getattr(vision_res, "success", False)),
            "frames_analyzed": getattr(vision_res, "frames_analyzed", 0),
            "model_name": getattr(vision_res, "model_name", "SmolVLM-256M-Instruct"),
            "frame_extraction_seconds": getattr(vision_res, "frame_extraction_seconds", None),
            "model_load_seconds": getattr(vision_res, "model_load_seconds", None),
            "inference_seconds": getattr(vision_res, "inference_seconds", None),
            "failure_category": getattr(vision_res, "failure_category", None),
        }
    else:
        metrics["vision"] = {
            "success": False,
            "frames_analyzed": 0,
            "failure_category": None,
        }

    # 4. OCR Metrics
    if ocr_res is not None:
        text_blocks = getattr(ocr_res, "text_blocks", []) or []
        metrics["ocr"] = {
            "success": bool(getattr(ocr_res, "success", False)),
            "frames_analyzed": getattr(ocr_res, "frames_analyzed", 0),
            "text_detected": bool(getattr(ocr_res, "text_detected", False)),
            "text_block_count": len(text_blocks),
            "engine_name": getattr(ocr_res, "model_name_or_engine", "rapidocr"),
            "frame_extraction_seconds": getattr(ocr_res, "frame_extraction_seconds", None),
            "engine_load_seconds": getattr(ocr_res, "model_load_seconds", None),
            "inference_seconds": getattr(ocr_res, "inference_seconds", None),
            "failure_category": getattr(ocr_res, "failure_category", None),
        }
    else:
        metrics["ocr"] = {
            "success": False,
            "frames_analyzed": 0,
            "text_detected": False,
            "text_block_count": 0,
            "failure_category": None,
        }

    # 5. Synthesis Metrics
    if synthesis_res is not None:
        metrics["synthesis"] = {
            "success": bool(getattr(synthesis_res, "success", False)),
            "model_name": getattr(synthesis_res, "model_name", "Qwen/Qwen2.5-3B-Instruct"),
            "request_latency_seconds": getattr(synthesis_res, "request_latency_seconds", None),
            "processing_time_seconds": getattr(synthesis_res, "processing_time_seconds", None),
            "prompt_tokens": getattr(synthesis_res, "prompt_tokens", None),
            "completion_tokens": getattr(synthesis_res, "completion_tokens", None),
            "total_tokens": getattr(synthesis_res, "total_tokens", None),
            "confidence": getattr(synthesis_res, "confidence", None),
            "failure_category": getattr(synthesis_res, "failure_category", None),
        }
    else:
        metrics["synthesis"] = {
            "success": False,
            "failure_category": None,
        }

    # 6. Cleanup Metrics
    if cleanup_res is not None:
        metrics["cleanup"] = cleanup_res
    else:
        metrics["cleanup"] = {
            "cleanup_attempted": False,
            "cleanup_success": True,
            "leftover_artifact_count": 0,
        }

    return metrics


def run_pipeline(
    url: str,
    *,
    config: Any = None,
    run_id: Optional[str] = None,
    keep_media: bool = False,
    skip_speech: bool = False,
    skip_vision: bool = False,
    skip_ocr: bool = False,
    skip_synthesis: bool = False,
    extractor: Any = None,
    speech_runner: Any = None,
    vision_runner: Any = None,
    ocr_runner: Any = None,
    synthesis_runner: Any = None,
) -> RunResult:
    """Run the complete end-to-end Instagram analysis pipeline on *url*.

    Accepts optional dependency runners/mocks to allow 100% deterministic offline testing.
    Never raises uncaught exceptions — all errors are captured in the RunResult.
    """
    import config as app_config
    from extractor.instagram_extractor import ExtractionOptions, YtDlpExtractor
    from extractor.models import ExtractionMode
    from extractor.url_validator import validate_instagram_url

    t_start = time.perf_counter()
    started_at = _iso_now()
    active_run_id = run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    settings = config or app_config.Config.load()
    extraction_mode_str = getattr(settings, "extraction_mode", "public")

    # Step 1: URL Validation
    validation = validate_instagram_url(url)
    val_dict = validation.as_dict()

    if not validation.valid:
        t_end = time.perf_counter()
        return RunResult(
            run_id=active_run_id,
            source_url=url,
            started_at=started_at,
            completed_at=_iso_now(),
            overall_success=False,
            content_type=validation.content_type_hint or "unknown",
            extraction_mode=extraction_mode_str,
            failure_stage="url_validation",
            failure_category=validation.error_code,
            failure_message=f"URL validation failed: {validation.error_code}",
            stage_metrics={"validation": val_dict},
            total_execution_seconds=t_end - t_start,
            validation=val_dict,
        )

    # Step 2: Extraction
    if extractor is None:
        extractor = YtDlpExtractor(
            temp_dir=settings.temp_dir,
            output_dir=settings.output_dir,
            cookies_from_browser=getattr(settings, "cookies_from_browser", "") or None,
            cookie_file=getattr(settings, "cookie_file", "") or None,
        )

    ext_options = ExtractionOptions(
        mode=ExtractionMode.from_string(extraction_mode_str),
        cookies_from_browser=getattr(settings, "cookies_from_browser", "") or None,
        cookie_file=getattr(settings, "cookie_file", "") or None,
        keep_media=True,  # we manage run cleanup at end of run_pipeline
    )

    try:
        extraction_result = extractor.extract(validation, ext_options)
    except Exception as exc:
        logger.error("Extraction crashed unexpectedly: %s", exc)
        from extractor.models import ExtractionResult
        extraction_result = ExtractionResult(
            success=False,
            source_url=url,
            failure_category="UNEXPECTED_EXTRACTION_ERROR",
            failure_reason=str(exc),
        )

    ext_dict = extraction_result.as_dict() if extraction_result else None
    content_type = extraction_result.media_type if extraction_result else (validation.content_type_hint or "unknown")

    if not extraction_result or not extraction_result.success:
        t_end = time.perf_counter()
        # Attempt cleanup if run_id was created
        cleanup_success, leftover_count = _execute_cleanup(settings, extraction_result, keep_media)
        stage_metrics = build_stage_metrics(
            validation_res=validation,
            extraction_res=extraction_result,
            cleanup_res={"cleanup_attempted": True, "cleanup_success": cleanup_success, "leftover_artifact_count": leftover_count},
        )
        return RunResult(
            run_id=active_run_id,
            source_url=url,
            started_at=started_at,
            completed_at=_iso_now(),
            overall_success=False,
            content_type=content_type,
            extraction_mode=extraction_mode_str,
            failure_stage="extraction",
            failure_category=extraction_result.failure_category if extraction_result else "EXTRACTION_FAILED",
            failure_message=extraction_result.failure_reason if extraction_result else "Extraction failed to return usable media",
            stage_metrics=stage_metrics,
            total_execution_seconds=t_end - t_start,
            cleanup_attempted=True,
            cleanup_success=cleanup_success,
            leftover_artifact_count=leftover_count,
            validation=val_dict,
            extraction=ext_dict,
        )

    # Step 3: Speech Processing
    speech_result = None
    media_path = extraction_result.media_path
    has_audio = extraction_result.media_type in ("reel", "video")

    if has_audio and media_path and not skip_speech:
        if speech_runner is not None:
            speech_result = speech_runner(media_path)
        else:
            from processor.pipeline import process_speech
            try:
                speech_result = process_speech(media_path)
            except Exception as exc:
                logger.error("Speech processing error: %s", exc)
                from processor.models import SpeechFailureCategory, SpeechResult
                speech_result = SpeechResult(
                    success=False,
                    failure_category=SpeechFailureCategory.UNKNOWN.value,
                    failure_message=str(exc),
                )

    speech_dict = speech_result.as_dict() if speech_result else None

    # Step 4: Vision Processing
    vision_result = None
    if media_path and not skip_vision:
        if vision_runner is not None:
            vision_result = vision_runner(media_path)
        else:
            from processor.pipeline import process_vision
            try:
                vision_result = process_vision(media_path, is_video=(extraction_result.media_type in ("reel", "video")))
            except Exception as exc:
                logger.error("Vision processing error: %s", exc)
                from processor.models import VisionFailureCategory, VisionResult
                vision_result = VisionResult(
                    success=False,
                    failure_category=VisionFailureCategory.UNKNOWN.value,
                    failure_message=str(exc),
                )

    vision_dict = vision_result.as_dict() if vision_result else None

    # Step 5: OCR Processing
    ocr_result = None
    if not skip_ocr:
        if extraction_result.media_type == "carousel" and len(extraction_result.media_files) > 1:
            media_input = [m.path for m in extraction_result.media_files]
        else:
            media_input = media_path

        if media_input:
            if ocr_runner is not None:
                ocr_result = ocr_runner(media_input)
            else:
                from processor.pipeline import process_ocr
                try:
                    ocr_result = process_ocr(media_input, is_video=(extraction_result.media_type in ("reel", "video")))
                except Exception as exc:
                    logger.error("OCR processing error: %s", exc)
                    from processor.models import OCRFailureCategory, OCRResult
                    ocr_result = OCRResult(
                        success=False,
                        failure_category=OCRFailureCategory.UNKNOWN.value,
                        failure_message=str(exc),
                    )

    ocr_dict = ocr_result.as_dict() if ocr_result else None

    # Step 6: Multimodal Synthesis
    synthesis_result = None
    if not skip_synthesis:
        if synthesis_runner is not None:
            synthesis_result = synthesis_runner(
                extraction=extraction_result,
                speech=speech_result,
                vision=vision_result,
                ocr=ocr_result,
            )
        else:
            from processor.pipeline import process_synthesis
            try:
                synthesis_result = process_synthesis(
                    extraction=extraction_result,
                    speech=speech_result,
                    vision=vision_result,
                    ocr=ocr_result,
                )
            except Exception as exc:
                logger.error("Synthesis processing error: %s", exc)
                from processor.synthesis_models import MultimodalAnalysisResult, SynthesisFailureCategory
                synthesis_result = MultimodalAnalysisResult(
                    success=False,
                    failure_category=SynthesisFailureCategory.UNKNOWN.value,
                    failure_message=str(exc),
                )

    synthesis_dict = synthesis_result.as_dict() if synthesis_result else None

    # Step 7: Cleanup
    cleanup_success, leftover_count = _execute_cleanup(settings, extraction_result, keep_media)

    # Determine overall success & failure tracking
    overall_success = True
    failure_stage = None
    failure_category = None
    failure_message = None

    if synthesis_result is not None and not synthesis_result.success:
        overall_success = False
        failure_stage = "synthesis"
        failure_category = synthesis_result.failure_category
        failure_message = synthesis_result.failure_message
    elif ocr_result is not None and not ocr_result.success and ocr_result.failure_category not in (None, "NO_TEXT_DETECTED"):
        # Note: NO_TEXT_DETECTED is a legitimate outcome, not an engine failure
        overall_success = False
        failure_stage = "ocr"
        failure_category = ocr_result.failure_category
        failure_message = ocr_result.failure_message
    elif vision_result is not None and not vision_result.success:
        overall_success = False
        failure_stage = "vision"
        failure_category = vision_result.failure_category
        failure_message = vision_result.failure_message
    elif speech_result is not None and not speech_result.success and speech_result.failure_category not in (None, "NO_SPEECH_DETECTED"):
        overall_success = False
        failure_stage = "speech"
        failure_category = speech_result.failure_category
        failure_message = speech_result.failure_message

    t_end = time.perf_counter()
    stage_metrics = build_stage_metrics(
        validation_res=validation,
        extraction_res=extraction_result,
        speech_res=speech_result,
        vision_res=vision_result,
        ocr_res=ocr_result,
        synthesis_res=synthesis_result,
        cleanup_res={
            "cleanup_attempted": not keep_media,
            "cleanup_success": cleanup_success,
            "leftover_artifact_count": leftover_count,
        },
    )

    return RunResult(
        run_id=active_run_id,
        source_url=url,
        started_at=started_at,
        completed_at=_iso_now(),
        overall_success=overall_success,
        content_type=content_type,
        extraction_mode=extraction_mode_str,
        failure_stage=failure_stage,
        failure_category=failure_category,
        failure_message=failure_message,
        stage_metrics=stage_metrics,
        total_execution_seconds=t_end - t_start,
        cleanup_attempted=not keep_media,
        cleanup_success=cleanup_success,
        leftover_artifact_count=leftover_count,
        validation=val_dict,
        extraction=ext_dict,
        speech=speech_dict,
        vision=vision_dict,
        ocr=ocr_dict,
        synthesis=synthesis_dict,
    )


def _execute_cleanup(settings: Any, extraction_result: Any, keep_media: bool) -> tuple[bool, int]:
    """Clean up run artifacts and report success plus remaining file count."""
    if keep_media or not extraction_result or not getattr(extraction_result, "run_id", None):
        return True, 0

    run_dir = Path(settings.temp_dir) / extraction_result.run_id
    if not run_dir.exists():
        return True, 0

    try:
        shutil.rmtree(run_dir, ignore_errors=False)
        return True, 0
    except Exception as exc:
        logger.warning("Cleanup failed for %s: %s", run_dir, exc)
        leftovers = len(list(run_dir.glob("*"))) if run_dir.exists() else 0
        return False, leftovers
