"""End-to-end processing pipeline for speech (Phase 5) and vision (Phase 6).

Chains together:
1. Speech processing (audio detection → WAV extraction → faster-whisper transcription → cleanup).
2. Vision processing (media validation → keyframe sampling → VLM visual understanding → cleanup).

The caller supplies a path to an already-extracted media file (from the
extractor layer). This module does NOT download Instagram content.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from processor.audio import AudioExtractionError, extract_audio_to_wav, has_audio_stream
from processor.frames import FrameExtractionError, sample_frames
from processor.models import (
    OCRFailureCategory,
    OCRResult,
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    VisionFailureCategory,
    VisionResult,
)
from processor.ocr import BaseOCRAnalyzer, LocalOCRAnalyzer
from processor.speech import BaseSpeechRecognizer, FasterWhisperRecognizer
from processor.synthesis import (
    BaseSynthesizer,
    LocalQwenSynthesizer,
    TokenRouterGLMSynthesizer,
)
from processor.synthesis_models import (
    MultimodalAnalysisResult,
    MultimodalEvidence,
    SynthesisFailureCategory,
)
from processor.vision import BaseVisionAnalyzer, LocalVisionAnalyzer

logger = logging.getLogger("analyzer.processor.pipeline")




def process_speech(
    media_path: str | Path,
    recognizer: BaseSpeechRecognizer | None = None,
    keep_audio: bool = False,
) -> SpeechResult:
    """Run the full audio/speech pipeline on *media_path*.

    Parameters
    ----------
    media_path:
        Path to the extracted video/audio file.
    recognizer:
        ASR backend to use. If ``None``, a default
        :class:`FasterWhisperRecognizer` is created from environment config.
    keep_audio:
        If True, the temporary WAV is *not* deleted after processing.
        Intended for debugging only.

    Returns
    -------
    SpeechResult
        A fully populated result. Never raises — errors are captured in the
        result's failure fields.
    """
    total_t0 = time.perf_counter()
    media_path = Path(media_path)

    # Validate input.
    if not media_path.exists():
        return SpeechResult(
            success=False,
            failure_category=SpeechFailureCategory.AUDIO_NOT_FOUND.value,
            failure_message=f"Media file does not exist: {media_path}",
        )

    # Check for supported media types.
    suffix = media_path.suffix.lower()
    supported = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    if suffix not in supported:
        return SpeechResult(
            success=False,
            failure_category=SpeechFailureCategory.UNSUPPORTED_MEDIA.value,
            failure_message=f"Unsupported media extension: {suffix}",
        )

    # Step 1: probe for audio stream.
    if not has_audio_stream(media_path):
        return SpeechResult(
            success=True,
            audio_present=False,
            speech_present=False,
            classification=SpeechClassification.NO_AUDIO.value,
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    # Step 2: extract audio to temporary WAV.
    wav_path: Optional[Path] = None
    try:
        wav_path, duration, extraction_seconds = extract_audio_to_wav(media_path)
    except AudioExtractionError as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=exc.category,
            failure_message=exc.message,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    except Exception as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=SpeechFailureCategory.AUDIO_EXTRACTION_FAILED.value,
            failure_message=str(exc),
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    # Step 3: transcribe.
    try:
        if recognizer is None:
            recognizer = FasterWhisperRecognizer()

        result = recognizer.transcribe(wav_path)

        # Enrich the result with pipeline-level timing.
        result.audio_present = True
        result.audio_extraction_seconds = extraction_seconds
        result.duration_seconds = duration or result.duration_seconds
        result.total_processing_seconds = time.perf_counter() - total_t0

        return result
    except Exception as exc:
        return SpeechResult(
            success=False,
            audio_present=True,
            failure_category=SpeechFailureCategory.ASR_INFERENCE_FAILED.value,
            failure_message=str(exc),
            audio_extraction_seconds=extraction_seconds,
            duration_seconds=duration,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    finally:
        # Step 4: clean up temporary audio.
        if wav_path and wav_path.exists() and not keep_audio:
            try:
                wav_path.unlink()
                logger.debug("Cleaned up temporary audio: %s", wav_path)
            except OSError as exc:
                logger.warning("Could not delete temporary audio %s: %s", wav_path, exc)


def process_vision(
    media_path: str | Path,
    analyzer: BaseVisionAnalyzer | None = None,
    max_frames: int | None = None,
    keep_frames: bool = False,
) -> VisionResult:
    """Run the vision understanding pipeline on *media_path* (Phase 6).

    Parameters
    ----------
    media_path:
        Path to image or video file.
    analyzer:
        Vision backend to use. If ``None``, a default :class:`LocalVisionAnalyzer`
        is created from environment config.
    max_frames:
        Number of frames to sample if video. If ``None``, reads VISION_MAX_FRAMES
        from environment (default: 6).
    keep_frames:
        If True, temporary frame JPEGs are *not* deleted after processing.

    Returns
    -------
    VisionResult
        A fully populated result. Never raises — errors are captured in failure fields.
    """
    total_t0 = time.perf_counter()
    media_path = Path(media_path)

    # Validate input file existence
    if not media_path.exists():
        return VisionResult(
            success=False,
            media_path=str(media_path),
            failure_category=VisionFailureCategory.MEDIA_NOT_FOUND.value,
            failure_message=f"Media file does not exist: {media_path}",
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    ext = media_path.suffix.lower()
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    vid_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}

    if ext in img_exts:
        input_type = "image"
    elif ext in vid_exts:
        input_type = "video"
    else:
        return VisionResult(
            success=False,
            media_path=str(media_path),
            failure_category=VisionFailureCategory.UNSUPPORTED_MEDIA.value,
            failure_message=f"Unsupported media extension for vision: {ext}",
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    if max_frames is None:
        try:
            max_frames = int(os.environ.get("VISION_MAX_FRAMES", "6"))
        except ValueError:
            max_frames = 6

    # Step 1: Sample frames
    extracted_frames: List[Path] = []
    frame_extraction_seconds = 0.0
    try:
        extracted_frames, _duration, frame_extraction_seconds = sample_frames(
            media_path, max_frames=max_frames
        )
    except FrameExtractionError as exc:
        return VisionResult(
            success=False,
            media_path=str(media_path),
            input_type=input_type,
            failure_category=exc.category,
            failure_message=exc.message,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    except Exception as exc:
        return VisionResult(
            success=False,
            media_path=str(media_path),
            input_type=input_type,
            failure_category=VisionFailureCategory.FRAME_EXTRACTION_FAILED.value,
            failure_message=str(exc),
            total_processing_seconds=time.perf_counter() - total_t0,
        )

    # Step 2: Analyze frames with vision model
    try:
        if analyzer is None:
            analyzer = LocalVisionAnalyzer()

        result = analyzer.analyze_frames(
            frame_paths=extracted_frames,
            media_path=media_path,
            input_type=input_type,
            frame_extraction_seconds=frame_extraction_seconds,
        )
        result.total_processing_seconds = time.perf_counter() - total_t0
        return result

    except Exception as exc:
        return VisionResult(
            success=False,
            media_path=str(media_path),
            input_type=input_type,
            failure_category=VisionFailureCategory.VISION_INFERENCE_FAILED.value,
            failure_message=str(exc),
            frame_extraction_seconds=frame_extraction_seconds,
            total_processing_seconds=time.perf_counter() - total_t0,
        )
    finally:
        # Step 3: Clean up temporary frame files if extracted for video
        if input_type == "video" and not keep_frames:
            for fpath in extracted_frames:
                try:
                    if fpath.exists() and fpath != media_path:
                        fpath.unlink()
                        logger.debug("Cleaned up temp frame: %s", fpath)
                except OSError as exc:
                    logger.warning("Could not delete temp frame %s: %s", fpath, exc)


def process_ocr(
    media_input: str | Path | Sequence[str | Path],
    analyzer: BaseOCRAnalyzer | None = None,
    max_frames: int | None = None,
    keep_frames: bool = False,
) -> OCRResult:
    """Run the OCR / on-screen text extraction pipeline (Phase 7).

    Supports:
    - Single image (.jpg, .jpeg, .png, .webp, .bmp)
    - Video / Reel (.mp4, .mkv, .webm, .avi, .mov, .m4v) via frame sampling
    - Carousel images (Sequence of image paths)

    Parameters
    ----------
    media_input:
        Path to image/video file, or a sequence of paths for carousel media.
    analyzer:
        OCR backend to use. If ``None``, defaults to :class:`LocalOCRAnalyzer`.
    max_frames:
        Maximum frames to sample for video. If ``None``, reads OCR_MAX_FRAMES
        from environment (default: 6).
    keep_frames:
        If True, temporary sampled video frames are retained.

    Returns
    -------
    OCRResult
        Structured OCR result. Never raises — errors are captured in result fields.
    """
    total_t0 = time.perf_counter()
    extracted_frames: List[Path] = []
    frame_extraction_seconds = 0.0
    media_path_str: Optional[str] = None

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    vid_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}

    # Case 1: Carousel input (sequence of media paths)
    if isinstance(media_input, (list, tuple)):
        media_type = "carousel"
        if not media_input:
            return OCRResult(
                success=False,
                media_type=media_type,
                failure_category=OCRFailureCategory.INVALID_MEDIA.value,
                failure_message="Empty carousel media list provided.",
                processing_time_seconds=time.perf_counter() - total_t0,
            )

        frame_items: List[Tuple[Path, int, float]] = []
        for idx, item in enumerate(media_input):
            p = Path(item)
            if not p.exists():
                return OCRResult(
                    success=False,
                    media_type=media_type,
                    media_path=str(p),
                    failure_category=OCRFailureCategory.INVALID_MEDIA.value,
                    failure_message=f"Carousel item does not exist: {p}",
                    processing_time_seconds=time.perf_counter() - total_t0,
                )
            frame_items.append((p, idx, 0.0))

    # Case 2: Single media file (image or video)
    else:
        media_path = Path(media_input)
        media_path_str = str(media_path)

        if not media_path.exists():
            return OCRResult(
                success=False,
                media_type="unknown",
                media_path=media_path_str,
                failure_category=OCRFailureCategory.INVALID_MEDIA.value,
                failure_message=f"Media file does not exist: {media_path}",
                processing_time_seconds=time.perf_counter() - total_t0,
            )

        ext = media_path.suffix.lower()

        if ext in img_exts:
            media_type = "image"
            frame_items = [(media_path, 0, 0.0)]

        elif ext in vid_exts:
            media_type = "video"
            if max_frames is None:
                try:
                    max_frames = int(os.environ.get("OCR_MAX_FRAMES", "6"))
                except ValueError:
                    max_frames = 6

            try:
                extracted_paths, duration, frame_extraction_seconds = sample_frames(
                    media_path, max_frames=max_frames
                )
            except FrameExtractionError as exc:
                return OCRResult(
                    success=False,
                    media_type=media_type,
                    media_path=media_path_str,
                    failure_category=exc.category,
                    failure_message=exc.message,
                    processing_time_seconds=time.perf_counter() - total_t0,
                )
            except Exception as exc:
                return OCRResult(
                    success=False,
                    media_type=media_type,
                    media_path=media_path_str,
                    failure_category=OCRFailureCategory.FRAME_EXTRACTION_FAILED.value,
                    failure_message=str(exc),
                    processing_time_seconds=time.perf_counter() - total_t0,
                )

            extracted_frames = extracted_paths
            # Compute timestamps
            num_frames = len(extracted_paths)
            if num_frames == 1:
                timestamps = [round(duration / 2.0, 2)]
            elif duration > 0 and num_frames > 1:
                step = duration / (num_frames - 1)
                timestamps = [
                    round(min(i * step, max(0.0, duration - 0.1)), 2)
                    for i in range(num_frames)
                ]
            else:
                timestamps = [float(i * 2.0) for i in range(num_frames)]

            frame_items = [
                (extracted_paths[i], i, timestamps[i]) for i in range(num_frames)
            ]

        else:
            return OCRResult(
                success=False,
                media_type="unknown",
                media_path=media_path_str,
                failure_category=OCRFailureCategory.UNSUPPORTED_MEDIA.value,
                failure_message=f"Unsupported media extension for OCR: {ext}",
                processing_time_seconds=time.perf_counter() - total_t0,
            )

    # Step 2: Run OCR analysis
    try:
        if analyzer is None:
            analyzer = LocalOCRAnalyzer()

        result = analyzer.analyze_frames(
            frame_items=frame_items,
            media_type=media_type,
            media_path=Path(media_path_str) if media_path_str else None,
            frame_extraction_seconds=frame_extraction_seconds,
        )
        result.processing_time_seconds = time.perf_counter() - total_t0
        return result


    except Exception as exc:
        logger.error("Unexpected OCR processing error: %s", exc)
        return OCRResult(
            success=False,
            media_type=media_type,
            media_path=media_path_str,
            failure_category=OCRFailureCategory.OCR_INFERENCE_FAILED.value,
            failure_message=str(exc),
            frame_extraction_seconds=frame_extraction_seconds,
            processing_time_seconds=time.perf_counter() - total_t0,
        )

    finally:
        # Step 3: Temporary frame cleanup
        if media_type == "video" and not keep_frames:
            for fpath in extracted_frames:
                try:
                    if fpath.exists() and str(fpath) != media_path_str:
                        fpath.unlink()
                        logger.debug("Cleaned up temp OCR frame: %s", fpath)
                except OSError as exc:
                    logger.warning("Could not delete temp OCR frame %s: %s", fpath, exc)


def process_synthesis(
    extraction_result: Any = None,
    speech_result: Optional[SpeechResult] = None,
    vision_result: Optional[VisionResult] = None,
    ocr_result: Optional[OCRResult] = None,
    synthesizer: Optional[BaseSynthesizer] = None,
    enabled: Optional[bool] = None,
    *,
    extraction: Any = None,
    speech: Optional[SpeechResult] = None,
    vision: Optional[VisionResult] = None,
    ocr: Optional[OCRResult] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> MultimodalAnalysisResult:
    """Run multimodal synthesis combining extracted evidence (Phase 8).

    Parameters
    ----------
    extraction_result / extraction:
        ExtractionResult from Phase 3/4.
    speech_result / speech:
        Optional SpeechResult from Phase 5.
    vision_result / vision:
        Optional VisionResult from Phase 6.
    ocr_result / ocr:
        Optional OCRResult from Phase 7.
    synthesizer:
        Synthesizer backend. If ``None``, defaults to :class:`TokenRouterGLMSynthesizer`.
    enabled:
        Whether synthesis is active. If ``None``, reads SYNTHESIS_ENABLED env var (default: True).

    Returns
    -------
    MultimodalAnalysisResult
        Structured synthesis result. Never raises — errors are captured in result fields.
    """
    actual_extraction = extraction_result if extraction_result is not None else extraction
    actual_speech = speech_result if speech_result is not None else speech
    actual_vision = vision_result if vision_result is not None else vision
    actual_ocr = ocr_result if ocr_result is not None else ocr

    if enabled is None:
        enabled_raw = os.environ.get("SYNTHESIS_ENABLED", "true").lower()
        enabled = enabled_raw in ("true", "1", "yes")

    if not enabled:
        return MultimodalAnalysisResult(
            success=False,
            failure_category=SynthesisFailureCategory.SYNTHESIS_DISABLED.value,
            failure_message="Multimodal synthesis is disabled (SYNTHESIS_ENABLED=false).",
        )

    evidence = MultimodalEvidence.from_results(
        extraction_result=actual_extraction,
        speech_result=actual_speech,
        vision_result=actual_vision,
        ocr_result=actual_ocr,
    )

    if synthesizer is None:
        backend = (os.environ.get("SYNTHESIS_BACKEND") or "local").strip().lower()
        if backend in ("tokenrouter", "cloud", "glm"):
            synthesizer = TokenRouterGLMSynthesizer(
                api_key=api_key,
                model=model,
                endpoint=endpoint,
            )
        else:
            model_id = model or os.environ.get("SYNTHESIS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
            device = os.environ.get("SYNTHESIS_DEVICE", "cpu")
            synthesizer = LocalQwenSynthesizer(
                model_id=model_id,
                device=device,
            )

    return synthesizer.synthesize(evidence)


