"""Instagram Content Analyzer - command line entry point.

Phase 5 (Audio / Speech Understanding). The pipeline is:

    input URL
      -> Phase 2 offline URL validation
      -> if invalid: report and stop
      -> Phase 3/4 extraction (yt-dlp) into an isolated temp directory
      -> Phase 5 audio/speech processing (faster-whisper, local, free)
      -> structured extraction + speech report
      -> temporary media cleaned up (unless --keep-media)

No multimodal analysis (OCR, vision, LLM) happens here; that is a later
phase. The only network access in the whole program lives inside the
extraction layer (and one-time model download via HuggingFace Hub).

Usage:
    python analyzer.py "https://www.instagram.com/reel/example/"
    python analyzer.py --keep-media "https://www.instagram.com/p/example/"
    python analyzer.py --help
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap

import config as app_config
from extractor import (
    ExtractionMode,
    ExtractionOptions,
    ExtractionResult,
    ValidationResult,
    YtDlpExtractor,
    validate_instagram_url,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_ENVIRONMENT = 2
EXIT_INVALID_URL = 3
EXIT_EXTRACTION_FAILED = 4

USAGE_HINT = (
    "No Instagram URL provided.\n"
    "\n"
    "Usage:\n"
    '    python analyzer.py "https://www.instagram.com/reel/example/"\n'
    "\n"
    "Run 'python analyzer.py --help' for all options."
)

FOOTER = "Multimodal analysis pipeline complete."
CAPTION_PREVIEW_CHARS = 500


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="analyzer.py",
        description=(
            f"{app_config.PROJECT_NAME} - Phase {app_config.PHASE_NUMBER}: "
            f"{app_config.PHASE}. Validates an Instagram URL, extracts media, "
            "and performs speech, vision, OCR, and multimodal synthesis."
        ),
        epilog=(
            "Examples:\n"
            '  python analyzer.py "https://www.instagram.com/reel/example/"\n'
            '  python analyzer.py --ocr-only "https://www.instagram.com/reel/example/"\n'
            '  python analyzer.py --skip-synthesis "https://www.instagram.com/p/example/"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Public Instagram URL (post, reel or IGTV).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging and show extra (sanitised) diagnostics.",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Do not delete the downloaded media / temp run directory.",
    )
    parser.add_argument(
        "--skip-speech",
        action="store_true",
        help="Skip Phase 5 audio/speech processing.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip Phase 6 vision understanding.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip Phase 7 OCR processing.",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Run only Phase 7 OCR processing (skip speech, vision, and synthesis).",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Skip Phase 8 Multimodal Synthesis.",
    )
    parser.add_argument(
        "--synthesis-only",
        action="store_true",
        help="Run extraction and synthesis directly (skip speech, vision, and OCR).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"{app_config.PROJECT_NAME} "
            f"(phase {app_config.PHASE_NUMBER}: {app_config.PHASE})"
        ),
    )
    return parser




def print_header() -> None:
    """Print the two identifying lines shared by every report."""
    print(app_config.PROJECT_NAME)
    print(f"Phase: {app_config.PHASE}")


def report_invalid_url(result: ValidationResult) -> None:
    """Print the report for a URL that failed Phase 2 validation."""
    print_header()
    print()
    if result.input_url:
        print("Input URL:")
        print(result.input_url)
        print()
    print("URL validation: FAIL")
    print("Reason:")
    print(result.error_code)
    print()
    print("Details:")
    print(result.error_message)
    print()
    print(FOOTER)


def _yn(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def report_extraction(result: ExtractionResult, verbose: bool) -> None:
    """Print the structured extraction report."""
    print_header()
    print()
    print("Input URL:")
    print(result.source_url)
    if result.normalized_url:
        print("Normalized URL:")
        print(result.normalized_url)
    print()
    print("URL validation: PASS")
    print()
    print(f"Extraction: {_yn(result.success)}")
    print()
    print(f"Media downloaded: {_yn(result.media_downloaded)}")
    print(f"Caption extracted: {_yn(result.caption_extracted)}")
    print(f"Media type detected: {_yn(result.media_type_detected)}")
    print()

    print("Media type:")
    print(result.media_type)
    print()

    print("Media path:")
    print(result.media_path if result.media_path else "(none)")
    print()

    print("Caption:")
    if result.caption:
        preview = result.caption.strip()
        if len(preview) > CAPTION_PREVIEW_CHARS:
            preview = preview[:CAPTION_PREVIEW_CHARS] + " [...]"
        print(preview)
    else:
        print("(unavailable)")
    print()

    print("Hashtags:")
    print(" ".join(result.hashtags) if result.hashtags else "(none)")
    print()

    if not result.success or result.failure_category:
        print("Failure category:")
        print(result.failure_category or "(none)")
        if result.failure_reason:
            print("Failure reason:")
            print(result.failure_reason)
        print()

    if verbose:
        _print_diagnostics(result)

    print(FOOTER)


def report_speech(result) -> None:
    """Print the Phase 5 speech processing report."""
    print()
    print("--- Phase 5: Audio / Speech Understanding ---")
    print()
    print(f"Audio present: {'yes' if result.audio_present else 'no'}")
    print(f"Speech detected: {'yes' if result.speech_present else 'no'}")
    print(f"Classification: {result.classification}")
    print()

    if result.detected_language:
        print(f"Detected language: {result.detected_language}")
        if result.language_probability is not None:
            print(f"Language probability: {result.language_probability:.4f}")
        print()

    if result.transcript:
        print("Transcript:")
        preview = result.transcript.strip()
        if len(preview) > CAPTION_PREVIEW_CHARS:
            preview = preview[:CAPTION_PREVIEW_CHARS] + " [...]"
        print(preview)
        print()

    if result.segments:
        print(f"Segments: {len(result.segments)}")
        for seg in result.segments[:5]:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        if len(result.segments) > 5:
            print(f"  ... and {len(result.segments) - 5} more")
        print()

    # Timing.
    timings = []
    if result.audio_extraction_seconds is not None:
        timings.append(f"audio extraction: {result.audio_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        timings.append(f"model load: {result.model_load_seconds:.3f}s")
    if result.transcription_time_seconds is not None:
        timings.append(f"transcription: {result.transcription_time_seconds:.3f}s")
    if result.total_processing_seconds is not None:
        timings.append(f"total: {result.total_processing_seconds:.3f}s")
    if timings:
        print("Speech timing:")
        for t in timings:
            print(f"  {t}")
        print()

    if result.model_name:
        print(f"ASR model: {result.model_name}")
        print()

    if not result.success:
        print(f"Speech failure: {result.failure_category}")
        if result.failure_message:
            print(f"  {result.failure_message}")
        print()

    print(FOOTER)


def report_vision(result) -> None:
    """Print the Phase 6 vision processing report."""
    print()
    print("--- Phase 6: Vision Understanding ---")
    print()
    print(f"Input type: {result.input_type}")
    print(f"Frames analyzed: {result.frames_analyzed}")
    print()

    if result.subjects:
        print(f"Subjects: {', '.join(result.subjects)}")
    if result.objects:
        print(f"Objects: {', '.join(result.objects)}")
    if result.actions:
        print(f"Actions: {', '.join(result.actions)}")
    if result.scenes:
        print(f"Scenes: {', '.join(result.scenes)}")
    if result.demonstrations:
        print(f"Demonstrations: {', '.join(result.demonstrations)}")
    print()

    if result.frame_observations:
        print(f"Frame observations ({len(result.frame_observations)}):")
        for fobs in result.frame_observations[:5]:
            print(f"  [Frame {fobs.frame_index} @ {fobs.timestamp_seconds:.1f}s] {fobs.description}")
        if len(result.frame_observations) > 5:
            print(f"  ... and {len(result.frame_observations) - 5} more")
        print()

    # Timing
    timings = []
    if result.frame_extraction_seconds is not None:
        timings.append(f"frame extraction: {result.frame_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        timings.append(f"model load: {result.model_load_seconds:.3f}s")
    if result.inference_seconds is not None:
        timings.append(f"inference: {result.inference_seconds:.3f}s")
    if result.total_processing_seconds is not None:
        timings.append(f"total: {result.total_processing_seconds:.3f}s")
    if timings:
        print("Vision timing:")
        for t in timings:
            print(f"  {t}")
        print()

    if result.model_name:
        print(f"Vision model: {result.model_name}")
        print()

    if not result.success:
        print(f"Vision failure: {result.failure_category}")
        if result.failure_message:
            print(f"  {result.failure_message}")
        print()

    print(FOOTER)


def report_ocr(result) -> None:
    """Print the Phase 7 OCR / on-screen text processing report."""
    print()
    print("--- Phase 7: OCR / On-Screen Text ---")
    print()
    print("OCR:")
    print("PASS" if result.success else "FAIL")
    print()
    print("Frames analyzed:")
    print(result.frames_analyzed)
    print()
    print("Detected text:")
    if result.text_blocks:
        for block in result.text_blocks:
            ts_str = f" [t={block.timestamp_seconds:.1f}s]" if block.timestamp_seconds is not None else ""
            conf_str = f" [conf: {block.confidence:.2f}]" if block.confidence is not None else ""
            print(f"- {block.text}{ts_str}{conf_str}")
    elif result.success:
        print("(no visible text detected)")
    else:
        print("(unavailable due to OCR error)")
    print()

    # Timing
    timings = []
    if result.frame_extraction_seconds is not None:
        timings.append(f"frame extraction: {result.frame_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        timings.append(f"model load: {result.model_load_seconds:.3f}s")
    if result.inference_seconds is not None:
        timings.append(f"inference: {result.inference_seconds:.3f}s")
    if result.processing_time_seconds is not None:
        timings.append(f"total: {result.processing_time_seconds:.3f}s")
    if timings:
        print("OCR processing time:")
        for t in timings:
            print(f"  {t}")
        print()

    if result.model_name_or_engine:
        print(f"OCR engine: {result.model_name_or_engine}")
        print()

    if not result.success:
        print(f"OCR failure: {result.failure_category}")
        if result.failure_message:
            print(f"  {result.failure_message}")
        print()

    print(FOOTER)


def report_synthesis(result) -> None:
    """Print the Phase 8 Multimodal Synthesis processing report."""
    print()
    print("--- Phase 8: Multimodal Synthesis ---")
    print()
    print("Synthesis:")
    print("PASS" if result.success else "FAIL")
    print()
    if result.success:
        print("Summary:")
        print(result.summary)
        print()
        print("Key points:")
        for point in result.key_points:
            print(f"- {point}")
        print()
        print("Core takeaway:")
        print(result.core_takeaway)
        print()
        if result.relevant_context:
            print("Relevant context:")
            print(result.relevant_context)
            print()
        print(f"Confidence: {result.confidence:.2f}")
        print()
        print("Evidence used:")
        for src, used in sorted(result.evidence_used.items()):
            print(f"  {src}: {'true' if used else 'false'}")
        print()
        # Metrics
        metrics = []
        if result.prompt_tokens is not None:
            metrics.append(f"prompt tokens: {result.prompt_tokens}")
        if result.completion_tokens is not None:
            metrics.append(f"completion tokens: {result.completion_tokens}")
        if result.total_tokens is not None:
            metrics.append(f"total tokens: {result.total_tokens}")
        if result.request_latency_seconds is not None:
            metrics.append(f"request latency: {result.request_latency_seconds:.3f}s")
        if result.processing_time_seconds is not None:
            metrics.append(f"total time: {result.processing_time_seconds:.3f}s")
        if metrics:
            print("Synthesis metrics:")
            for m in metrics:
                print(f"  {m}")
            print()
        if result.model_name:
            print(f"Synthesis model: {result.model_name}")
            print()
    else:
        print(f"Synthesis failure: {result.failure_category}")
        if result.failure_message:
            print(f"  {result.failure_message}")
        print()
    print(FOOTER)




def _print_diagnostics(result: ExtractionResult) -> None:
    """Show extra, already-sanitised diagnostics under --verbose."""
    print("--- diagnostics (sanitised) ---")
    print(f"run_id: {result.run_id}")
    print(f"extraction_mode: {result.extraction_mode}")
    print(f"cookie_file_configured: {'yes' if result.cookie_file_configured else 'no'}")
    print(f"content_type_hint: {result.content_type_hint}")
    print(f"download_seconds: {result.download_seconds}")
    print(f"media_files: {len(result.media_files)}")
    for media in result.media_files:
        print(f"  - {media.kind} {media.ext} {media.size_bytes}B")
    if result.metadata:
        print("metadata:")
        for key, value in result.metadata.items():
            line = f"  {key}: {value}"
            print(textwrap.shorten(line, width=100, placeholder=" [...]"))
    print()


def _cleanup_run(settings, result: ExtractionResult, logger) -> None:
    """Remove the temp run directory once the result has been printed.

    The extractor already removes partial downloads on failure; this handles
    the successful case, where media was intentionally kept until the CLI (the
    current consumer) finished using it. A future Phase 4 consumer would take
    ownership before this point instead.
    """
    if not result.run_id:
        return
    run_path = settings.temp_dir / result.run_id
    if run_path.exists():
        shutil.rmtree(run_path, ignore_errors=True)
        logger.info("Cleanup: removed temp run directory run=%s", result.run_id)


def _ensure_utf8_stdout() -> None:
    """Make stdout/stderr tolerate emoji-laden captions.

    Instagram captions routinely contain emoji; on a legacy Windows console
    (cp1252) a bare ``print`` would raise UnicodeEncodeError. Reconfigure to
    UTF-8 and, failing that, replace unencodable characters rather than crash.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass


def run(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = app_config.get_config()
    log_level = "DEBUG" if args.verbose else settings.log_level
    logger = app_config.configure_logging(log_level)
    logger.debug("Configuration loaded: %s", settings)

    if args.url is None:
        logger.debug("No URL argument supplied.")
        print(USAGE_HINT)
        return EXIT_USAGE

    environment_ok, problems = settings.check_environment()
    for problem in problems:
        logger.warning("Environment problem: %s", problem)
    if not environment_ok:
        print_header()
        print()
        print("Environment check: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print(FOOTER)
        return EXIT_ENVIRONMENT

    # -- Phase 2: validation ------------------------------------------------
    logger.info("Validating URL")
    validation = validate_instagram_url(args.url)
    if not validation.valid:
        logger.warning(
            "Validation failed: %s", validation.error_code
        )
        report_invalid_url(validation)
        return EXIT_INVALID_URL

    logger.info(
        "Validation passed (hint=%s); proceeding to extraction",
        validation.content_type_hint,
    )

    # -- Phase 3: extraction ------------------------------------------------
    options = ExtractionOptions(
        mode=ExtractionMode.from_string(settings.extraction_mode),
        cookies_from_browser=settings.cookies_from_browser or None,
        cookie_file=settings.cookie_file or None,
        keep_media=args.keep_media,
    )
    extractor = YtDlpExtractor(temp_dir=settings.temp_dir)
    result = extractor.extract(validation, options)

    if args.ocr_only:
        args.skip_speech = True
        args.skip_vision = True
        args.skip_synthesis = True
    if args.synthesis_only:
        args.skip_speech = True
        args.skip_vision = True
        args.skip_ocr = True

    report_extraction(result, verbose=args.verbose)

    # -- Phase 5: speech processing (video only) ---------------------------
    speech_result = None
    if (
        result.success
        and result.media_path
        and result.media_type in ("reel", "video")
        and not args.skip_speech
    ):
        from processor.pipeline import process_speech

        speech_result = process_speech(result.media_path)
        report_speech(speech_result)
    elif result.success and result.media_type in ("image", "carousel") and not (args.ocr_only or args.synthesis_only):
        print()
        print("--- Phase 5: Audio / Speech Understanding ---")
        print("Skipped (media type is not video).")
        print()
        print(FOOTER)
    elif args.skip_speech and result.success and not (args.ocr_only or args.synthesis_only):
        print()
        print("--- Phase 5: Audio / Speech Understanding ---")
        print("Skipped (--skip-speech).")
        print()
        print(FOOTER)

    # -- Phase 6: vision processing ---------------------------------------
    vision_result = None
    if result.success and result.media_path and not args.skip_vision:
        from processor.pipeline import process_vision

        vision_result = process_vision(result.media_path)
        report_vision(vision_result)
    elif args.skip_vision and result.success and not (args.ocr_only or args.synthesis_only):
        print()
        print("--- Phase 6: Vision Understanding ---")
        print("Skipped (--skip-vision).")
        print()
        print(FOOTER)

    # -- Phase 7: OCR / On-Screen Text ------------------------------------
    ocr_result = None
    if (
        result.success
        and (result.media_path or result.media_files)
        and not args.skip_ocr
    ):
        from processor.pipeline import process_ocr

        if result.media_type == "carousel" and len(result.media_files) > 1:
            media_input = [m.path for m in result.media_files]
        else:
            media_input = result.media_path

        ocr_result = process_ocr(media_input)
        report_ocr(ocr_result)
    elif args.skip_ocr and result.success and not args.synthesis_only:
        print()
        print("--- Phase 7: OCR / On-Screen Text ---")
        print("Skipped (--skip-ocr).")
        print()
        print(FOOTER)

    # -- Phase 8: Multimodal Synthesis ------------------------------------
    synthesis_result = None
    if (
        result.success
        and not args.skip_synthesis
        and settings.synthesis_enabled
    ):
        from processor.pipeline import process_synthesis

        synthesis_result = process_synthesis(
            extraction=result,
            speech=speech_result,
            vision=vision_result,
            ocr=ocr_result,
        )
        report_synthesis(synthesis_result)
    elif args.skip_synthesis and result.success and not args.ocr_only:
        print()
        print("--- Phase 8: Multimodal Synthesis ---")
        print("Skipped (--skip-synthesis).")
        print()
        print(FOOTER)
    elif not settings.synthesis_enabled and result.success and not args.ocr_only:
        print()
        print("--- Phase 8: Multimodal Synthesis ---")
        print("Skipped (synthesis disabled in config).")
        print()
        print(FOOTER)

    if not args.keep_media:
        _cleanup_run(settings, result, logger)
    elif result.run_id:
        logger.info("Retained temp run directory run=%s (--keep-media)", result.run_id)

    return EXIT_OK if result.success else EXIT_EXTRACTION_FAILED



def main() -> None:
    """Console entry point."""
    sys.exit(run())


if __name__ == "__main__":
    main()
