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

FOOTER = "No OCR or vision analysis is being performed yet."
CAPTION_PREVIEW_CHARS = 500


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="analyzer.py",
        description=(
            f"{app_config.PROJECT_NAME} - Phase {app_config.PHASE_NUMBER}: "
            f"{app_config.PHASE}. Validates an Instagram URL, then attempts to "
            "extract media and metadata with yt-dlp. No multimodal analysis "
            "is performed yet."
        ),
        epilog=(
            "Examples:\n"
            '  python analyzer.py "https://www.instagram.com/reel/example/"\n'
            '  python analyzer.py --keep-media "https://www.instagram.com/p/example/"'
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
    elif result.success and result.media_type in ("image", "carousel"):
        print()
        print("--- Phase 5: Audio / Speech Understanding ---")
        print("Skipped (media type is not video).")
        print()
        print(FOOTER)
    elif args.skip_speech and result.success:
        print()
        print("--- Phase 5: Audio / Speech Understanding ---")
        print("Skipped (--skip-speech).")
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
