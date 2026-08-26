"""Live audio/speech integration test runner (Phase 5).

This script loads the REAL faster-whisper model and processes REAL extracted
Instagram video. It is kept out of the pytest suite on purpose: ``pytest``
must stay fully offline and must never download models.

Usage:
    python scripts/test_audio.py "https://www.instagram.com/p/DcWXVZlMwOB/"
    python scripts/test_audio.py --media-path path/to/video.mp4
    python scripts/test_audio.py --keep-audio "https://www.instagram.com/p/DcWXVZlMwOB/"
    python scripts/test_audio.py --help

With --media-path, the script skips extraction and uses the given file
directly. Without it, the full extraction → speech pipeline runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config  # noqa: E402
from processor.pipeline import process_speech  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/test_audio.py",
        description="Run real ASR on an extracted Instagram video (Phase 5).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Instagram URL to extract and process.",
    )
    parser.add_argument(
        "--media-path",
        default=None,
        help="Path to an already-downloaded media file (skips extraction).",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Retain the downloaded video after processing.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain the temporary WAV after processing (debug).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser


def _extract_media(url: str, settings, logger, keep_media: bool) -> tuple[str | None, str | None]:
    """Extract media from the Instagram URL. Returns (media_path, run_id)."""
    from extractor import (
        ExtractionMode,
        ExtractionOptions,
        YtDlpExtractor,
        validate_instagram_url,
    )

    validation = validate_instagram_url(url)
    if not validation.valid:
        print(f"URL validation failed: {validation.error_code}")
        print(f"  {validation.error_message}")
        return None, None

    options = ExtractionOptions(
        mode=ExtractionMode.from_string(settings.extraction_mode),
        cookies_from_browser=settings.cookies_from_browser or None,
        cookie_file=settings.cookie_file or None,
        keep_media=True,  # always keep for speech processing
    )
    extractor = YtDlpExtractor(temp_dir=settings.temp_dir)
    result = extractor.extract(validation, options)

    if not result.success or not result.media_path:
        print(f"Extraction failed: {result.failure_category}")
        print(f"  {result.failure_reason}")
        return None, result.run_id

    print(f"Extraction: PASS")
    print(f"Media type: {result.media_type}")
    print(f"Media path: {result.media_path}")
    print(f"Download time: {result.download_seconds:.3f}s")
    print()

    return result.media_path, result.run_id


def _print_speech_report(result) -> None:
    """Print a structured speech processing report."""
    print(f"{app_config.PROJECT_NAME}")
    print(f"Phase: Audio / Speech Understanding")
    print()

    print(f"Audio:")
    print(f"  present: {'yes' if result.audio_present else 'no'}")
    print()

    print(f"Speech:")
    print(f"  detected: {'yes' if result.speech_present else 'no'}")
    print()

    print(f"Classification:")
    print(f"  {result.classification}")
    print()

    if result.detected_language:
        print(f"Detected language:")
        print(f"  {result.detected_language}")
        if result.language_probability is not None:
            print(f"  probability: {result.language_probability:.4f}")
        print()

    if result.transcript:
        print(f"Transcript:")
        preview = result.transcript[:1000]
        if len(result.transcript) > 1000:
            preview += " [...]"
        print(f"  {preview}")
        print()

    if result.segments:
        print(f"Segments ({len(result.segments)}):")
        for seg in result.segments[:10]:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        if len(result.segments) > 10:
            print(f"  ... and {len(result.segments) - 10} more")
        print()

    print(f"Timing:")
    if result.audio_extraction_seconds is not None:
        print(f"  audio extraction: {result.audio_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        print(f"  model load: {result.model_load_seconds:.3f}s")
    if result.transcription_time_seconds is not None:
        print(f"  transcription: {result.transcription_time_seconds:.3f}s")
    if result.total_processing_seconds is not None:
        print(f"  total processing: {result.total_processing_seconds:.3f}s")
    if result.duration_seconds is not None:
        print(f"  audio duration: {result.duration_seconds:.1f}s")
    print()

    if result.model_name:
        print(f"Model: {result.model_name}")
        print()

    if not result.success:
        print(f"Failure:")
        print(f"  category: {result.failure_category}")
        print(f"  message: {result.failure_message}")
        print()

    print(f"Result: {'PASS' if result.success else 'FAIL'}")
    print()
    print("No OCR or vision analysis is being performed yet.")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.url is None and args.media_path is None:
        print("Provide either a URL or --media-path.")
        return 1

    settings = app_config.get_config()
    logger = app_config.configure_logging("DEBUG" if args.verbose else settings.log_level)
    settings.ensure_directories()

    # Ensure stdout handles unicode.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    media_path = args.media_path
    run_id = None

    if media_path is None:
        # Extract from Instagram.
        media_path, run_id = _extract_media(args.url, settings, logger, args.keep_media)
        if media_path is None:
            return 1

    print(f"--- Speech Processing ---")
    print()

    result = process_speech(
        media_path,
        keep_audio=args.keep_audio,
    )

    _print_speech_report(result)

    # Save result to output.
    output_path = settings.output_dir / "speech_result.json"
    output_path.write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResult saved to: {output_path}")

    # Cleanup extraction artifacts (but NOT the source video before we're done).
    if run_id and not args.keep_media:
        run_path = settings.temp_dir / run_id
        if run_path.exists():
            shutil.rmtree(run_path, ignore_errors=True)
            logger.info("Cleanup: removed temp run directory run=%s", run_id)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
