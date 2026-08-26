"""Live vision integration test runner (Phase 6).

This script loads the REAL local vision-language model (HuggingFaceTB/SmolVLM-256M-Instruct)
and processes REAL extracted Instagram media/video frames.
It is kept out of the pytest suite on purpose: ``pytest`` must stay fully offline.

Usage:
    python scripts/test_vision.py "https://www.instagram.com/p/DcWXVZlMwOB/"
    python scripts/test_vision.py --media-path path/to/video.mp4
    python scripts/test_vision.py --keep-frames --keep-media "https://www.instagram.com/p/DcWXVZlMwOB/"
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
from processor.pipeline import process_vision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/test_vision.py",
        description="Run local Vision-Language Model on an extracted Instagram media file (Phase 6).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.instagram.com/p/DcWXVZlMwOB/",
        help="Instagram URL to extract and process visually.",
    )
    parser.add_argument(
        "--media-path",
        default=None,
        help="Path to an already-downloaded media file (skips extraction).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=6,
        help="Maximum keyframes to sample from video (default: 6).",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Retain the downloaded video/image after processing.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Retain temporary sampled frame JPEGs after processing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser


def _extract_media(url: str, settings, logger) -> tuple[str | None, str | None]:
    """Extract media from the Instagram URL using configured extraction mode."""
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
        keep_media=True,
    )
    extractor = YtDlpExtractor(temp_dir=settings.temp_dir)
    result = extractor.extract(validation, options)

    if not result.success or not result.media_path:
        print(f"Extraction failed: {result.failure_category}")
        print(f"  {result.failure_reason}")
        return None, result.run_id

    print("Extraction: PASS")
    print(f"Media type: {result.media_type}")
    print(f"Media path: {result.media_path}")
    print(f"Download time: {result.download_seconds:.3f}s")
    print()

    return result.media_path, result.run_id


def _print_vision_report(result) -> None:
    """Print a structured visual understanding report."""
    print("Instagram Content Analyzer")
    print("Phase: Vision Understanding (Phase 6)")
    print()

    print(f"Input type: {result.input_type}")
    print(f"Frames analyzed: {result.frames_analyzed}")
    print()

    if result.model_name:
        print(f"Vision model: {result.model_name}")
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
        for fobs in result.frame_observations:
            print(f"  [Frame {fobs.frame_index} @ {fobs.timestamp_seconds:.1f}s] {fobs.description}")
        print()

    print("Timing:")
    if result.frame_extraction_seconds is not None:
        print(f"  frame extraction: {result.frame_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        print(f"  model load: {result.model_load_seconds:.3f}s")
    if result.inference_seconds is not None:
        print(f"  model inference: {result.inference_seconds:.3f}s")
    if result.total_processing_seconds is not None:
        print(f"  total processing: {result.total_processing_seconds:.3f}s")
    print()

    if not result.success:
        print("Failure:")
        print(f"  category: {result.failure_category}")
        print(f"  message: {result.failure_message}")
        print()

    print(f"Result: {'PASS' if result.success else 'FAIL'}")
    print()
    print("No OCR or multimodal LLM synthesis is being performed yet.")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    settings = app_config.get_config()
    logger = app_config.configure_logging("DEBUG" if args.verbose else settings.log_level)
    settings.ensure_directories()

    # Unicode stdout handling
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
        media_path, run_id = _extract_media(args.url, settings, logger)
        if media_path is None:
            return 1

    print("--- Vision Processing ---")
    print()

    result = process_vision(
        media_path,
        max_frames=args.max_frames,
        keep_frames=args.keep_frames,
    )

    _print_vision_report(result)

    # Save result to output directory
    output_path = settings.output_dir / "vision_result.json"
    output_path.write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResult saved to: {output_path}")

    # Clean up extraction run dir if not keep_media
    if run_id and not args.keep_media:
        run_path = settings.temp_dir / run_id
        if run_path.exists():
            shutil.rmtree(run_path, ignore_errors=True)
            logger.info("Cleanup: removed temp run directory run=%s", run_id)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
