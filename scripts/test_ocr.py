"""Live OCR / on-screen text integration test runner (Phase 7).

This script loads the REAL local OCR engine (RapidOCR / PP-OCRv4 ONNX) and
processes REAL extracted Instagram media / video frames.
It is kept out of the pytest suite on purpose: ``pytest`` must stay fully offline.

Usage:
    python scripts/test_ocr.py "https://www.instagram.com/p/DcWXVZlMwOB/"
    python scripts/test_ocr.py --media-path path/to/video.mp4
    python scripts/test_ocr.py --keep-frames --keep-media "https://www.instagram.com/p/DcWXVZlMwOB/"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config  # noqa: E402
from processor.pipeline import process_ocr  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/test_ocr.py",
        description="Run local OCR engine on extracted Instagram media (Phase 7).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.instagram.com/p/DcWXVZlMwOB/",
        help="Instagram URL to extract and process with OCR.",
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


def _print_ocr_report(result) -> None:
    """Print structured OCR experiment report."""
    print("Instagram Content Analyzer")
    print("Phase: OCR / On-Screen Text (Phase 7)")
    print()

    print(f"Extraction: PASS")
    print(f"Media type: {result.media_type}")
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
            f_str = f" [frame {block.frame_index}]"
            print(f"- {block.text}{f_str}{ts_str}{conf_str}")
    elif result.success:
        print("(no visible text detected)")
    else:
        print("(unavailable due to OCR error)")
    print()

    if result.per_frame_results:
        print(f"Per-frame breakdown ({len(result.per_frame_results)} frames):")
        for f in result.per_frame_results:
            ts = f"{f.timestamp_seconds:.1f}s" if f.timestamp_seconds is not None else "n/a"
            count = len(f.text_blocks)
            print(f"  Frame {f.frame_index} (@ {ts}): {count} text blocks detected")
            for b in f.text_blocks:
                conf = f"{b.confidence:.2f}" if b.confidence is not None else "n/a"
                print(f"    * '{b.text}' (conf={conf}, bbox={b.bbox})")
        print()

    print("OCR processing time:")
    if result.frame_extraction_seconds is not None:
        print(f"  frame extraction: {result.frame_extraction_seconds:.3f}s")
    if result.model_load_seconds is not None:
        print(f"  model load: {result.model_load_seconds:.3f}s")
    if result.inference_seconds is not None:
        print(f"  inference: {result.inference_seconds:.3f}s")
    if result.processing_time_seconds is not None:
        print(f"  total: {result.processing_time_seconds:.3f}s")
    print()

    if result.model_name_or_engine:
        print(f"OCR engine: {result.model_name_or_engine}")
        print()

    if not result.success:
        print("Failure:")
        print(f"  category: {result.failure_category}")
        print(f"  message: {result.failure_message}")
        print()

    print("No speech or multimodal synthesis is being performed here.")


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

    print("--- OCR Processing ---")
    print()

    result = process_ocr(
        media_path,
        max_frames=args.max_frames,
        keep_frames=args.keep_frames,
    )

    _print_ocr_report(result)

    # Save result to output directory
    output_path = settings.output_dir / "ocr_result.json"
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
