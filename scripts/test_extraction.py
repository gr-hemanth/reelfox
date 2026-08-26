"""Live Instagram extraction test runner (Phase 3).

This is the ONLY entry point that intentionally hits Instagram. It is kept out
of the pytest suite on purpose: `pytest` must stay fully offline. Run this by
hand to gather real-world extraction data for the eventual 20-URL benchmark.

    python scripts/test_extraction.py "https://www.instagram.com/reel/XXXX/"
    python scripts/test_extraction.py url1 url2 url3
    python scripts/test_extraction.py --keep-media "https://www.instagram.com/p/XXXX/"
    python scripts/test_extraction.py --record "https://www.instagram.com/reel/XXXX/"

A failed URL is a legitimate result - it is recorded, not hidden. Each row
captures: url, success, media_downloaded, caption_extracted, media_type,
failure_category, failure_message, download_seconds.

With --record, one JSON line per URL is appended to
output/extraction_benchmark.jsonl for later analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config  # noqa: E402
from extractor import (  # noqa: E402
    ExtractionMode,
    ExtractionOptions,
    YtDlpExtractor,
    validate_instagram_url,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/test_extraction.py",
        description="Run real Instagram extractions and record the results.",
    )
    parser.add_argument("urls", nargs="+", help="One or more Instagram URLs.")
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Retain downloaded media (temp run dirs are not deleted).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append each result as JSON to output/extraction_benchmark.jsonl.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser


def _benchmark_row(url: str, result) -> dict:
    return {
        "url": url,
        "success": result.success,
        "media_downloaded": result.media_downloaded,
        "caption_extracted": result.caption_extracted,
        "media_type": result.media_type,
        "media_type_detected": result.media_type_detected,
        "failure_category": result.failure_category,
        "failure_message": result.failure_reason,
        "download_seconds": result.download_seconds,
        "hashtag_count": len(result.hashtags),
        "extraction_mode": result.extraction_mode,
        "cookie_file_configured": result.cookie_file_configured,
    }


def _print_row(row: dict) -> None:
    status = "OK  " if row["success"] else "FAIL"
    print(f"[{status}] {row['url']}")
    print(f"        media_downloaded : {row['media_downloaded']}")
    print(f"        caption_extracted: {row['caption_extracted']}")
    print(f"        media_type       : {row['media_type']} "
          f"(detected={row['media_type_detected']})")
    print(f"        download_seconds : {row['download_seconds']}")
    if not row["success"] or row["failure_category"]:
        print(f"        failure_category : {row['failure_category']}")
        print(f"        failure_message  : {row['failure_message']}")
    print()


def _cleanup(settings, result) -> None:
    if not result.run_id:
        return
    import shutil

    run_path = settings.temp_dir / result.run_id
    if run_path.exists():
        shutil.rmtree(run_path, ignore_errors=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    settings = app_config.get_config()
    logger = app_config.configure_logging("DEBUG" if args.verbose else settings.log_level)
    settings.ensure_directories()

    options = ExtractionOptions(
        mode=ExtractionMode.from_string(settings.extraction_mode),
        cookies_from_browser=settings.cookies_from_browser or None,
        cookie_file=settings.cookie_file or None,
        keep_media=args.keep_media,
    )
    extractor = YtDlpExtractor(temp_dir=settings.temp_dir)

    # Report the auth configuration WITHOUT ever revealing cookie contents.
    print(f"Extraction mode: {options.mode.value}")
    print(f"Cookie file configured: {'yes' if options.cookie_file_configured else 'no'}")
    print()

    record_path = settings.output_dir / "extraction_benchmark.jsonl"
    rows: list[dict] = []
    any_success = False

    for url in args.urls:
        logger.info("=== extracting: %s ===", url)
        validation = validate_instagram_url(url)
        if not validation.valid:
            row = {
                "url": url,
                "success": False,
                "media_downloaded": False,
                "caption_extracted": False,
                "media_type": "unknown",
                "media_type_detected": False,
                "failure_category": "URL_VALIDATION",
                "failure_message": validation.error_message,
                "download_seconds": None,
                "hashtag_count": 0,
            }
        else:
            result = extractor.extract(validation, options)
            row = _benchmark_row(url, result)
            any_success = any_success or result.success
            if not args.keep_media:
                _cleanup(settings, result)

        rows.append(row)
        _print_row(row)

        if args.record:
            with record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    total = len(rows)
    ok = sum(1 for r in rows if r["success"])
    print(f"Summary: {ok}/{total} succeeded.")
    if args.record:
        print(f"Recorded to: {record_path}")

    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
