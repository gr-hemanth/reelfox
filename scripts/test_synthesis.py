"""Live Multimodal Synthesis integration test runner (Phase 8).

This script performs real end-to-end multimodal synthesis using TokenRouter
and the free GLM model (z-ai/glm-5.3-free).

It:
1. Loads TOKENROUTER_API_KEY from environment/config (NEVER logs/prints it).
2. Extracts the Instagram post/reel (metadata, caption, media).
3. Executes speech recognition (ASR), vision understanding (VLM), and OCR.
4. Aggregates all structured textual evidence into MultimodalEvidence.
5. Sends evidence to TokenRouter (z-ai/glm-5.3-free) for structured JSON synthesis.
6. Validates strict schema and prints formatted results.
7. Saves the output to output/multimodal_result.json.
8. Exits with 0 on genuine success, non-zero on failure.

Usage:
    python scripts/test_synthesis.py
    python scripts/test_synthesis.py "https://www.instagram.com/p/DcWXVZlMwOB/"
    python scripts/test_synthesis.py --keep-media
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
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
from processor.pipeline import (  # noqa: E402
    process_ocr,
    process_speech,
    process_synthesis,
    process_vision,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/test_synthesis.py",
        description="Run live Multimodal Synthesis on Instagram content (Phase 8).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.instagram.com/p/DcWXVZlMwOB/",
        help="Instagram URL to analyze (default: https://www.instagram.com/p/DcWXVZlMwOB/).",
    )
    parser.add_argument(
        "--output-file",
        default="output/multimodal_result.json",
        help="Path to save final structured multimodal JSON (default: output/multimodal_result.json).",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Retain downloaded media files after processing.",
    )
    parser.add_argument(
        "--skip-speech",
        action="store_true",
        help="Skip speech recognition stage.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision understanding stage.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR on-screen text stage.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum keyframes to sample for vision & OCR.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = app_config.get_config()
    log_level = "DEBUG" if args.verbose else settings.log_level
    logger = app_config.configure_logging(log_level)

    print("=" * 60)
    print("REELFOX - Phase 8: Multimodal Synthesis Live Test")
    print("=" * 60)
    print(f"Target URL: {args.url}")
    print(f"Model: {settings.synthesis_model}")
    print(f"Endpoint: {settings.synthesis_endpoint}")

    # Check API key presence without printing
    api_key = settings.tokenrouter_api_key
    if not api_key:
        print("\nERROR: TOKENROUTER_API_KEY is not set in environment or .env file.")
        print("Set TOKENROUTER_API_KEY in .env before running this test.")
        return 1

    print("TOKENROUTER_API_KEY: [Configured]")
    print()

    # Step 1: URL Validation
    print(">>> Step 1: Validating Instagram URL...")
    validation = validate_instagram_url(args.url)
    if not validation.valid:
        print(f"Validation failed: {validation.error_code} - {validation.error_message}")
        return 2
    print(f"  Valid: YES (hint={validation.content_type_hint})")

    # Step 2: Extraction
    print("\n>>> Step 2: Extracting Instagram Media...")
    options = ExtractionOptions(
        mode=ExtractionMode.from_string(settings.extraction_mode),
        cookies_from_browser=settings.cookies_from_browser or None,
        cookie_file=settings.cookie_file or None,
        keep_media=True,
    )
    extractor = YtDlpExtractor(temp_dir=settings.temp_dir)
    extraction_result = extractor.extract(validation, options)

    if not extraction_result.success:
        print(f"Extraction failed: {extraction_result.failure_category} - {extraction_result.failure_reason}")
        return 3

    print(f"  Extraction: PASS (media_type={extraction_result.media_type})")
    print(f"  Caption: {extraction_result.caption[:80] if extraction_result.caption else '(none)'}...")
    print(f"  Media path: {extraction_result.media_path}")

    speech_result = None
    vision_result = None
    ocr_result = None

    try:
        # Step 3: Speech Recognition (ASR)
        if (
            not args.skip_speech
            and extraction_result.media_type in ("reel", "video")
            and extraction_result.media_path
        ):
            print("\n>>> Step 3: Processing Speech / Audio (ASR)...")
            speech_result = process_speech(extraction_result.media_path)
            print(f"  Speech: {'PASS' if speech_result.success else 'FAIL'}")
            print(f"  Speech present: {speech_result.speech_present}")
            if speech_result.transcript:
                print(f"  Transcript: {speech_result.transcript[:100]}...")
        else:
            reason = "(--skip-speech)" if args.skip_speech else "(not a video)"
            print(f"\n>>> Step 3: Speech skipped {reason}.")

        # Step 4: Vision Understanding (VLM)
        if not args.skip_vision and extraction_result.media_path:
            print("\n>>> Step 4: Processing Vision Understanding (VLM)...")
            vision_result = process_vision(
                extraction_result.media_path,
                max_frames=args.max_frames,
            )
            print(f"  Vision: {'PASS' if vision_result.success else 'FAIL'}")
            print(f"  Frames analyzed: {vision_result.frames_analyzed}")
            if vision_result.observations:
                print(f"  Observations: {len(vision_result.observations)} frame summaries")
        else:
            print("\n>>> Step 4: Vision skipped (--skip-vision).")

        # Step 5: OCR / On-Screen Text
        if not args.skip_ocr and (extraction_result.media_path or extraction_result.media_files):
            print("\n>>> Step 5: Processing OCR / On-Screen Text...")
            if extraction_result.media_type == "carousel" and len(extraction_result.media_files) > 1:
                ocr_input = [m.path for m in extraction_result.media_files]
            else:
                ocr_input = extraction_result.media_path

            ocr_result = process_ocr(
                ocr_input,
                max_frames=args.max_frames,
            )
            print(f"  OCR: {'PASS' if ocr_result.success else 'FAIL'}")
            print(f"  Text detected: {ocr_result.text_detected}")
            if ocr_result.text_blocks:
                print(f"  Detected blocks: {len(ocr_result.text_blocks)}")
                for block in ocr_result.text_blocks[:3]:
                    print(f"    - {block.text}")
        else:
            print("\n>>> Step 5: OCR skipped (--skip-ocr).")

        # Step 6: Multimodal Synthesis
        print("\n>>> Step 6: Synthesizing Multimodal Evidence with GLM-5.3-free...")
        t_start = time.perf_counter()
        synthesis_result = process_synthesis(
            extraction=extraction_result,
            speech=speech_result,
            vision=vision_result,
            ocr=ocr_result,
            api_key=api_key,
            model=settings.synthesis_model,
            endpoint=settings.synthesis_endpoint,
        )
        total_time = time.perf_counter() - t_start

        print("\n" + "=" * 60)
        print("SYNTHESIS RESULT")
        print("=" * 60)
        print(f"Success: {'YES' if synthesis_result.success else 'NO'}")

        if not synthesis_result.success:
            print(f"Failure Category: {synthesis_result.failure_category}")
            print(f"Failure Message: {synthesis_result.failure_message}")
            return 4

        print(f"\n--- Summary ---\n{synthesis_result.summary}")
        print("\n--- Key Points ---")
        for i, pt in enumerate(synthesis_result.key_points, 1):
            print(f"  {i}. {pt}")
        print(f"\n--- Core Takeaway ---\n{synthesis_result.core_takeaway}")
        if synthesis_result.relevant_context:
            print(f"\n--- Relevant Context ---\n{synthesis_result.relevant_context}")
        print(f"\n--- Confidence ---\n{synthesis_result.confidence:.2f}")

        print("\n--- Evidence Used ---")
        for k, v in sorted(synthesis_result.evidence_used.items()):
            print(f"  {k}: {'true' if v else 'false'}")

        print("\n--- Metrics ---")
        print(f"  Model: {synthesis_result.model_name}")
        print(f"  Request Latency: {synthesis_result.request_latency_seconds:.3f}s" if synthesis_result.request_latency_seconds else "  Request Latency: N/A")
        print(f"  Total Processing Time: {synthesis_result.processing_time_seconds:.3f}s" if synthesis_result.processing_time_seconds else f"  Total Processing Time: {total_time:.3f}s")
        print(f"  Prompt Tokens: {synthesis_result.prompt_tokens}")
        print(f"  Completion Tokens: {synthesis_result.completion_tokens}")
        print(f"  Total Tokens: {synthesis_result.total_tokens}")

        # Step 7: Save JSON output
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "source_url": args.url,
            "media_type": extraction_result.media_type,
            "synthesis": synthesis_result.as_dict(),
            "extracted_evidence": {
                "caption": extraction_result.caption,
                "hashtags": extraction_result.hashtags,
                "speech_transcript": speech_result.transcript if speech_result else None,
                "vision_observations": vision_result.observations if vision_result else [],
                "ocr_detected_text": ocr_result.combined_text if ocr_result else None,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved structured synthesis result to: {out_path.resolve()}")

    finally:
        if not args.keep_media and extraction_result.run_id:
            run_dir = settings.temp_dir / extraction_result.run_id
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
                logger.debug("Cleaned up temp media directory: %s", run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
