
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config
from processor.synthesis import LocalQwenSynthesizer
from processor.synthesis_models import MultimodalEvidence

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='scripts/test_local_synthesis.py',
        description='Run local multimodal synthesis with Qwen2.5-3B-Instruct at zero cost.',
    )
    parser.add_argument(
        'url',
        nargs='?',
        default='https://www.instagram.com/p/DcWXVZlMwOB/',
        help='Instagram URL to analyze.',
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run full live download and extraction pipeline instead of re-using output/multimodal_result.json.',
    )
    parser.add_argument(
        '--output-file',
        default='output/local_multimodal_result.json',
        help='Destination JSON path (default: output/local_multimodal_result.json).',
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = app_config.Config.load()
    print('=' * 60)
    print('LOCAL MULTIMODAL SYNTHESIS TEST (Qwen2.5-3B-Instruct)')
    print('=' * 60)
    print(f'Target URL: {args.url}')
    print(f'Synthesis Backend: {settings.synthesis_backend}')
    print(f'Synthesis Model: {settings.synthesis_model}')
    print(f'Synthesis Device: {settings.synthesis_device}')
    print('Zero Cost: YES (100% local CPU inference)')

    evidence = None

    ref_file = Path('output/multimodal_result.json')
    if not args.live and ref_file.exists():
        try:
            with open(ref_file, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            extracted = saved.get('extracted_evidence', {})
            evidence = MultimodalEvidence(
                source_url=args.url,
                metadata={
                    'caption': extracted.get('caption'),
                    'caption_present': bool(extracted.get('caption')),
                    'hashtags': extracted.get('hashtags') or [],
                },
                speech={
                    'available': bool(extracted.get('speech_transcript')),
                    'speech_present': bool(extracted.get('speech_transcript')),
                    'transcript': extracted.get('speech_transcript'),
                },
                vision={
                    'available': bool(extracted.get('vision_observations')),
                    'observations': extracted.get('vision_observations') or [],
                },
                ocr={
                    'available': bool(extracted.get('ocr_detected_text')),
                    'text_detected': bool(extracted.get('ocr_detected_text')),
                    'combined_text': extracted.get('ocr_detected_text'),
                },
            )
            print(f'\n[OK] Loaded extracted evidence from {ref_file}')
        except Exception as exc:
            print(f'[WARN] Failed to load saved evidence: {exc}. Falling back to live pipeline.')
            evidence = None

    if evidence is None:
        print('\nRunning live extraction pipeline...')
        from extractor import ExtractionOptions, YtDlpExtractor, validate_instagram_url
        from processor.pipeline import process_ocr, process_speech, process_vision

        valid_url = validate_instagram_url(args.url)
        extractor = YtDlpExtractor(
            temp_dir=settings.temp_dir,
            output_dir=settings.output_dir,
            cookies_from_browser=settings.cookies_from_browser,
            cookie_file=settings.cookie_file,
        )
        extraction_result = extractor.extract(valid_url, ExtractionOptions(cleanup_media=False))
        media_path = extraction_result.media_path

        speech_res = process_speech(media_path) if extraction_result.has_audio else None
        vision_res = process_vision(media_path, is_video=(extraction_result.media_type == 'video'))
        ocr_res = process_ocr(media_path, is_video=(extraction_result.media_type == 'video'))

        evidence = MultimodalEvidence.from_results(
            extraction_result=extraction_result,
            speech_result=speech_res,
            vision_result=vision_res,
            ocr_result=ocr_res,
        )

    print('\nInitializing LocalQwenSynthesizer...')
    t0 = time.perf_counter()
    synthesizer = LocalQwenSynthesizer(
        model_id=settings.synthesis_model,
        device=settings.synthesis_device,
    )

    print('Running local synthesis inference...')
    result = synthesizer.synthesize(evidence)
    elapsed = time.perf_counter() - t0

    print('\n' + '=' * 60)
    print('SYNTHESIS RESULT')
    print('=' * 60)
    print('Success: YES' if result.success else 'Success: NO')
    if not result.success:
        print(f'Failure Category: {result.failure_category}')
        print(f'Failure Message: {result.failure_message}')
        return 1

    print(f'\n--- Summary ---\n{result.summary}')
    print('\n--- Key Points ---')
    for i, pt in enumerate(result.key_points, 1):
        print(f'  {i}. {pt}')
    print(f'\n--- Core Takeaway ---\n{result.core_takeaway}')
    if result.relevant_context:
        print(f'\n--- Relevant Context ---\n{result.relevant_context}')
    print(f'\n--- Confidence ---\n{result.confidence:.2f}')

    print('\n--- Evidence Used ---')
    for k, v in sorted(result.evidence_used.items()):
        val_str = 'true' if v else 'false'
        print(f'  {k}: {val_str}')

    print('\n--- Metrics ---')
    print(f'  Model: {result.model_name}')
    if result.request_latency_seconds:
        print(f'  Inference Latency: {result.request_latency_seconds:.2f}s')
    if result.processing_time_seconds:
        print(f'  Total Processing Time: {result.processing_time_seconds:.2f}s')
    print(f'  Prompt Tokens: {result.prompt_tokens}')
    print(f'  Completion Tokens: {result.completion_tokens}')
    print(f'  Total Tokens: {result.total_tokens}')

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        'source_url': args.url,
        'media_type': 'video',
        'synthesis': result.as_dict(),
        'evidence': evidence.as_dict(),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f'\nSaved structured local synthesis result to: {out_path.resolve()}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
