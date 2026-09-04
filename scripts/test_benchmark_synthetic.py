"""Synthetic test verification for Phase 9 benchmark infrastructure.

Validates the full benchmark workflow end-to-end using purely synthetic/mocked
pipeline components without downloading models or contacting Instagram.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extractor.models import ExtractionResult
from processor.benchmark import (
    BenchmarkDecision,
    BenchmarkRecord,
    evaluate_benchmark,
    load_ground_truth,
)
from processor.models import OCRResult, OCRTextBlock, SpeechResult, VisionResult
from processor.output import save_run_result
from processor.run_result import run_pipeline
from processor.synthesis_models import MultimodalAnalysisResult


def run_synthetic_benchmark():
    print("=" * 65)
    print("RUNNING SYNTHETIC BENCHMARK INFRASTRUCTURE VERIFICATION")
    print("=" * 65)

    # 1. Prepare synthetic mock runners
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = ExtractionResult(
        success=True,
        source_url="https://www.instagram.com/reel/synthetic_01/",
        media_downloaded=True,
        media_path="temp/synthetic/vid.mp4",
        caption_extracted=True,
        caption="Synthetic caption for benchmark verification #test",
        media_type="reel",
        download_seconds=0.5,
    )

    mock_speech = SpeechResult(
        success=True,
        audio_present=True,
        speech_present=True,
        transcript="Synthetic audio transcript of machine learning concepts.",
        duration_seconds=10.0,
        transcription_time_seconds=0.8,
    )

    mock_vision = VisionResult(
        success=True,
        frames_analyzed=6,
        observations=["Synthetic visual observation of screen."],
        model_name="SmolVLM-256M-Instruct",
    )

    mock_ocr = OCRResult(
        success=True,
        frames_analyzed=6,
        text_detected=True,
        combined_text="SYNTHETIC OCR TEXT 100",
        text_blocks=[OCRTextBlock(text="SYNTHETIC OCR TEXT 100", confidence=0.99)],
        model_name_or_engine="rapidocr",
    )

    mock_synth = MultimodalAnalysisResult(
        success=True,
        summary="Synthetic summary of ML interview questions.",
        key_points=["Point 1", "Point 2"],
        core_takeaway="Serving costs are important.",
        confidence=0.95,
        model_name="Qwen/Qwen2.5-3B-Instruct",
    )

    # 2. Run pipeline
    print("1. Running synthetic run_pipeline()...")
    run_res = run_pipeline(
        url="https://www.instagram.com/reel/synthetic_01/",
        extractor=mock_extractor,
        speech_runner=lambda p: mock_speech,
        vision_runner=lambda p: mock_vision,
        ocr_runner=lambda p: mock_ocr,
        synthesis_runner=lambda **kwargs: mock_synth,
    )
    assert run_res.overall_success is True
    print(f"   [OK] Pipeline completed (run_id: {run_res.run_id})")

    # 3. Save run result
    print("2. Saving run result to output/runs/...")
    saved_path = save_run_result(run_res, output_dir="output/runs")
    assert saved_path.exists()
    print(f"   [OK] Saved run file: {saved_path}")

    # 4. Convert to benchmark record
    print("3. Creating BenchmarkRecord...")
    bench_rec = BenchmarkRecord.from_run_result(run_res, test_id="T01")
    rec_dict = bench_rec.as_dict()
    assert rec_dict["extraction_success"] is True
    assert rec_dict["summary_quality"] is None  # Human eval null by default
    print("   [OK] BenchmarkRecord created and validated against schema")

    # 5. Load ground truth
    print("4. Testing ground truth loading from benchmark/ground_truth.example.csv...")
    gt = load_ground_truth("benchmark/ground_truth.example.csv")
    assert "T01" in gt
    bench_rec_evaluated = BenchmarkRecord.from_run_result(run_res, test_id="T01", ground_truth=gt["T01"])
    assert bench_rec_evaluated.summary_quality == 0.95
    print("   [OK] Ground truth evaluation successfully merged")

    # 6. Evaluate benchmark
    print("5. Evaluating benchmark decisions...")
    eval_pending = evaluate_benchmark([bench_rec])
    assert eval_pending.decision == BenchmarkDecision.PENDING_EVALUATION
    print(f"   [OK] Unevaluated decision: {eval_pending.decision.value} (Expected)")

    eval_pass = evaluate_benchmark([bench_rec_evaluated])
    assert eval_pass.decision == BenchmarkDecision.PASS
    print(f"   [OK] Evaluated decision: {eval_pass.decision.value} (Expected)")

    # 7. Save synthetic report
    out_report = Path("output/synthetic_benchmark_report.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "VALIDATED",
                "evaluation": eval_pass.as_dict(),
                "sample_record": bench_rec_evaluated.as_dict(),
            },
            f,
            indent=2,
        )
    print(f"   [OK] Saved synthetic report to: {out_report}")
    print("\nALL SYNTHETIC BENCHMARK INFRASTRUCTURE CHECKS PASSED.")


if __name__ == "__main__":
    run_synthetic_benchmark()
