"""Benchmark runner for the Instagram Content Analyzer pipeline (Phase 9).

Executes the complete pipeline over a list of test URLs, records run-level
results and stage metrics, tracks cleanup, and compiles an aggregate PRD
feasibility evaluation report.

Usage:
    python scripts/run_benchmark.py --input benchmark/urls.example.txt
    python scripts/run_benchmark.py --input benchmark/urls.txt --ground-truth benchmark/ground_truth.csv
    python scripts/run_benchmark.py --limit 5 --output output/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as app_config  # noqa: E402
from processor.benchmark import (  # noqa: E402
    BenchmarkEvaluation,
    BenchmarkRecord,
    evaluate_benchmark,
    load_ground_truth,
)
from processor.output import save_run_result  # noqa: E402
from processor.run_result import DEFAULT_MODEL_VERSIONS, run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reelfox.benchmark_runner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/run_benchmark.py",
        description="Run end-to-end benchmark on Instagram URLs (Phase 9).",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="benchmark/urls.example.txt",
        help="Path to text file with URLs (one per line, # for comments).",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        default=None,
        help="Optional path to ground truth CSV for manual evaluation scoring.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/benchmark_report.json",
        help="Path to save aggregate benchmark report JSON (default: output/benchmark_report.json).",
    )
    parser.add_argument(
        "--runs-dir",
        default="output/runs",
        help="Directory to save individual per-run JSON records (default: output/runs).",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of URLs to process from input file.",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Retain downloaded temporary media instead of cleaning up.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    return parser


def load_urls(input_path: Path | str, limit: Optional[int] = None) -> List[str]:
    """Read URLs from text file, skipping comments and blank lines."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input URL file not found: {p}")

    urls: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            urls.append(stripped)
            if limit is not None and len(urls) >= limit:
                break
    return urls


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = app_config.Config.load()
    urls = load_urls(args.input, limit=args.limit)

    if not urls:
        logger.error("No valid URLs found in %s", args.input)
        return 1

    gt_records = {}
    if args.ground_truth:
        gt_records = load_ground_truth(args.ground_truth)
        logger.info("Loaded %d ground-truth records from %s", len(gt_records), args.ground_truth)

    print("=" * 65)
    print("REELFOX END-TO-END BENCHMARK RUNNER (Phase 9)")
    print("=" * 65)
    print(f"Target URLs Count : {len(urls)}")
    print(f"Input File        : {args.input}")
    print(f"Runs Directory    : {args.runs_dir}")
    print(f"Report Output     : {args.output}")
    print(f"Keep Media        : {'YES' if args.keep_media else 'NO (auto-cleanup)'}")
    print("=" * 65)

    benchmark_records: List[BenchmarkRecord] = []
    t_suite_start = time.perf_counter()

    for idx, url in enumerate(urls, 1):
        test_id = f"T{idx:02d}"
        print(f"\n[{idx}/{len(urls)}] Processing {test_id}: {url}")

        t0 = time.perf_counter()
        run_res = run_pipeline(
            url=url,
            config=settings,
            keep_media=args.keep_media,
        )
        elapsed = time.perf_counter() - t0

        # Save run record JSON
        saved_run_path = save_run_result(run_res, output_dir=args.runs_dir)

        # Map to benchmark record
        gt_entry = gt_records.get(test_id)
        bench_rec = BenchmarkRecord.from_run_result(
            run_result=run_res,
            test_id=test_id,
            ground_truth=gt_entry,
        )
        benchmark_records.append(bench_rec)

        status_str = "PASS" if run_res.overall_success else "FAIL"
        print(f"  -> Result: {status_str} ({elapsed:.1f}s) | Saved run record: {saved_run_path.name}")
        if not run_res.overall_success:
            print(f"     Failure: [{run_res.failure_stage}] {run_res.failure_message}")

    # Compile aggregate evaluation
    total_time = time.perf_counter() - t_suite_start
    evaluation: BenchmarkEvaluation = evaluate_benchmark(benchmark_records)

    # Save aggregate report JSON
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_urls_processed": len(urls),
        "total_benchmark_time_seconds": round(total_time, 2),
        "model_versions": dict(DEFAULT_MODEL_VERSIONS),
        "evaluation": evaluation.as_dict(),
        "records": [r.as_dict() for r in benchmark_records],
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("AGGREGATE BENCHMARK REPORT")
    print("=" * 65)
    print(f"Decision                      : {evaluation.decision.value}")
    print(f"Decision Rationale            : {evaluation.decision_rationale}")
    print("-" * 65)
    print(f"Extraction Success Rate       : {evaluation.extraction_success_rate:.1%} (Target >= 90%)")
    print(f"Caption Retrieval Rate        : {evaluation.caption_retrieval_rate:.1%} (Target >= 95%)")
    print(f"Speech Availability Rate      : {evaluation.speech_availability_rate:.1%}")
    print(f"Vision Success Rate           : {evaluation.vision_success_rate:.1%}")
    print(f"OCR Detection Rate            : {evaluation.ocr_detection_rate:.1%}")
    print(f"Synthesis Success Rate        : {evaluation.synthesis_success_rate:.1%}")
    print(f"Cleanup Success Rate          : {evaluation.cleanup_success_rate:.1%} (Target = 100%)")
    if evaluation.overall_useful_analysis_rate is not None:
        print(f"Overall Useful Analysis Rate  : {evaluation.overall_useful_analysis_rate:.1%} (Target >= 85%)")
    else:
        print("Overall Useful Analysis Rate  : PENDING MANUAL HUMAN EVALUATION")
    print("-" * 65)
    print(f"Average Run Latency           : {evaluation.average_processing_time_seconds:.1f}s")
    print(f"Total Benchmark Time          : {total_time:.1f}s")
    print(f"Aggregate Report Saved To     : {out_file.resolve()}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
