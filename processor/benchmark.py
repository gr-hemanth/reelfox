"""Benchmark data structures, evaluation targets, and aggregation logic (Phase 9).

Encodes official PRD targets:
- Extraction Success: >= 90%
- Caption Retrieval: >= 95% (on successfully extracted captions)
- Speech Quality: >= 85% qualitative accuracy
- OCR Quality: >= 85% qualitative accuracy
- Vision Quality: >= 85% qualitative accuracy
- Overall Useful Analysis: >= 85% (MANUAL human evaluation only)
- Cleanup Success: 100%

Enforces strict separation between mechanical model success and human-judged usefulness.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from processor.run_result import RunResult

# Official PRD Targets
TARGET_EXTRACTION_SUCCESS_RATE = 0.90
TARGET_CAPTION_RETRIEVAL_RATE = 0.95
TARGET_SPEECH_QUALITY_RATE = 0.85
TARGET_OCR_QUALITY_RATE = 0.85
TARGET_VISION_QUALITY_RATE = 0.85
TARGET_OVERALL_USEFUL_ANALYSIS_RATE = 0.85
TARGET_CLEANUP_SUCCESS_RATE = 1.00


class BenchmarkDecision(str, Enum):
    """Feasibility gate decision categories."""

    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    PENDING_EVALUATION = "PENDING_EVALUATION"


@dataclass
class BenchmarkRecord:
    """Individual benchmark record for a single test URL.

    Quality fields are null by default and populated ONLY via human ground-truth evaluation.
    They are never auto-calculated from model confidence.
    """

    test_id: str
    source_url: str
    content_type: str

    extraction_success: bool
    media_retrieved: bool
    caption_retrieved: bool

    speech_present: bool
    transcript_quality: Optional[float] = None

    ocr_quality: Optional[float] = None

    visual_understanding_quality: Optional[float] = None

    summary_quality: Optional[float] = None
    core_takeaway_quality: Optional[float] = None

    confidence: float = 0.0

    processing_time_seconds: float = 0.0

    cleanup_success: bool = True

    failure_reason: Optional[str] = None

    # Auxiliary stage success tracking
    vision_success: bool = False
    ocr_detected: bool = False
    synthesis_success: bool = False
    model_versions: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return clean dictionary representation conforming strictly to benchmark schema."""
        return {
            "test_id": self.test_id,
            "source_url": self.source_url,
            "content_type": self.content_type,
            "extraction_success": self.extraction_success,
            "media_retrieved": self.media_retrieved,
            "caption_retrieved": self.caption_retrieved,
            "speech_present": self.speech_present,
            "transcript_quality": self.transcript_quality,
            "ocr_quality": self.ocr_quality,
            "visual_understanding_quality": self.visual_understanding_quality,
            "summary_quality": self.summary_quality,
            "core_takeaway_quality": self.core_takeaway_quality,
            "confidence": round(self.confidence, 2),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "cleanup_success": self.cleanup_success,
            "failure_reason": self.failure_reason,
            "_stage_status": {
                "vision_success": self.vision_success,
                "ocr_detected": self.ocr_detected,
                "synthesis_success": self.synthesis_success,
            },
            "_model_versions": dict(self.model_versions),
        }

    @classmethod
    def from_run_result(
        cls,
        run_result: RunResult,
        test_id: str = "",
        ground_truth: Optional[GroundTruthRecord] = None,
    ) -> BenchmarkRecord:
        """Construct a BenchmarkRecord from a complete RunResult."""
        ext_metrics = run_result.stage_metrics.get("extraction", {})
        speech_metrics = run_result.stage_metrics.get("speech", {})
        vision_metrics = run_result.stage_metrics.get("vision", {})
        ocr_metrics = run_result.stage_metrics.get("ocr", {})
        synth_metrics = run_result.stage_metrics.get("synthesis", {})

        confidence = 0.0
        if run_result.synthesis:
            confidence = float(run_result.synthesis.get("confidence", 0.0) or 0.0)

        failure_reason = None
        if not run_result.overall_success:
            failure_reason = run_result.failure_message or run_result.failure_category or "Run failed"

        rec = cls(
            test_id=test_id or run_result.run_id,
            source_url=run_result.source_url,
            content_type=run_result.content_type,
            extraction_success=bool(ext_metrics.get("success", False)),
            media_retrieved=bool(ext_metrics.get("media_downloaded", False)),
            caption_retrieved=bool(ext_metrics.get("caption_extracted", False)),
            speech_present=bool(speech_metrics.get("speech_present", False)),
            transcript_quality=None,
            ocr_quality=None,
            visual_understanding_quality=None,
            summary_quality=None,
            core_takeaway_quality=None,
            confidence=confidence,
            processing_time_seconds=run_result.total_execution_seconds,
            cleanup_success=run_result.cleanup_success,
            failure_reason=failure_reason,
            vision_success=bool(vision_metrics.get("success", False)),
            ocr_detected=bool(ocr_metrics.get("text_detected", False)),
            synthesis_success=bool(synth_metrics.get("success", False)),
            model_versions=run_result.model_versions,
        )

        if ground_truth is not None:
            # Attach human evaluations if provided in ground truth
            rec.transcript_quality = ground_truth.transcript_quality
            rec.ocr_quality = ground_truth.ocr_quality
            rec.visual_understanding_quality = ground_truth.visual_understanding_quality
            rec.summary_quality = ground_truth.summary_quality
            rec.core_takeaway_quality = ground_truth.core_takeaway_quality

        return rec


@dataclass
class GroundTruthRecord:
    """Manual ground-truth evaluation record for comparing against pipeline outputs."""

    test_id: str
    actual_content_type: str = "unknown"
    actual_has_speech: bool = False
    actual_important_points: str = ""
    actual_ocr_text: str = ""
    actual_visual_facts: str = ""
    actual_summary: str = ""
    actual_core_takeaway: str = ""
    notes: str = ""

    # Human evaluated qualitative scores (0.0 to 1.0 or None)
    transcript_quality: Optional[float] = None
    ocr_quality: Optional[float] = None
    visual_understanding_quality: Optional[float] = None
    summary_quality: Optional[float] = None
    core_takeaway_quality: Optional[float] = None


def load_ground_truth(csv_path: Union[Path, str]) -> Dict[str, GroundTruthRecord]:
    """Load ground-truth records from a local CSV file."""
    path = Path(csv_path)
    if not path.exists():
        return {}

    records: Dict[str, GroundTruthRecord] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("test_id", "").strip()
            if not tid:
                continue

            def _parse_float(key: str) -> Optional[float]:
                val = row.get(key, "").strip()
                try:
                    return float(val) if val else None
                except ValueError:
                    return None

            records[tid] = GroundTruthRecord(
                test_id=tid,
                actual_content_type=row.get("actual_content_type", "").strip(),
                actual_has_speech=row.get("actual_has_speech", "").lower() in ("true", "1", "yes"),
                actual_important_points=row.get("actual_important_points", "").strip(),
                actual_ocr_text=row.get("actual_ocr_text", "").strip(),
                actual_visual_facts=row.get("actual_visual_facts", "").strip(),
                actual_summary=row.get("actual_summary", "").strip(),
                actual_core_takeaway=row.get("actual_core_takeaway", "").strip(),
                notes=row.get("notes", "").strip(),
                transcript_quality=_parse_float("transcript_quality"),
                ocr_quality=_parse_float("ocr_quality"),
                visual_understanding_quality=_parse_float("visual_understanding_quality"),
                summary_quality=_parse_float("summary_quality"),
                core_takeaway_quality=_parse_float("core_takeaway_quality"),
            )
    return records


@dataclass
class BenchmarkEvaluation:
    """Consolidated aggregate metrics and PRD feasibility decision."""

    total_runs: int
    decision: BenchmarkDecision
    decision_rationale: str

    # Quantitative rates
    extraction_success_rate: float
    caption_retrieval_rate: float
    speech_availability_rate: float
    ocr_detection_rate: float
    vision_success_rate: float
    synthesis_success_rate: float
    cleanup_success_rate: float

    # Evaluated qualitative rates (None if human evaluation pending)
    overall_useful_analysis_rate: Optional[float] = None
    evaluated_speech_quality_rate: Optional[float] = None
    evaluated_ocr_quality_rate: Optional[float] = None
    evaluated_vision_quality_rate: Optional[float] = None

    # Target compliance
    target_status: Dict[str, bool] = field(default_factory=dict)
    average_processing_time_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "decision": self.decision.value,
            "decision_rationale": self.decision_rationale,
            "rates": {
                "extraction_success_rate": round(self.extraction_success_rate, 4),
                "caption_retrieval_rate": round(self.caption_retrieval_rate, 4),
                "speech_availability_rate": round(self.speech_availability_rate, 4),
                "ocr_detection_rate": round(self.ocr_detection_rate, 4),
                "vision_success_rate": round(self.vision_success_rate, 4),
                "synthesis_success_rate": round(self.synthesis_success_rate, 4),
                "cleanup_success_rate": round(self.cleanup_success_rate, 4),
                "overall_useful_analysis_rate": round(self.overall_useful_analysis_rate, 4) if self.overall_useful_analysis_rate is not None else None,
            },
            "qualitative_evaluation": {
                "human_evaluation_completed": self.overall_useful_analysis_rate is not None,
                "overall_useful_analysis_rate": self.overall_useful_analysis_rate,
                "evaluated_speech_quality_rate": self.evaluated_speech_quality_rate,
                "evaluated_ocr_quality_rate": self.evaluated_ocr_quality_rate,
                "evaluated_vision_quality_rate": self.evaluated_vision_quality_rate,
            },
            "target_status": dict(self.target_status),
            "average_processing_time_seconds": round(self.average_processing_time_seconds, 2),
        }


def evaluate_benchmark(records: List[BenchmarkRecord]) -> BenchmarkEvaluation:
    """Calculate aggregate rates and evaluate against PRD feasibility criteria."""
    total = len(records)
    if total == 0:
        return BenchmarkEvaluation(
            total_runs=0,
            decision=BenchmarkDecision.FAIL,
            decision_rationale="No benchmark records provided.",
            extraction_success_rate=0.0,
            caption_retrieval_rate=0.0,
            speech_availability_rate=0.0,
            ocr_detection_rate=0.0,
            vision_success_rate=0.0,
            synthesis_success_rate=0.0,
            cleanup_success_rate=0.0,
            target_status={"all_targets_met": False},
        )

    # 1. Mechanical pipeline rates
    ext_success_count = sum(1 for r in records if r.extraction_success)
    caption_count = sum(1 for r in records if r.caption_retrieved)
    speech_count = sum(1 for r in records if r.speech_present)
    ocr_count = sum(1 for r in records if r.ocr_detected)
    vision_count = sum(1 for r in records if r.vision_success)
    synthesis_count = sum(1 for r in records if r.synthesis_success)
    cleanup_count = sum(1 for r in records if r.cleanup_success)

    ext_rate = ext_success_count / total
    caption_rate = caption_count / max(1, ext_success_count)
    speech_rate = speech_count / total
    ocr_rate = ocr_count / total
    vision_rate = vision_count / total
    synthesis_rate = synthesis_count / total
    cleanup_rate = cleanup_count / total

    avg_time = sum(r.processing_time_seconds for r in records) / total

    # 2. Qualitative rates (from manual ground-truth evaluations ONLY)
    evaluated_useful = [
        r for r in records
        if r.summary_quality is not None and r.core_takeaway_quality is not None
    ]

    useful_rate: Optional[float] = None
    speech_qual_rate: Optional[float] = None
    ocr_qual_rate: Optional[float] = None
    vision_qual_rate: Optional[float] = None

    if evaluated_useful:
        # A run is qualitatively useful if summary >= 0.85 and takeaway >= 0.85
        useful_count = sum(
            1 for r in evaluated_useful
            if (r.summary_quality or 0.0) >= 0.85 and (r.core_takeaway_quality or 0.0) >= 0.85
        )
        useful_rate = useful_count / len(evaluated_useful)

        speech_evals = [r.transcript_quality for r in records if r.transcript_quality is not None]
        if speech_evals:
            speech_qual_rate = sum(speech_evals) / len(speech_evals)

        ocr_evals = [r.ocr_quality for r in records if r.ocr_quality is not None]
        if ocr_evals:
            ocr_qual_rate = sum(ocr_evals) / len(ocr_evals)

        vision_evals = [r.visual_understanding_quality for r in records if r.visual_understanding_quality is not None]
        if vision_evals:
            vision_qual_rate = sum(vision_evals) / len(vision_evals)

    # 3. PRD Targets Check
    target_status = {
        "extraction_target_met": ext_rate >= TARGET_EXTRACTION_SUCCESS_RATE,
        "caption_target_met": caption_rate >= TARGET_CAPTION_RETRIEVAL_RATE,
        "cleanup_target_met": cleanup_rate >= TARGET_CLEANUP_SUCCESS_RATE,
        "speech_quality_target_met": (speech_qual_rate is not None and speech_qual_rate >= TARGET_SPEECH_QUALITY_RATE),
        "ocr_quality_target_met": (ocr_qual_rate is not None and ocr_qual_rate >= TARGET_OCR_QUALITY_RATE),
        "vision_quality_target_met": (vision_qual_rate is not None and vision_qual_rate >= TARGET_VISION_QUALITY_RATE),
        "overall_useful_target_met": (useful_rate is not None and useful_rate >= TARGET_OVERALL_USEFUL_ANALYSIS_RATE),
    }

    # 4. Decision Logic
    if useful_rate is None:
        decision = BenchmarkDecision.PENDING_EVALUATION
        rationale = (
            "Mechanical pipeline execution finished, but PRD targets require human ground-truth "
            "evaluation. Synthesis success is distinct from factual usefulness."
        )
    else:
        ai_quality_ok = (
            useful_rate >= TARGET_OVERALL_USEFUL_ANALYSIS_RATE
            and (speech_qual_rate is None or speech_qual_rate >= TARGET_SPEECH_QUALITY_RATE)
            and (ocr_qual_rate is None or ocr_qual_rate >= TARGET_OCR_QUALITY_RATE)
            and (vision_qual_rate is None or vision_qual_rate >= TARGET_VISION_QUALITY_RATE)
            and cleanup_rate >= TARGET_CLEANUP_SUCCESS_RATE
        )

        if ai_quality_ok and ext_rate >= TARGET_EXTRACTION_SUCCESS_RATE:
            decision = BenchmarkDecision.PASS
            rationale = "All feasibility targets met: extraction >= 90%, cleanup = 100%, and AI quality >= 85%."
        elif ai_quality_ok and ext_rate < TARGET_EXTRACTION_SUCCESS_RATE:
            decision = BenchmarkDecision.CONDITIONAL_PASS
            rationale = (
                f"AI understanding is acceptable (useful rate = {useful_rate:.1%}), but extraction "
                f"success rate ({ext_rate:.1%}) fell below the 90% threshold."
            )
        else:
            decision = BenchmarkDecision.FAIL
            rationale = (
                f"Fundamental reliability criteria failed: extraction={ext_rate:.1%} (target 90%), "
                f"useful={useful_rate:.1%} (target 85%), cleanup={cleanup_rate:.1%} (target 100%)."
            )

    return BenchmarkEvaluation(
        total_runs=total,
        decision=decision,
        decision_rationale=rationale,
        extraction_success_rate=ext_rate,
        caption_retrieval_rate=caption_rate,
        speech_availability_rate=speech_rate,
        ocr_detection_rate=ocr_rate,
        vision_success_rate=vision_rate,
        synthesis_success_rate=synthesis_rate,
        cleanup_success_rate=cleanup_rate,
        overall_useful_analysis_rate=useful_rate,
        evaluated_speech_quality_rate=speech_qual_rate,
        evaluated_ocr_quality_rate=ocr_qual_rate,
        evaluated_vision_quality_rate=vision_qual_rate,
        target_status=target_status,
        average_processing_time_seconds=avg_time,
    )
