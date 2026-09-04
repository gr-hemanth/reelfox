"""Offline unit tests for Phase 9 End-to-End Benchmark Infrastructure.

100% network-free and mock-based. Tests all 16 required capabilities:
1. Complete successful run
2. Extraction failure handling
3. ASR failure handling
4. Vision failure handling
5. OCR failure handling
6. Synthesis failure handling
7. Partial evidence handling (e.g. image without audio)
8. Cleanup failure handling
9. Result serialization (JSON round-trip, no raw media)
10. Benchmark record serialization (strict schema match)
11. Aggregate metrics calculation
12. Human evaluation fields (null by default, populated from ground truth)
13. Feasibility pass/fail decision logic
14. Model-version tracking across outputs
15. Secret exclusion / credential leakage prevention
16. Raw media exclusion from JSON output
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extractor.models import ExtractionResult, MediaFile
from extractor.url_validator import ValidationResult
from processor.benchmark import (
    TARGET_CLEANUP_SUCCESS_RATE,
    TARGET_EXTRACTION_SUCCESS_RATE,
    TARGET_OVERALL_USEFUL_ANALYSIS_RATE,
    BenchmarkDecision,
    BenchmarkEvaluation,
    BenchmarkRecord,
    GroundTruthRecord,
    evaluate_benchmark,
    load_ground_truth,
)
from processor.models import (
    FrameObservation,
    OCRFailureCategory,
    OCRFrameResult,
    OCRResult,
    OCRTextBlock,
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    VisionFailureCategory,
    VisionResult,
)
from processor.output import sanitize_dict_for_output, save_run_result
from processor.run_result import (
    DEFAULT_MODEL_VERSIONS,
    RunResult,
    build_stage_metrics,
    run_pipeline,
)
from processor.synthesis_models import (
    MultimodalAnalysisResult,
    SynthesisFailureCategory,
)


@pytest.fixture
def mock_validation():
    return ValidationResult(
        valid=True,
        input_url="https://www.instagram.com/reel/valid_test_123/",
        normalized_url="https://www.instagram.com/reel/valid_test_123/",
        content_type_hint="reel",
    )


@pytest.fixture
def mock_extraction_success():
    return ExtractionResult(
        success=True,
        source_url="https://www.instagram.com/reel/valid_test_123/",
        normalized_url="https://www.instagram.com/reel/valid_test_123/",
        media_downloaded=True,
        media_path="temp/mock_run/test.mp4",
        caption_extracted=True,
        caption="Mock tech interview questions #ai #ml",
        hashtags=["#ai", "#ml"],
        media_type_detected=True,
        media_type="reel",
        download_seconds=1.25,
        run_id="mock_run",
    )


@pytest.fixture
def mock_speech_success():
    return SpeechResult(
        success=True,
        audio_present=True,
        speech_present=True,
        classification=SpeechClassification.SPEECH.value,
        transcript="How do you fine tune English models on Hindi data?",
        detected_language="en",
        duration_seconds=15.0,
        transcription_time_seconds=1.8,
        total_processing_seconds=2.0,
    )


@pytest.fixture
def mock_vision_success():
    return VisionResult(
        success=True,
        input_type="video",
        frames_analyzed=6,
        observations=["A presenter in front of a monitor showing code slides."],
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        total_processing_seconds=4.5,
    )


@pytest.fixture
def mock_ocr_success():
    return OCRResult(
        success=True,
        media_type="video",
        frames_analyzed=6,
        text_detected=True,
        combined_text="SARVAM AI INTERVIEW 2026",
        text_blocks=[OCRTextBlock(text="SARVAM AI INTERVIEW 2026", confidence=0.95)],
        model_name_or_engine="rapidocr",
        processing_time_seconds=0.85,
    )


@pytest.fixture
def mock_synthesis_success():
    return MultimodalAnalysisResult(
        success=True,
        summary="A walkthrough of ML interview questions asked at Sarvam.",
        key_points=["Focuses on model serving costs", "Mentions Hindi fine-tuning"],
        core_takeaway="Serving costs can surge when fine-tuning multilingual models.",
        relevant_context="ML Engineering Interview",
        confidence=0.92,
        evidence_used={"caption": True, "speech": True, "vision": True, "ocr": True},
        model_name="Qwen/Qwen2.5-3B-Instruct",
        processing_time_seconds=15.2,
        prompt_tokens=450,
        completion_tokens=120,
        total_tokens=570,
    )


# 1. Complete successful run
def test_complete_successful_run(
    mock_extraction_success,
    mock_speech_success,
    mock_vision_success,
    mock_ocr_success,
    mock_synthesis_success,
):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        speech_runner=lambda p: mock_speech_success,
        vision_runner=lambda p: mock_vision_success,
        ocr_runner=lambda p: mock_ocr_success,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    assert res.overall_success is True
    assert res.failure_stage is None
    assert res.content_type == "reel"
    assert res.extraction["media_downloaded"] is True
    assert res.speech["transcript"] == "How do you fine tune English models on Hindi data?"
    assert res.synthesis["confidence"] == 0.92
    assert res.stage_metrics["extraction"]["success"] is True
    assert res.stage_metrics["synthesis"]["prompt_tokens"] == 450


# 2. Extraction failure handling
def test_extraction_failure_handling():
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = ExtractionResult(
        success=False,
        source_url="https://www.instagram.com/reel/fail_test/",
        failure_category="AUTH_REQUIRED",
        failure_reason="Login required by Instagram",
    )

    res = run_pipeline(
        url="https://www.instagram.com/reel/fail_test/",
        extractor=mock_extractor,
    )

    assert res.overall_success is False
    assert res.failure_stage == "extraction"
    assert res.failure_category == "AUTH_REQUIRED"
    assert "Login required" in res.failure_message
    assert res.speech is None
    assert res.synthesis is None


# 3. ASR failure handling
def test_asr_failure_handling(
    mock_extraction_success,
    mock_vision_success,
    mock_ocr_success,
    mock_synthesis_success,
):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    asr_failed = SpeechResult(
        success=False,
        failure_category=SpeechFailureCategory.AUDIO_EXTRACTION_FAILED.value,
        failure_message="FFmpeg audio decode error",
    )

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        speech_runner=lambda p: asr_failed,
        vision_runner=lambda p: mock_vision_success,
        ocr_runner=lambda p: mock_ocr_success,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    assert res.overall_success is False
    assert res.failure_stage == "speech"
    assert res.failure_category == "AUDIO_EXTRACTION_FAILED"


# 4. Vision failure handling
def test_vision_failure_handling(
    mock_extraction_success,
    mock_speech_success,
    mock_ocr_success,
    mock_synthesis_success,
):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    vision_failed = VisionResult(
        success=False,
        failure_category=VisionFailureCategory.FRAME_EXTRACTION_FAILED.value,
        failure_message="Corrupted video frame at index 0",
    )

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        speech_runner=lambda p: mock_speech_success,
        vision_runner=lambda p: vision_failed,
        ocr_runner=lambda p: mock_ocr_success,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    assert res.overall_success is False
    assert res.failure_stage == "vision"
    assert res.failure_category == "FRAME_EXTRACTION_FAILED"


# 5. OCR failure handling
def test_ocr_failure_handling(
    mock_extraction_success,
    mock_speech_success,
    mock_vision_success,
    mock_synthesis_success,
):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    ocr_failed = OCRResult(
        success=False,
        failure_category=OCRFailureCategory.OCR_INFERENCE_FAILED.value,
        failure_message="ONNXRuntime crash",
    )

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        speech_runner=lambda p: mock_speech_success,
        vision_runner=lambda p: mock_vision_success,
        ocr_runner=lambda p: ocr_failed,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    assert res.overall_success is False
    assert res.failure_stage == "ocr"
    assert res.failure_category == "OCR_INFERENCE_FAILED"


# 6. Synthesis failure handling
def test_synthesis_failure_handling(
    mock_extraction_success,
    mock_speech_success,
    mock_vision_success,
    mock_ocr_success,
):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    synth_failed = MultimodalAnalysisResult(
        success=False,
        failure_category=SynthesisFailureCategory.JSON_PARSE_FAILED.value,
        failure_message="Malformed JSON returned",
    )

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        speech_runner=lambda p: mock_speech_success,
        vision_runner=lambda p: mock_vision_success,
        ocr_runner=lambda p: mock_ocr_success,
        synthesis_runner=lambda **kwargs: synth_failed,
    )

    assert res.overall_success is False
    assert res.failure_stage == "synthesis"
    assert res.failure_category == "JSON_PARSE_FAILED"


# 7. Partial evidence handling (image with no audio)
def test_partial_evidence_handling(
    mock_vision_success,
    mock_ocr_success,
    mock_synthesis_success,
):
    image_ext = ExtractionResult(
        success=True,
        source_url="https://www.instagram.com/p/image_post_123/",
        media_downloaded=True,
        media_path="temp/mock/img.jpg",
        caption_extracted=True,
        caption="Photo of architectural diagram",
        media_type="image",
    )
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = image_ext

    res = run_pipeline(
        url="https://www.instagram.com/p/image_post_123/",
        extractor=mock_extractor,
        vision_runner=lambda p: mock_vision_success,
        ocr_runner=lambda p: mock_ocr_success,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    # Audio should be cleanly bypassed for images
    assert res.speech is None
    assert res.overall_success is True
    assert res.content_type == "image"


# 8. Cleanup failure handling
def test_cleanup_failure_handling(mock_extraction_success, tmp_path):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    with patch("processor.run_result._execute_cleanup", return_value=(False, 2)):
        res = run_pipeline(
            url="https://www.instagram.com/reel/valid_test_123/",
            extractor=mock_extractor,
            synthesis_runner=lambda **kwargs: MultimodalAnalysisResult(success=True),
        )

        assert res.cleanup_attempted is True
        assert res.cleanup_success is False
        assert res.leftover_artifact_count == 2


# 9. Result serialization
def test_result_serialization(mock_extraction_success, mock_synthesis_success):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    json_str = res.to_json()
    parsed = json.loads(json_str)

    assert parsed["run_id"] == res.run_id
    assert parsed["source_url"] == res.source_url
    assert parsed["stages"]["synthesis"]["confidence"] == 0.92
    assert "cleanup" in parsed
    assert "model_versions" in parsed


# 10. Benchmark record serialization
def test_benchmark_record_serialization(mock_extraction_success, mock_synthesis_success):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = mock_extraction_success

    res = run_pipeline(
        url="https://www.instagram.com/reel/valid_test_123/",
        extractor=mock_extractor,
        synthesis_runner=lambda **kwargs: mock_synthesis_success,
    )

    rec = BenchmarkRecord.from_run_result(res, test_id="T01")
    rec_dict = rec.as_dict()

    required_keys = [
        "test_id",
        "source_url",
        "content_type",
        "extraction_success",
        "media_retrieved",
        "caption_retrieved",
        "speech_present",
        "transcript_quality",
        "ocr_quality",
        "visual_understanding_quality",
        "summary_quality",
        "core_takeaway_quality",
        "confidence",
        "processing_time_seconds",
        "cleanup_success",
        "failure_reason",
    ]
    for k in required_keys:
        assert k in rec_dict


# 11. Aggregate metrics calculation
def test_aggregate_metrics_calculation():
    r1 = BenchmarkRecord(
        test_id="T01",
        source_url="http://example.com/1",
        content_type="reel",
        extraction_success=True,
        media_retrieved=True,
        caption_retrieved=True,
        speech_present=True,
        processing_time_seconds=10.0,
        cleanup_success=True,
        synthesis_success=True,
        ocr_detected=True,
        vision_success=True,
    )
    r2 = BenchmarkRecord(
        test_id="T02",
        source_url="http://example.com/2",
        content_type="image",
        extraction_success=True,
        media_retrieved=True,
        caption_retrieved=False,
        speech_present=False,
        processing_time_seconds=5.0,
        cleanup_success=True,
        synthesis_success=True,
        ocr_detected=False,
        vision_success=True,
    )
    r3 = BenchmarkRecord(
        test_id="T03",
        source_url="http://example.com/3",
        content_type="reel",
        extraction_success=False,
        media_retrieved=False,
        caption_retrieved=False,
        speech_present=False,
        processing_time_seconds=2.0,
        cleanup_success=True,
        synthesis_success=False,
        ocr_detected=False,
        vision_success=False,
    )

    eval_res = evaluate_benchmark([r1, r2, r3])

    # 2 out of 3 extraction success = 66.67%
    assert round(eval_res.extraction_success_rate, 2) == 0.67
    # 1 caption retrieved out of 2 successful extractions = 50%
    assert round(eval_res.caption_retrieval_rate, 2) == 0.50
    # Cleanup: 3 out of 3 = 100%
    assert eval_res.cleanup_success_rate == 1.0
    # Average time: (10 + 5 + 2) / 3 = 5.67s
    assert round(eval_res.average_processing_time_seconds, 2) == 5.67


# 12. Human evaluation fields
def test_human_evaluation_fields():
    rec = BenchmarkRecord(
        test_id="T01",
        source_url="http://example.com/1",
        content_type="reel",
        extraction_success=True,
        media_retrieved=True,
        caption_retrieved=True,
        speech_present=True,
    )

    # Defaults must be None
    assert rec.transcript_quality is None
    assert rec.ocr_quality is None
    assert rec.visual_understanding_quality is None
    assert rec.summary_quality is None
    assert rec.core_takeaway_quality is None

    # Apply manual ground truth
    gt = GroundTruthRecord(
        test_id="T01",
        actual_has_speech=True,
        transcript_quality=0.95,
        summary_quality=0.90,
        core_takeaway_quality=0.88,
    )

    mock_run = MagicMock()
    mock_run.run_id = "T01"
    mock_run.source_url = "http://example.com/1"
    mock_run.content_type = "reel"
    mock_run.overall_success = True
    mock_run.stage_metrics = {}
    mock_run.synthesis = {"confidence": 0.9}
    mock_run.cleanup_success = True
    mock_run.total_execution_seconds = 10.0
    mock_run.model_versions = {}

    rec_with_gt = BenchmarkRecord.from_run_result(mock_run, test_id="T01", ground_truth=gt)
    assert rec_with_gt.transcript_quality == 0.95
    assert rec_with_gt.summary_quality == 0.90
    assert rec_with_gt.core_takeaway_quality == 0.88


# 13. Pass/Fail decision logic
def test_decision_logic_pass_and_conditional():
    # Scenario A: All pass
    recs_pass = [
        BenchmarkRecord(
            test_id=f"T{i}",
            source_url="http://example.com",
            content_type="reel",
            extraction_success=True,
            media_retrieved=True,
            caption_retrieved=True,
            speech_present=True,
            summary_quality=0.90,
            core_takeaway_quality=0.90,
            cleanup_success=True,
            synthesis_success=True,
        )
        for i in range(10)
    ]
    res_pass = evaluate_benchmark(recs_pass)
    assert res_pass.decision == BenchmarkDecision.PASS

    # Scenario B: Conditional Pass (high AI quality but extraction < 90%)
    recs_cond = list(recs_pass)
    # 2 extraction failures -> extraction rate = 80% (< 90%)
    recs_cond[0].extraction_success = False
    recs_cond[1].extraction_success = False
    res_cond = evaluate_benchmark(recs_cond)
    assert res_cond.decision == BenchmarkDecision.CONDITIONAL_PASS

    # Scenario C: Fail (poor quality < 85%)
    recs_fail = list(recs_pass)
    for r in recs_fail:
        r.summary_quality = 0.50
    res_fail = evaluate_benchmark(recs_fail)
    assert res_fail.decision == BenchmarkDecision.FAIL

    # Scenario D: Pending Evaluation (no human evaluation)
    recs_pending = [
        BenchmarkRecord(
            test_id=f"T{i}",
            source_url="http://example.com",
            content_type="reel",
            extraction_success=True,
            media_retrieved=True,
            caption_retrieved=True,
            speech_present=True,
            cleanup_success=True,
            summary_quality=None,
        )
        for i in range(5)
    ]
    res_pending = evaluate_benchmark(recs_pending)
    assert res_pending.decision == BenchmarkDecision.PENDING_EVALUATION


# 14. Model-version recording
def test_model_version_recording():
    run_res = RunResult(
        run_id="run_version_test",
        source_url="http://example.com",
        started_at="2026-09-05T00:00:00Z",
        completed_at="2026-09-05T00:00:10Z",
        overall_success=True,
    )

    assert "asr" in run_res.model_versions
    assert "vision" in run_res.model_versions
    assert "ocr" in run_res.model_versions
    assert "synthesis" in run_res.model_versions
    assert "Qwen/Qwen2.5-3B-Instruct" in run_res.model_versions["synthesis"]


# 15. No secrets in output
def test_no_secrets_in_output(tmp_path):
    tainted_dict = {
        "run_id": "test_leak",
        "api_key": "sk-secret-token-router-12345",
        "cookie_file": "secrets/instagram_cookies.txt",
        "auth_token": "bearer xyz987",
        "safe_field": "public_data",
        "nested": {
            "password": "super_secret_password",
        },
    }

    cleaned = sanitize_dict_for_output(tainted_dict)

    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["auth_token"] == "[REDACTED]"
    assert cleaned["nested"]["password"] == "[REDACTED]"
    assert cleaned["safe_field"] == "public_data"
    assert "sk-secret" not in json.dumps(cleaned)

    saved_path = save_run_result(cleaned, output_dir=tmp_path)
    file_content = saved_path.read_text(encoding="utf-8")
    assert "sk-secret" not in file_content
    assert "super_secret_password" not in file_content


# 16. No raw media in JSON
def test_no_raw_media_in_json():
    media_dict = {
        "run_id": "test_media",
        "image_bytes": b"\xff\xd8\xff\xe0\x00\x10JFIF" * 100,
        "video_stream": bytearray(b"\x00\x00\x00\x18ftypmp42" * 50),
        "text": "Valid text description",
    }

    sanitized = sanitize_dict_for_output(media_dict)

    assert "[RAW_BYTES" in sanitized["image_bytes"]
    assert "[RAW_BYTES" in sanitized["video_stream"]
    assert sanitized["text"] == "Valid text description"

    # Verifies standard json.dumps succeeds without binary serialization error
    json_output = json.dumps(sanitized)
    assert "Valid text description" in json_output
