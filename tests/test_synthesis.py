"""Offline unit tests for Phase 8 Multimodal Synthesis.

All tests run completely offline without external network or TokenRouter calls.
Verifies all 14 required cases:
1. successful synthesis
2. malformed JSON
3. schema failure
4. API authentication failure
5. timeout
6. rate limit
7. API request failure
8. missing API key
9. missing evidence sources
10. long transcript/OCR input truncation
11. confidence bounds
12. secret leakage prevention
13. token/latency metadata
14. deterministic evidence_used flags
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

import config as app_config
from processor.synthesis import (
    LocalQwenSynthesizer,
    TokenRouterGLMSynthesizer,
    _compute_evidence_used,
)
from processor.synthesis_models import (
    MultimodalAnalysisResult,
    MultimodalEvidence,
    SynthesisFailureCategory,
    _truncate_text,
)


def _make_mock_client(status_code: int = 200, json_body: dict | None = None, text_body: str | None = None, exc: Exception | None = None) -> httpx.Client:
    """Create an httpx.Client with a MockTransport for offline testing."""
    def handler(request: httpx.Request) -> httpx.Response:
        if exc is not None:
            raise exc
        if json_body is not None:
            return httpx.Response(status_code=status_code, json=json_body)
        return httpx.Response(status_code=status_code, text=text_body or "")

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def _valid_model_response_body(summary="A factual summary", points=None, takeaway="Core conclusion", conf=0.92):
    if points is None:
        points = ["Key finding 1", "Key finding 2"]
    content_obj = {
        "summary": summary,
        "key_points": points,
        "core_takeaway": takeaway,
        "relevant_context": "Instagram Reel demonstration",
        "confidence": conf,
    }
    return {
        "id": "chatcmpl-test-123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content_obj),
                }
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 85,
            "total_tokens": 235,
        },
    }


@pytest.fixture
def sample_evidence():
    return MultimodalEvidence(
        source_url="https://www.instagram.com/p/test/",
        metadata={"caption": "Check out this workflow tips!", "caption_present": True, "hashtags": ["#tech"]},
        speech={"available": True, "speech_present": True, "transcript": "Here is the best method to automate tasks."},
        vision={"available": True, "observations": ["A computer screen showing code", "A person gesturing"]},
        ocr={"available": True, "text_detected": True, "combined_text": "AUTOMATION GUIDE 101", "text_blocks": [{"text": "AUTOMATION GUIDE 101"}]},
    )


# 1. Successful synthesis
def test_successful_synthesis(sample_evidence):
    body = _valid_model_response_body()
    client = _make_mock_client(200, json_body=body)
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is True
    assert result.summary == "A factual summary"
    assert len(result.key_points) == 2
    assert result.core_takeaway == "Core conclusion"
    assert result.relevant_context == "Instagram Reel demonstration"
    assert result.confidence == 0.92
    assert result.evidence_used["caption"] is True
    assert result.evidence_used["speech"] is True
    assert result.evidence_used["vision"] is True
    assert result.evidence_used["ocr"] is True
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 85
    assert result.total_tokens == 235
    assert result.processing_time_seconds is not None and result.processing_time_seconds >= 0


# 2. Malformed JSON
def test_malformed_json(sample_evidence):
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here is your analysis: {summary: broken json, not valid}",
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }
    client = _make_mock_client(200, json_body=body)
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.JSON_PARSE_FAILED.value
    assert "not valid JSON" in result.failure_message


# 3. Schema failure
def test_schema_failure_missing_keys(sample_evidence):
    # Missing core_takeaway and key_points
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"summary": "Incomplete", "confidence": 0.5}),
                }
            }
        ]
    }
    client = _make_mock_client(200, json_body=body)
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value
    assert "Missing required keys" in result.failure_message


def test_schema_failure_invalid_types(sample_evidence):
    # key_points is not a list
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "summary": "Valid summary",
                        "key_points": "not a list",
                        "core_takeaway": "Takeaway",
                        "relevant_context": "Context",
                        "confidence": 0.8,
                    }),
                }
            }
        ]
    }
    client = _make_mock_client(200, json_body=body)
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value
    assert "key_points must be a list" in result.failure_message


# 4. API authentication failure
def test_api_authentication_failure(sample_evidence):
    client = _make_mock_client(401, text_body="Unauthorized: Invalid API key")
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-bad-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.API_AUTHENTICATION_FAILED.value
    assert "401" in result.failure_message


# 5. Timeout
def test_api_timeout(sample_evidence):
    client = _make_mock_client(exc=httpx.TimeoutException("Read timed out"))
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.API_TIMEOUT.value
    assert "timed out" in result.failure_message


# 6. Rate limit (429)
def test_api_rate_limited(sample_evidence):
    client = _make_mock_client(429, text_body="Rate limit exceeded")
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.API_RATE_LIMITED.value
    assert "rate limit" in result.failure_message.lower()


# 7. API request failure (500)
def test_api_request_failure(sample_evidence):
    client = _make_mock_client(500, text_body="Internal Server Error")
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.API_REQUEST_FAILED.value
    assert "500" in result.failure_message


# 8. Missing API key
def test_missing_api_key(monkeypatch, sample_evidence):
    monkeypatch.delenv("TOKENROUTER_API_KEY", raising=False)
    synthesizer = TokenRouterGLMSynthesizer(api_key="")

    result = synthesizer.synthesize(sample_evidence)

    assert result.success is False
    assert result.failure_category == SynthesisFailureCategory.API_KEY_MISSING.value
    assert "missing in environment" in result.failure_message


# 9. Missing evidence sources
def test_missing_evidence_sources():
    fake_extraction = MagicMock()
    fake_extraction.source_url = "https://instagram.com/p/123"
    fake_extraction.caption = "Just a caption"
    fake_extraction.media_type = "image"
    fake_extraction.hashtags = []

    # Speech, Vision, OCR all None or unavailable
    evidence = MultimodalEvidence.from_results(
        extraction_result=fake_extraction,
        speech_result=None,
        vision_result=None,
        ocr_result=None,
    )

    assert evidence.speech["available"] is False
    assert evidence.vision["available"] is False
    assert evidence.ocr["available"] is False

    used = _compute_evidence_used(evidence.as_dict())
    assert used["caption"] is True
    assert used["speech"] is False
    assert used["vision"] is False
    assert used["ocr"] is False


# 10. Long transcript / OCR input truncation
def test_long_input_truncation():
    long_caption = "A" * 5000
    long_speech = "B" * 10000
    long_ocr = "C" * 10000

    truncated_cap = _truncate_text(long_caption, max_chars=1000)
    truncated_speech = _truncate_text(long_speech, max_chars=4000)
    truncated_ocr = _truncate_text(long_ocr, max_chars=4000)

    assert len(truncated_cap) < 1100
    assert "[... truncated for input limit ...]" in truncated_cap
    assert len(truncated_speech) < 4100
    assert "[... truncated for input limit ...]" in truncated_speech
    assert len(truncated_ocr) < 4100
    assert "[... truncated for input limit ...]" in truncated_ocr


# 11. Confidence bounds
def test_confidence_bounds(sample_evidence):
    # Test confidence > 1.0 clamped to 1.0
    body_high = _valid_model_response_body(conf=1.75)
    client_high = _make_mock_client(200, json_body=body_high)
    res_high = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client_high).synthesize(sample_evidence)
    assert res_high.confidence == 1.0

    # Test confidence < 0.0 clamped to 0.0
    body_low = _valid_model_response_body(conf=-0.8)
    client_low = _make_mock_client(200, json_body=body_low)
    res_low = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client_low).synthesize(sample_evidence)
    assert res_low.confidence == 0.0


# 12. Secret leakage prevention
def test_secret_leakage_prevention():
    secret_key = "sk-super-secret-key-that-must-not-leak"
    synthesizer = TokenRouterGLMSynthesizer(api_key=secret_key)
    
    # Check repr / str
    assert secret_key not in repr(synthesizer)
    assert secret_key not in str(synthesizer)

    # Check config repr
    cfg = app_config.Config(tokenrouter_api_key=secret_key)
    assert secret_key not in repr(cfg)

    # Check failure message does not contain key
    client = _make_mock_client(401, text_body="Unauthorized key")
    res = TokenRouterGLMSynthesizer(api_key=secret_key, client=client).synthesize({})
    assert secret_key not in (res.failure_message or "")


# 13. Token / latency metadata
def test_token_latency_metadata(sample_evidence):
    body = _valid_model_response_body()
    client = _make_mock_client(200, json_body=body)
    synthesizer = TokenRouterGLMSynthesizer(api_key="sk-test-key", client=client)

    result = synthesizer.synthesize(sample_evidence)

    assert result.prompt_tokens == 150
    assert result.completion_tokens == 85
    assert result.total_tokens == 235
    assert result.processing_time_seconds is not None
    assert result.processing_time_seconds >= 0.0


# 14. Deterministic evidence_used flags
def test_deterministic_evidence_used_flags():
    # Only caption
    dict1 = {
        "metadata": {"caption": "Hello world", "caption_present": True},
        "speech": {"available": False},
        "vision": {"available": False},
        "ocr": {"available": False},
    }
    flags1 = _compute_evidence_used(dict1)
    assert flags1 == {"caption": True, "speech": False, "vision": False, "ocr": False}

    # Caption + Speech + OCR (no vision)
    dict2 = {
        "metadata": {"caption": "Hello", "caption_present": True},
        "speech": {"available": True, "speech_present": True, "transcript": "Spoken words"},
        "vision": {"available": False},
        "ocr": {"available": True, "text_detected": True, "combined_text": "Text on screen"},
    }
    flags2 = _compute_evidence_used(dict2)
    assert flags2 == {"caption": True, "speech": True, "vision": False, "ocr": True}


# 15. Local Qwen Synthesizer (offline tests with mock tokenizer/model)
def _create_mock_qwen(generated_text: str):
    """Create mock tokenizer and model for offline LocalQwenSynthesizer testing."""
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.apply_chat_template.return_value = "<mock_prompt>"
    # Return input_ids of shape (1, 5)
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[10, 20, 30, 40, 50]])}
    mock_tokenizer.decode.return_value = generated_text

    mock_model = MagicMock()
    # Output tensor with 5 prompt tokens + 3 new tokens
    mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 100, 101, 102]])

    return mock_tokenizer, mock_model


def test_local_qwen_synthesizer_success(sample_evidence):
    content_obj = {
        "summary": "Local summary from Qwen",
        "key_points": ["Point A", "Point B"],
        "core_takeaway": "Local core takeaway",
        "relevant_context": "Local context",
        "confidence": 0.95,
    }
    tok, mod = _create_mock_qwen(json.dumps(content_obj))
    synthesizer = LocalQwenSynthesizer(tokenizer=tok, model=mod)

    res = synthesizer.synthesize(sample_evidence)

    assert res.success is True
    assert res.summary == "Local summary from Qwen"
    assert res.key_points == ["Point A", "Point B"]
    assert res.core_takeaway == "Local core takeaway"
    assert res.confidence == 0.95
    assert res.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert res.prompt_tokens == 5
    assert res.completion_tokens == 3
    assert res.total_tokens == 8


def test_local_qwen_synthesizer_malformed_json(sample_evidence):
    tok, mod = _create_mock_qwen("This is not JSON at all!")
    synthesizer = LocalQwenSynthesizer(tokenizer=tok, model=mod)

    res = synthesizer.synthesize(sample_evidence)

    assert res.success is False
    assert res.failure_category == SynthesisFailureCategory.JSON_PARSE_FAILED.value
    assert "not valid JSON" in (res.failure_message or "")


def test_local_qwen_synthesizer_schema_missing_keys(sample_evidence):
    incomplete_obj = {"summary": "Only summary"}
    tok, mod = _create_mock_qwen(json.dumps(incomplete_obj))
    synthesizer = LocalQwenSynthesizer(tokenizer=tok, model=mod)

    res = synthesizer.synthesize(sample_evidence)

    assert res.success is False
    assert res.failure_category == SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value
    assert "Missing required keys" in (res.failure_message or "")


def test_local_qwen_synthesizer_empty_output(sample_evidence):
    tok, mod = _create_mock_qwen("")
    synthesizer = LocalQwenSynthesizer(tokenizer=tok, model=mod)

    res = synthesizer.synthesize(sample_evidence)

    assert res.success is False
    assert res.failure_category == SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value
    assert "empty output" in (res.failure_message or "")

