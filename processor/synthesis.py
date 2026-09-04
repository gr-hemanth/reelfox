"""Multimodal synthesis abstraction & TokenRouter GLM backend (Phase 8).

Combines structured evidence from:
1. Instagram metadata/caption
2. Speech / ASR
3. Vision
4. OCR

into a structured, useful interpretation with strict anti-hallucination rules.
The synthesis model receives ONLY structured textual evidence, never raw media.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import httpx

from processor.synthesis_models import (
    MultimodalAnalysisResult,
    MultimodalEvidence,
    SynthesisFailureCategory,
)

logger = logging.getLogger("analyzer.processor.synthesis")

DEFAULT_LOCAL_SYNTHESIS_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SYNTHESIS_MODEL = "z-ai/glm-5.3-free"
DEFAULT_SYNTHESIS_ENDPOINT = "https://api.tokenrouter.com/v1/chat/completions"
DEFAULT_SYNTHESIS_TIMEOUT = 60.0

SYNTHESIS_SYSTEM_PROMPT = """You are a multimodal content interpretation system analyzing an Instagram post or Reel.
You are given structured textual evidence extracted by specialized upstream layers:
- Metadata & Caption
- Speech Transcript (ASR)
- Visual Observations (Vision VLM)
- On-Screen Text (OCR)

STRICT ANTI-HALLUCINATION POLICY:
1. Use ONLY the supplied evidence. Do NOT invent facts, statistics, names, or events.
2. Do NOT assume or extrapolate information that is not directly supported by the evidence.
3. Do NOT use external pre-training knowledge to fill in missing details about specific people or companies.
4. If evidence is ambiguous, incomplete, or conflicts, explicitly state the limitation or conflict.
5. Do NOT infer private personal details about people.
6. EVIDENCE HIERARCHY when determining core topic and facts:
   a. Spoken audio transcript (primary truth of spoken message)
   b. On-screen OCR text and slide content (primary truth of visual text/slides)
   c. Post caption
   d. Visual observations (visual context, subjects, and actions)

OUTPUT REQUIREMENTS:
You MUST output valid JSON ONLY matching the following schema:
{
  "summary": "<concise, factual summary strictly grounded in evidence>",
  "key_points": [
    "<factual point 1>",
    "<factual point 2>",
    "<factual point 3>"
  ],
  "core_takeaway": "<single primary conclusion or takeaway directly supported by evidence>",
  "relevant_context": "<relevant setting, context, technical topic, or format grounded in evidence>",
  "confidence": <float between 0.0 and 1.0 reflecting overall evidence quality and certainty>
}

CONFIDENCE GUIDELINES:
- Assign high confidence (0.80 - 1.0) only when multiple evidence sources are clear and mutually supportive.
- Lower confidence (0.40 - 0.79) if evidence is noisy, partial, or missing key modalities.
- Assign very low confidence (< 0.40) if evidence is heavily conflicting or insufficient.
Do NOT output markdown code fences or any text outside the JSON object.
"""


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and extraneous whitespace from model response."""
    cleaned = text.strip()
    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _compute_evidence_used(evidence_dict: Dict[str, Any]) -> Dict[str, bool]:
    """Deterministically determine which evidence modalities were available and usable."""
    metadata = evidence_dict.get("metadata", {})
    speech = evidence_dict.get("speech", {})
    vision = evidence_dict.get("vision", {})
    ocr = evidence_dict.get("ocr", {})

    caption_present = bool(metadata.get("caption_present") or metadata.get("caption"))
    speech_used = bool(speech.get("available") and speech.get("speech_present") and speech.get("transcript"))
    vision_used = bool(vision.get("available") and vision.get("observations"))
    ocr_used = bool(ocr.get("available") and ocr.get("text_detected") and (ocr.get("combined_text") or ocr.get("text_blocks")))

    return {
        "caption": caption_present,
        "speech": speech_used,
        "vision": vision_used,
        "ocr": ocr_used,
    }


class BaseSynthesizer(abc.ABC):
    """Abstract interface for multimodal evidence synthesizers."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the synthesis model."""
        pass

    @abc.abstractmethod
    def synthesize(
        self,
        evidence: MultimodalEvidence | Dict[str, Any],
    ) -> MultimodalAnalysisResult:
        """Synthesize multimodal evidence into a structured MultimodalAnalysisResult."""
        pass


class TokenRouterGLMSynthesizer(BaseSynthesizer):
    """Multimodal synthesizer using TokenRouter (z-ai/glm-5.3-free).

    Operates at ₹0 cost via TokenRouter.
    Receives only structured evidence; never raw video or images.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_SYNTHESIS_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ):
        self._api_key = (api_key or os.environ.get("TOKENROUTER_API_KEY", "")).strip()
        self._model = (model or os.environ.get("SYNTHESIS_MODEL", DEFAULT_SYNTHESIS_MODEL)).strip()
        self._endpoint = (endpoint or os.environ.get("SYNTHESIS_ENDPOINT", DEFAULT_SYNTHESIS_ENDPOINT)).strip()
        self._timeout = timeout
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    def synthesize(
        self,
        evidence: MultimodalEvidence | Dict[str, Any],
    ) -> MultimodalAnalysisResult:
        """Synthesize structured evidence into MultimodalAnalysisResult."""
        t0 = time.perf_counter()

        # Step 1: Validate API key existence without logging it
        if not self._api_key:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.API_KEY_MISSING.value,
                failure_message="TOKENROUTER_API_KEY is not configured or missing in environment.",
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 2: Prepare evidence dict
        if isinstance(evidence, MultimodalEvidence):
            evidence_dict = evidence.as_dict()
        elif isinstance(evidence, dict):
            evidence_dict = evidence
        else:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message=f"Unsupported evidence type: {type(evidence)}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        evidence_used = _compute_evidence_used(evidence_dict)
        user_content = json.dumps(evidence_dict, indent=2, ensure_ascii=False)

        # Step 3: Execute API request
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"EVIDENCE FROM INSTAGRAM POST/REEL:\n{user_content}\n\nProduce the structured JSON analysis.",
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 2
        response = None
        for attempt in range(max_retries + 1):
            t_req_start = time.perf_counter()
            try:
                if self._client is not None:
                    response = self._client.post(
                        self._endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self._timeout,
                    )
                else:
                    with httpx.Client(timeout=self._timeout) as client:
                        response = client.post(
                            self._endpoint,
                            json=payload,
                            headers=headers,
                        )
                request_latency = time.perf_counter() - t_req_start

                if (
                    response.status_code in (429, 502, 503, 504)
                    and attempt < max_retries
                    and self._client is None
                ):
                    logger.warning(
                        "TokenRouter transient HTTP %d (attempt %d/%d). Retrying...",
                        response.status_code,
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break

            except httpx.TimeoutException as exc:
                if attempt < max_retries and self._client is None:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return MultimodalAnalysisResult(
                    success=False,
                    model_name=self._model,
                    failure_category=SynthesisFailureCategory.API_TIMEOUT.value,
                    failure_message=f"TokenRouter API request timed out after {self._timeout}s: {exc}",
                    processing_time_seconds=time.perf_counter() - t0,
                )
            except httpx.RequestError as exc:
                if attempt < max_retries and self._client is None:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return MultimodalAnalysisResult(
                    success=False,
                    model_name=self._model,
                    failure_category=SynthesisFailureCategory.API_REQUEST_FAILED.value,
                    failure_message=f"TokenRouter HTTP connection error: {exc}",
                    processing_time_seconds=time.perf_counter() - t0,
                )
            except Exception as exc:
                return MultimodalAnalysisResult(
                    success=False,
                    model_name=self._model,
                    failure_category=SynthesisFailureCategory.SYNTHESIS_FAILED.value,
                    failure_message=f"Unexpected request error: {exc}",
                    processing_time_seconds=time.perf_counter() - t0,
                )


        # Step 4: Handle HTTP status codes
        status_code = response.status_code
        if status_code in (401, 403):
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.API_AUTHENTICATION_FAILED.value,
                failure_message=f"Authentication failed with status {status_code}. Verify TOKENROUTER_API_KEY.",
                processing_time_seconds=time.perf_counter() - t0,
            )
        if status_code == 429:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.API_RATE_LIMITED.value,
                failure_message="TokenRouter API rate limit exceeded (HTTP 429).",
                processing_time_seconds=time.perf_counter() - t0,
            )
        if status_code >= 400:
            error_body = response.text[:300] if response.text else "No response body"
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.API_REQUEST_FAILED.value,
                failure_message=f"TokenRouter API error (HTTP {status_code}): {error_body}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 5: Parse response body & token usage
        try:
            resp_json = response.json()
        except Exception as exc:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message=f"Non-JSON response from API: {exc}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        choices = resp_json.get("choices", [])
        if not choices:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message="No choices returned in model response.",
                processing_time_seconds=time.perf_counter() - t0,
            )

        msg = choices[0].get("message", {})
        raw_content = msg.get("content")
        if not raw_content:
            reasoning_content = msg.get("reasoning_content", "")
            if reasoning_content:
                json_match = re.search(r"(\{[\s\S]*\"summary\"[\s\S]*\"core_takeaway\"[\s\S]*\})", reasoning_content)
                if json_match:
                    raw_content = json_match.group(1)

        if not raw_content:
            finish_reason = choices[0].get("finish_reason")
            msg_text = (
                "Model exhausted max_tokens during reasoning before completing response."
                if finish_reason == "length"
                else "Model returned empty message content."
            )
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message=msg_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        usage = resp_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # Step 6: Parse model JSON content
        cleaned_content = _clean_json_text(raw_content)
        try:
            parsed_data = json.loads(cleaned_content)
        except Exception as exc:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.JSON_PARSE_FAILED.value,
                failure_message=f"Model output is not valid JSON: {exc}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 7: Strict schema validation
        if not isinstance(parsed_data, dict):
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message=f"Expected JSON object, got {type(parsed_data).__name__}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        required_keys = ("summary", "key_points", "core_takeaway", "relevant_context", "confidence")
        missing_keys = [k for k in required_keys if k not in parsed_data]
        if missing_keys:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message=f"Missing required keys in synthesis response: {missing_keys}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Validate types
        summary = str(parsed_data.get("summary") or "").strip()
        raw_points = parsed_data.get("key_points")
        if not isinstance(raw_points, list):
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model,
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message="key_points must be a list of strings.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )
        key_points = [str(pt).strip() for pt in raw_points if str(pt).strip()]

        core_takeaway = str(parsed_data.get("core_takeaway") or "").strip()
        relevant_context = str(parsed_data.get("relevant_context") or "").strip()

        try:
            confidence = float(parsed_data.get("confidence", 0.0))
            # Clamp bounds
            confidence = max(0.0, min(1.0, round(confidence, 2)))
        except (ValueError, TypeError):
            confidence = 0.5

        total_time = time.perf_counter() - t0

        return MultimodalAnalysisResult(
            success=True,
            summary=summary,
            key_points=key_points,
            core_takeaway=core_takeaway,
            relevant_context=relevant_context,
            confidence=confidence,
            evidence_used=evidence_used,
            model_name=self._model,
            processing_time_seconds=total_time,
            request_latency_seconds=request_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class LocalQwenSynthesizer(BaseSynthesizer):
    """Permanent local multimodal synthesizer using Qwen2.5-3B-Instruct.

    Runs 100% locally with Hugging Face Transformers at ₹0 cost.
    Requires no cloud APIs or external internet connectivity once weights are cached.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cpu",
        torch_dtype: Any = None,
        tokenizer: Any = None,
        model: Any = None,
    ):
        if model_id:
            self._model_id = model_id.strip()
        else:
            cand = (os.environ.get("LOCAL_SYNTHESIS_MODEL") or os.environ.get("SYNTHESIS_MODEL") or "").strip()
            if cand and not cand.startswith("z-ai/"):
                self._model_id = cand
            else:
                self._model_id = DEFAULT_LOCAL_SYNTHESIS_MODEL
        self._device = device
        self._torch_dtype = torch_dtype
        self._tokenizer = tokenizer
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_id

    def _ensure_loaded(self) -> None:
        """Lazy-load the model and tokenizer upon first synthesis request."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        logger.info("Loading local synthesis model: %s (device=%s)", self._model_id, self._device)

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

        if self._model is None:
            dtype = self._torch_dtype or (torch.bfloat16 if self._device == "cpu" else torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self._model.eval()

        logger.info("Local synthesis model loaded in %.2fs", time.perf_counter() - t0)

    def synthesize(
        self,
        evidence: MultimodalEvidence | Dict[str, Any],
    ) -> MultimodalAnalysisResult:
        """Synthesize multimodal evidence using local Qwen model into MultimodalAnalysisResult."""
        t0 = time.perf_counter()

        # Step 1: Validate evidence input
        if isinstance(evidence, MultimodalEvidence):
            evidence_dict = evidence.as_dict()
        elif isinstance(evidence, dict):
            evidence_dict = evidence
        else:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message=f"Unsupported evidence type: {type(evidence)}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        evidence_used = _compute_evidence_used(evidence_dict)
        user_content = json.dumps(evidence_dict, indent=2, ensure_ascii=False)

        # Step 2: Ensure model is loaded
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.error("Failed to load local synthesis model %s: %s", self._model_id, exc)
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                failure_category=SynthesisFailureCategory.SYNTHESIS_FAILED.value,
                failure_message=f"Failed to load local model {self._model_id}: {exc}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 3: Run inference
        import torch

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"EVIDENCE FROM INSTAGRAM POST/REEL:\n{user_content}\n\nProduce the structured JSON analysis.",
            },
        ]

        t_gen_0 = time.perf_counter()
        try:
            prompt_formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt_formatted, return_tensors="pt")
            prompt_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            gen_latency = time.perf_counter() - t_gen_0
            new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
            completion_tokens = len(new_tokens)
            total_tokens = prompt_tokens + completion_tokens
            raw_content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        except Exception as exc:
            logger.error("Local inference error: %s", exc)
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                failure_category=SynthesisFailureCategory.SYNTHESIS_FAILED.value,
                failure_message=f"Local generation failed: {exc}",
                processing_time_seconds=time.perf_counter() - t0,
            )

        if not raw_content:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                failure_category=SynthesisFailureCategory.INVALID_MODEL_RESPONSE.value,
                failure_message="Model returned empty output.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 4: Parse model JSON content
        cleaned_content = _clean_json_text(raw_content)
        try:
            parsed_data = json.loads(cleaned_content)
        except Exception as exc:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.JSON_PARSE_FAILED.value,
                failure_message=f"Model output is not valid JSON: {exc}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        # Step 5: Strict schema validation
        if not isinstance(parsed_data, dict):
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message=f"Expected JSON object, got {type(parsed_data).__name__}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        required_keys = ("summary", "key_points", "core_takeaway", "relevant_context", "confidence")
        missing_keys = [k for k in required_keys if k not in parsed_data]
        if missing_keys:
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                raw_response=cleaned_content[:500],
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message=f"Missing required keys in synthesis response: {missing_keys}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )

        summary = str(parsed_data.get("summary") or "").strip()
        raw_points = parsed_data.get("key_points")
        if not isinstance(raw_points, list):
            return MultimodalAnalysisResult(
                success=False,
                model_name=self._model_id,
                failure_category=SynthesisFailureCategory.SCHEMA_VALIDATION_FAILED.value,
                failure_message="key_points must be a list of strings.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=time.perf_counter() - t0,
            )
        key_points = [str(pt).strip() for pt in raw_points if str(pt).strip()]
        core_takeaway = str(parsed_data.get("core_takeaway") or "").strip()
        relevant_context = str(parsed_data.get("relevant_context") or "").strip()

        try:
            confidence = float(parsed_data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, round(confidence, 2)))
        except (ValueError, TypeError):
            confidence = 0.5

        total_time = time.perf_counter() - t0

        return MultimodalAnalysisResult(
            success=True,
            summary=summary,
            key_points=key_points,
            core_takeaway=core_takeaway,
            relevant_context=relevant_context,
            confidence=confidence,
            evidence_used=evidence_used,
            model_name=self._model_id,
            processing_time_seconds=total_time,
            request_latency_seconds=gen_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
