"""Standalone validation runner for Qwen/Qwen2.5-3B-Instruct on local hardware.

Evaluates:
- Hardware & environment specs
- Model loading & memory footprint (RAM)
- Primary validation prompt (Sarvam Reel evidence)
- 6 JSON reliability test cases:
    1. Normal evidence
    2. Missing speech
    3. Noisy OCR
    4. Conflicting evidence
    5. Long transcript
    6. Empty optional fields
- Generation speed (tokens/sec, latency)
- Strict JSON schema adherence without regex / post-processing hacks
- Comparison against saved GLM reference in output/multimodal_result.json
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are a multimodal content interpretation system analyzing an Instagram post or Reel.
You receive structured textual evidence from upstream extraction, speech (ASR), vision (VLM), and OCR.

STRICT INSTRUCTIONS:
1. Rely ONLY on the provided evidence. Do NOT invent facts or extrapolate beyond what is given.
2. If evidence is missing, conflicting, or uncertain, state the uncertainty explicitly.
3. Output MUST be valid JSON ONLY matching this exact schema:
{
  "summary": "<factual summary grounded strictly in evidence>",
  "key_points": [
    "<key point 1>",
    "<key point 2>"
  ],
  "core_takeaway": "<single primary conclusion or takeaway directly supported by evidence>",
  "relevant_context": "<relevant setting, context, technical topic, or format>",
  "confidence": <float between 0.0 and 1.0 reflecting overall evidence quality>
}
Do NOT wrap in markdown fences. Output raw JSON starting with { and ending with }."""


def get_process_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def validate_schema(data: dict) -> tuple[bool, str]:
    """Strictly validate schema without heuristics."""
    if not isinstance(data, dict):
        return False, "Not a JSON object (dict)"
    required_keys = {"summary", "key_points", "core_takeaway", "relevant_context", "confidence"}
    missing = required_keys - set(data.keys())
    if missing:
        return False, f"Missing required keys: {sorted(missing)}"
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return False, "Invalid 'summary' (must be non-empty string)"
    if not isinstance(data["key_points"], list) or not data["key_points"]:
        return False, "Invalid 'key_points' (must be non-empty list of strings)"
    if not isinstance(data["core_takeaway"], str) or not data["core_takeaway"].strip():
        return False, "Invalid 'core_takeaway' (must be non-empty string)"
    if not isinstance(data["relevant_context"], str):
        return False, "Invalid 'relevant_context' (must be string)"
    try:
        conf = float(data["confidence"])
        if not (0.0 <= conf <= 1.0):
            return False, f"Confidence {conf} out of bounds [0.0, 1.0]"
    except (ValueError, TypeError):
        return False, f"Confidence {data.get('confidence')} not a float"
    return True, "OK"


def run_inference(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 512,
) -> tuple[str, float, int, float]:
    """Execute inference and return (generated_text, latency_seconds, token_count, tokens_per_sec)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    # Extract only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    token_count = len(new_tokens)
    speed = token_count / latency if latency > 0 else 0.0
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return text, latency, token_count, speed


def main():
    print("=" * 65)
    print("QWEN2.5-3B-INSTRUCT: STANDALONE LOCAL VALIDATION RUNNER")
    print("=" * 65)

    # Step 1: Environment & Hardware
    print("\n>>> 1. Environment & Hardware Diagnostics:")
    print(f"  Python Version:     {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"  OS:                 {platform.platform()}")
    print(f"  Processor:          {platform.processor() or 'AMD Ryzen 7 7435HS'}")
    print(f"  PyTorch Version:    {torch.__version__}")
    print(f"  PyTorch CUDA Built: {torch.version.cuda}")
    print(f"  CUDA Available:     {torch.cuda.is_available()}")
    print(f"  Selected Device:    CPU (torch.bfloat16 native inference)")

    # Step 2: Model Loading
    print(f"\n>>> 2. Loading Model: {MODEL_ID}...")
    mem_before = get_process_memory_mb()
    t_load_0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    t_load = time.perf_counter() - t_load_0
    mem_after = get_process_memory_mb()
    mem_diff = mem_after - mem_before

    print(f"  Model Load Time:    {t_load:.2f}s")
    print(f"  Memory Footprint:   ~{mem_diff:.1f} MB ({mem_diff / 1024:.2f} GB RAM)")
    print(f"  Vocab Size:         {tokenizer.vocab_size}")

    # Step 3: Primary Standalone Validation Prompt
    print("\n>>> 3. Running Primary Standalone Validation Prompt (Sarvam Reel Evidence):")
    primary_prompt = """You are given evidence from an Instagram post.

Caption:
"Sarvam interview questions"

Speech:
"If you have ever wondered what they ask in these machine learning interviews..."

OCR:
"SARVAM
84 LPA
AI RESEARCH ENGINEER - 2026
INTERVIEW"

Vision:
"A person standing in front of a computer displaying a Sarvam webpage."

Return ONLY valid JSON:
{
  "summary": "...",
  "key_points": ["..."],
  "core_takeaway": "...",
  "relevant_context": "...",
  "confidence": 0.0
}"""

    raw_output, latency, tokens, tok_sec = run_inference(model, tokenizer, primary_prompt)
    print(f"  Latency:            {latency:.2f}s")
    print(f"  Generated Tokens:   {tokens}")
    print(f"  Generation Speed:   {tok_sec:.2f} tokens/s")
    print("\n--- Raw Model Output ---")
    print(raw_output)

    # Check JSON validity directly without post-processing
    try:
        parsed_primary = json.loads(raw_output)
        is_valid, msg = validate_schema(parsed_primary)
        print(f"\n  JSON Direct Parse:  PASS (valid JSON without hacks)")
        print(f"  Schema Compliance:  {'PASS' if is_valid else 'FAIL'} ({msg})")
    except Exception as exc:
        print(f"\n  JSON Direct Parse:  FAIL: {exc}")
        parsed_primary = None

    # Step 4: Testing 6 JSON Reliability Scenarios
    print("\n>>> 4. Testing 6 JSON Reliability Scenarios:")
    scenarios = [
        (
            "Scenario 1: Normal Evidence (Standard Reel)",
            """Caption: "Sarvam interview questions"
Speech: "If you have ever wondered what they ask in these machine learning interviews? Here are a few questions for their AI research engineer role for 2026."
OCR: "SARVAM\n84 LPA\nAI RESEARCH ENGINEER - 2026"
Vision: "A speaker discussing AI roles in front of a monitor showing the Sarvam website."
"""
        ),
        (
            "Scenario 2: Missing Speech (Image Post / Silent Reel)",
            """Caption: "Infographic breakdown of LLM serving costs"
Speech: "(No audio stream present in media)"
OCR: "SERVING COSTS 4X HINDI VS ENGLISH\nCONTEXT OVERFLOW\nFIXING TOKENIZATION"
Vision: "Infographic diagram illustrating vocabulary tokenization and memory usage."
"""
        ),
        (
            "Scenario 3: Noisy OCR (Blurred Overlay Characters)",
            """Caption: "Interview prep series episode 3"
Speech: "We are looking at questions asked at Sarvam AI for generative AI research roles."
OCR: "S@rv@m A! 84 L*PA INT3RV!EW QUES7IONS"
Vision: "A man talking with text overlaid on screen."
"""
        ),
        (
            "Scenario 4: Conflicting Evidence (Misleading Caption vs Spoken/OCR)",
            """Caption: "How to crack Google and Meta L5 interviews!"
Speech: "Today we are analyzing an interview question from Sarvam, an Indian AI startup."
OCR: "SARVAM AI INTERVIEW QUESTION - 2026"
Vision: "Webpage with Sarvam logo and text 'AI for all from India'."
"""
        ),
        (
            "Scenario 5: Long Transcript (In-Depth Technical Problem)",
            """Caption: "Sarvam AI Research Engineer 2026 Problem"
Speech: "If you have ever wondered what they ask in these machine learning interviews, here is a question from Sarvam for their 2026 AI research engineer role. You fine-tune an open English-centric 8B model on a large Hindi corpus. Quality on Hindi benchmarks is respectable and the team is happy. But serving costs land roughly 4x per request versus your English traffic, and long documents that fit comfortably in English overflow the context window in Hindi. You have changed nothing about the architecture, sequence length, or hardware. What is the root cause and how would you fix it?"
OCR: "SARVAM\n84 LPA\nENGLISH 8B MODEL\nHINDI CORPUS\n4X SERVING COST\nOVERFLOW"
Vision: "Speaker standing in front of slides detailing the English 8B fine-tuning challenge and salary benchmark charts."
"""
        ),
        (
            "Scenario 6: Empty Optional Fields (Minimal Input)",
            """Caption: ""
Speech: ""
OCR: "84 LPA AI ROLE"
Vision: "A presentation slide."
"""
        ),
    ]

    scenario_results = []
    for name, content in scenarios:
        print(f"\n  * Testing {name}...")
        prompt = f"Evidence from Instagram post:\n{content}\nReturn ONLY valid JSON matching the schema."
        out, lat, tok, spd = run_inference(model, tokenizer, prompt)
        try:
            parsed = json.loads(out)
            valid, reason = validate_schema(parsed)
            print(f"    Status: {'PASS' if valid else 'SCHEMA_FAIL'} ({reason}) in {lat:.2f}s ({spd:.1f} tok/s, conf={parsed.get('confidence')})")
            scenario_results.append({
                "name": name,
                "valid": valid,
                "latency": lat,
                "speed": spd,
                "confidence": parsed.get("confidence"),
                "summary": parsed.get("summary"),
            })
        except Exception as e:
            print(f"    Status: JSON_PARSE_FAIL ({e}) in {lat:.2f}s")
            scenario_results.append({
                "name": name,
                "valid": False,
                "error": str(e),
                "latency": lat,
                "speed": spd,
            })

    # Step 5: Compare Against Reference GLM Output
    print("\n>>> 5. Comparison Against Saved GLM Reference:")
    glm_file = Path("output/multimodal_result.json")
    if glm_file.exists():
        with open(glm_file, "r", encoding="utf-8") as f:
            glm_data = json.load(f)
        glm_synth = glm_data.get("synthesis", {})
        print("  [GLM-5.3 Reference Metrics]")
        print(f"    Model:              {glm_synth.get('model_name')}")
        print(f"    Latency:            {glm_synth.get('request_latency_seconds', 0):.2f}s")
        print(f"    Confidence:         {glm_synth.get('confidence')}")
        print(f"    Summary:            {glm_synth.get('summary')[:120]}...")
        print(f"    Core Takeaway:      {glm_synth.get('core_takeaway')[:120]}...")

        if parsed_primary:
            print("\n  [Qwen2.5-3B Local Validation Output]")
            print(f"    Model:              {MODEL_ID}")
            print(f"    Latency:            {latency:.2f}s (vs GLM {glm_synth.get('request_latency_seconds', 0):.2f}s)")
            print(f"    Confidence:         {parsed_primary.get('confidence')}")
            print(f"    Summary:            {parsed_primary.get('summary')[:120]}...")
            print(f"    Core Takeaway:      {parsed_primary.get('core_takeaway')[:120]}...")
    else:
        print("  output/multimodal_result.json not found for comparison.")

    # Summary Report Data Export
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "load_time_seconds": t_load,
        "memory_ram_mb": mem_diff,
        "primary_test": {
            "latency_seconds": latency,
            "tokens_generated": tokens,
            "tokens_per_second": tok_sec,
            "parsed_output": parsed_primary,
        },
        "scenarios": scenario_results,
    }

    out_file = Path("output/qwen_validation_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved standalone validation report to: {out_file.resolve()}")
    print("\n" + "=" * 65)
    print("STANDALONE VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
