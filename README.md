# Instagram Content Analyzer

An experimental CLI prototype exploring whether a public Instagram URL can
eventually be turned into accurate, structured, multimodal information.

**This repository is at Phase 8 of that experiment. Speech, Vision, OCR, and
Multimodal Synthesis have been integrated at ₹0 cost, but overall
multimodal reliability across diverse datasets has not yet been benchmarked.**

## Purpose

The long-term feasibility question is whether this pipeline can be made to
produce trustworthy structured output:

```
Instagram URL
  -> URL validation
  -> Instagram content extraction
  -> Extraction validation
  -> Audio / speech understanding (Phase 5: faster-whisper)
  -> Vision understanding         (Phase 6: local SmolVLM)
  -> OCR / on-screen text         (Phase 7: local RapidOCR)
  -> Multimodal synthesis         (Phase 8: Local Qwen2.5-3B-Instruct baseline at ₹0; GLM fallback)
  -> Structured JSON & Evaluation
```

Each stage is built and validated as its own phase so that a failure can be
attributed to a specific step rather than to the pipeline as a whole.

## Current phase

**Phase 8 — Multimodal Synthesis (Permanent Local Baseline Complete).**
Phases 1 (environment), 2 (URL validation), 3 (content extraction), 4 (authenticated extraction experiment),
5 (audio / speech understanding), 6 (vision understanding), 7 (OCR / on-screen text), and 8 (multimodal synthesis with permanent local `Qwen/Qwen2.5-3B-Instruct` baseline) are fully implemented and verified at ₹0 cost.

> [!NOTE]
> Detailed internal architecture records, phase-by-phase design documents, PRDs, and implementation notes reside in the internal `docs/` workspace and are intentionally untracked in version control.

---

## Phase 8 — Multimodal Synthesis

### Objective

Combine structured evidence extracted by:
1. Instagram metadata and caption
2. Speech / ASR (`faster-whisper`)
3. Vision (`SmolVLM-256M-Instruct`)
4. OCR (`RapidOCR`)

into one structured, useful, grounded interpretation.

> [!IMPORTANT]
> **No raw media is passed to the LLM.**
> The synthesis model does NOT receive raw video, audio, or images. It receives only structured textual evidence and observations produced by upstream phases.

### Synthesis Models: Permanent Local Baseline & Cloud Fallback

The synthesis layer supports two backends configurable via `SYNTHESIS_BACKEND` (`local` or `cloud`/`tokenrouter`):

#### 1. Permanent Local Baseline: Qwen/Qwen2.5-3B-Instruct (Default)

| Property | Value |
| --- | --- |
| **Model ID** | `Qwen/Qwen2.5-3B-Instruct` |
| **Execution** | **100% Local Inference (Hugging Face Transformers)** |
| **Device / Precision** | CPU, `torch.bfloat16` (multi-threaded on AMD Ryzen 7 7435HS) |
| **Memory Footprint** | ~6.37 GB RAM working set (fits comfortably in 16 GB system RAM) |
| **Authentication** | None (no API key, zero external cloud dependencies) |
| **Cost** | **₹0 (Completely free, permanent offline capability)** |
| **Input Format** | Structured textual evidence payload (`MultimodalEvidence`) |
| **Output Format** | Strict JSON object (`MultimodalAnalysisResult`) |

##### Why Qwen2.5-3B-Instruct is the permanent baseline:
1. **₹0 Forever**: Runs completely offline with zero API limits, expiration, or cloud costs.
2. **Schema Adherence**: 100% compliant JSON schema adherence validated across all edge cases without regex post-processing.
3. **Hardware Fit**: Uses ~6.37 GB RAM in `torch.bfloat16`, leaving plenty of room on a 16 GB machine.

#### 2. Cloud Reference / Fallback: z-ai/glm-5.3-free via TokenRouter

| Property | Value |
| --- | --- |
| **Model ID** | `z-ai/glm-5.3-free` |
| **Provider** | TokenRouter (`https://api.tokenrouter.com/v1/chat/completions`) |
| **Authentication** | `TOKENROUTER_API_KEY` (optional fallback) |
| **Cost** | ₹0 (Temporary free tier) |

### Anti-Hallucination Rules & Evidence Hierarchy

To guarantee factual grounding, the synthesizer enforces:
1. **Evidence Grounding**: Generate statements derived ONLY from supplied evidence. Do not invent facts, statistics, names, or events.
2. **No Pre-training Extrapolation**: Do not use parametric pre-training memories to fabricate specific facts about private people or unstated company events.
3. **Explicit Uncertainty**: If evidence is missing, conflicting, or ambiguous, explicitly note the limitation rather than guessing.
4. **Evidence Hierarchy**:
   - **Speech (ASR)**: Primary truth of what the speaker said.
   - **OCR Text**: Primary truth of slides, code, infographics, and burned-in graphics.
   - **Caption**: Author's context, hashtags, and claims.
   - **Vision (VLM)**: Visual scene description, subjects, and non-verbal actions.

### Schema & Structured Output

Outputs conform to `MultimodalAnalysisResult`:
```json
{
  "success": true,
  "summary": "<concise, factual summary strictly grounded in evidence>",
  "key_points": [
    "<factual point 1>",
    "<factual point 2>",
    "<factual point 3>"
  ],
  "core_takeaway": "<single primary conclusion or takeaway directly supported by evidence>",
  "relevant_context": "<relevant setting, context, technical topic, or format grounded in evidence>",
  "confidence": 0.85,
  "evidence_used": {
    "caption": true,
    "speech": true,
    "vision": true,
    "ocr": true
  },
  "model_name": "z-ai/glm-5.3-free",
  "processing_time_seconds": 12.34,
  "request_latency_seconds": 12.20,
  "prompt_tokens": 1240,
  "completion_tokens": 310,
  "total_tokens": 1550
}
```

### Evidence Traceability

The synthesizer explicitly tracks `evidence_used` deterministically:
- `caption`: `true` if caption was present and non-empty.
- `speech`: `true` if audio was present, speech detected, and transcript extracted.
- `vision`: `true` if keyframes were sampled and visual observations recorded.
- `ocr`: `true` if on-screen text was detected and OCR blocks extracted.

### Token & Latency Metrics

Measures and records:
- Request latency (`request_latency_seconds`)
- Total synthesis time (`processing_time_seconds`)
- Prompt token count (`prompt_tokens`)
- Completion token count (`completion_tokens`)
- Total token count (`total_tokens`)

### Testing Strategy

- **Offline Unit Tests** (`pytest tests/test_synthesis.py`):
  - 100% offline, zero network or model download calls.
  - Covers 18 unit test scenarios: Local Qwen success, malformed JSON, schema validation failure, empty output, cloud success, 401 unauthorized, timeout, 429 rate limit, 500 error, missing API key, missing evidence sources, long input truncation, confidence clamping, secret leakage prevention, token/latency metrics, and deterministic `evidence_used` tracking.
- **Local Synthesis Runner** (`scripts/test_local_synthesis.py`):
  - 100% local, ₹0 inference using cached `Qwen/Qwen2.5-3B-Instruct` on CPU.
  - Generates structured JSON output to `output/local_multimodal_result.json`.
- **Cloud Fallback Runner** (`scripts/test_synthesis.py`):
  - Optional fallback test using live TokenRouter GLM-5.3-free.
  - Outputs structured JSON to `output/multimodal_result.json`.

### Known Limitations

> **Important:**
> 1. **Zero-shot Synthesis**: A single successful synthesis proves technical connectivity and schema adherence, but does **NOT** establish an overall ≥85% benchmark accuracy across diverse Instagram genres. Broad evaluation is deferred to the benchmark phase.
> 2. **Provider Rate Limits / Latency**: Free public endpoints on TokenRouter may experience variable request queueing or latency during peak usage.
> 3. **Input Length Bounds**: Conservative truncation bounds are applied to captions (1,000 chars), transcripts (4,000 chars), and OCR text (4,000 chars) to prevent context window exhaustion.

---

---

## Phase 7 — OCR / On-Screen Text

### Objective

Build a dedicated, local OCR layer that extracts important visible on-screen text from:
1. Instagram single-image posts
2. Instagram carousel images
3. Instagram video/Reel frames

Extracted text includes subtitles, burned-in captions, overlays, technical terms, code, numbers, headings, instructions, and labels.

> [!IMPORTANT]
> **Vision and OCR are strictly separate layers.**
> The vision model from Phase 6 (VLM) is NOT being treated as the OCR engine. Vision analyzes visual semantics and scene understanding, while OCR specializes in high-fidelity optical character recognition and bounding box localization.

### Why Dedicated OCR is Needed

While Vision-Language Models (VLMs) can provide general visual descriptions, they frequently hallucinate, omit small text, misread technical identifiers, and struggle with dense multi-line slides or rapidly changing subtitle overlays. A dedicated deep-learning OCR detector (DBNet) and recognizer (CRNN/SVTR) guarantees precise text extraction, coordinates, confidence scores, and fast CPU execution.

### OCR Technology Selection

- **Selected Engine**: `RapidOCR` (`rapidocr-onnxruntime` 1.2.3)
- **Underlying Models**: PP-OCRv4 (DBNet text detection + SVTR/CRNN recognition + direction classifier in ONNX format)
- **Runtime**: `onnxruntime` 1.29.0
- **Cost**: **₹0 (Free, 100% Local Inference)**
- **API Keys Required**: **None**

#### Why RapidOCR was selected:
1. **Zero External System Binaries**: Unlike Tesseract (which requires external Windows installer `.exe` and manual PATH configuration), RapidOCR is purely Python-accessible with bundled ONNX models.
2. **Pre-existing Dependency Reuse**: Runs on `onnxruntime`, which was already installed in the environment for `faster-whisper`.
3. **High Accuracy on Overlays & Subtitles**: Employs deep-learning DBNet polygon detection and text direction classification, excelling at scene text, tilted banners, burned-in video captions, and screen code.
4. **Fast CPU Performance**: Highly optimized ONNX inference (~0.6s per frame on CPU), 5x faster than heavier PyTorch-based alternatives like EasyOCR.
5. **Python 3.14 Compatibility**: Fully compatible with Python 3.14 on Windows 64-bit with pre-built wheels.

### Supported Inputs

- **Single Image**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (analyzed directly as a single frame).
- **Video / Reel**: `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.m4v` (sampled deterministically into keyframes).
- **Carousel**: Sequence of image files, analyzed sequentially with per-page index preservation.

### Frame Sampling Strategy

- For videos/Reels, representative frames are sampled evenly across media duration using `PyAV` (`av`).
- Maximum frames configurable via `OCR_MAX_FRAMES` (default: `6`).
- Samples are created as short-lived temporary files and **strictly cleaned up** immediately after processing in all scenarios (success, failure, no text, or exceptions).
- Source Instagram media is never modified or deleted by the OCR layer.

### Deduplication & Text Normalization

- **Light Normalization**: Whitespace collapsing, stripping leading/trailing whitespace, while preserving case, punctuation, numbers, and technical symbols. No LLM "hallucinated correction" is performed.
- **Conservative Multi-Frame Deduplication**: In Reels, subtitles and watermarks persist across multiple sampled frames. Duplicate text across frames is deduplicated in the top-level `text_blocks` and `combined_text` (retaining earliest frame index, timestamp, and confidence), while raw per-frame detections are fully retained in `per_frame_results`. Distinct text blocks are never merged.

### Confidence Handling

- Confidence scores provided by the engine `[0.0, 1.0]` are preserved as floating-point values.
- If confidence is unavailable, it remains `null`; numbers are never fabricated.

### Structured OCRResult Schema

```json
{
  "success": true,
  "media_type": "video",
  "media_path": "temp/run/DcWXVZlMwOB.mp4",
  "frames_analyzed": 6,
  "text_detected": true,
  "text_blocks": [
    {
      "text": "AIRESEARCHENGINEER-2026",
      "confidence": 0.90,
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "bbox": [[196.0, 198.0], [411.0, 198.0], [411.0, 214.0], [196.0, 214.0]]
    }
  ],
  "combined_text": "AIRESEARCHENGINEER-2026\nSARVAM\n84 LPA\n...",
  "per_frame_results": [ ... ],
  "model_name_or_engine": "rapidocr",
  "frame_extraction_seconds": 0.147,
  "model_load_seconds": 0.522,
  "inference_seconds": 4.188,
  "processing_time_seconds": 4.860,
  "failure_category": null,
  "failure_message": null
}
```

Distinguishes clearly between:
- "No visible text found": `success=True`, `text_detected=False`, `text_blocks=[]`, `failure_category=None`.
- "OCR failed": `success=False`, `failure_category="OCR_INFERENCE_FAILED"`.

### Real Experiment Results

Tested on the project benchmark Reel: `https://www.instagram.com/p/DcWXVZlMwOB/`
- **Frames Analyzed**: 6 sampled frames (t = 0.0s, 5.0s, 9.9s, 14.9s, 19.8s, 24.7s)
- **Key Text Detected**:
  - Title & Roles: `AIRESEARCHENGINEER-2026`, `SARVAM`, `84 LPA`, `INTERVIEW`
  - Technical slide details: `You fine-tune an open English-centric 8B model on a large Hindi corpus. Quality on Hindi benchmarks is respectable and the team is happy. Then serving costs land: roughly 4x per request versus your English traffic, and long documents that fit comfortably in English overflow the context window in Hindi. You have changed nothing about the architecture, the sequence length, or the hardware.`
  - Metrics & Salary Chart: `300000`, `259K`, `200000`, `158K`, `150K`, `121K`, `135K`, `100000`, `112 K`, `51K`, `Low`, `Avarage`, `High`, `Glassdoor`, `PayScale`, `Indeed`
  - Company Logos / Sponsors: `Al for all from India`, `CRED`, `DECENTRO`, `CredResolve`
  - Burned-in Subtitles: `ifyouhaveever`, `their AI research`, `quality on Hindi`, `request versus your`, `use cases`
- **Timing**:
  - Frame Extraction: 0.147s
  - Model Load: 0.522s
  - Inference (6 frames): 4.188s
  - Total Processing: 4.860s

### Known Limitations

1. **Space Tokenization in Dense Graphics**: When text in infographic banners is rendered without clear spacing, bounding boxes can merge words (e.g. `Youfine-tunean open English-centric8Bmodel`).
2. **Motion-Blurred Subtitles**: Fast-moving animated subtitles occasionally show minor character substitutions (e.g. `Hindt*` for `Hindi`).
3. **Partial Screen-Edge Text**: Words clipped by screen margins or foreground speaker occlusion may miss initial characters (e.g. `NTERVIE` for `INTERVIEW`).

---


---

## Phase 6 — Vision Understanding

### Objective

Determine whether extracted Instagram images or video keyframes can be understood
locally with sufficient accuracy to identify main subjects, visible objects,
actions, scene context, and visual demonstrations — without using paid vision APIs.

### Hardware Inspection & Environment

- **CPU**: AMD Ryzen 7 7435HS (8 Cores, 16 Threads)
- **RAM**: 16 GB DDR5
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM)
- **Available Disk Space**: ~99.3 GB free
- **Python**: 3.14.3 (64-bit)

### Baseline Model Selection: HuggingFaceTB/SmolVLM-256M-Instruct

| Property | Value |
| --- | --- |
| **Official Model** | `HuggingFaceTB/SmolVLM-256M-Instruct` (~256M parameters) |
| **Runtime** | `transformers` (`AutoModelForImageTextToText` + `AutoProcessor`) |
| **Device** | `auto` (`cuda` if available, `cpu` fallback) |
| **Cost** | **₹0 (Free, 100% Local Inference)** |
| **API Keys Required** | **None** |

#### Why SmolVLM-256M-Instruct was selected:
1. **Ultra-lightweight Footprint**: At ~256M parameters (~500 MB model size), it loads rapidly and runs comfortably on low-resource VRAM and CPU hardware.
2. **Local & Free**: 100% open-source local inference; no API keys, no paid cloud providers.
3. **High Visual Understanding**: Produces detailed natural language descriptions of subjects, objects, actions, and scenes.
4. **Clean Integration**: Native `transformers` support via `AutoModelForImageTextToText`.

### Frame Sampling Strategy

To understand Reels and videos without overcomplicating processing:
- Video frames are sampled deterministically across the media duration using `PyAV` (`av`).
- Number of frames is configurable via `VISION_MAX_FRAMES` (default: `6`).
- Extracted frames are stored as short-lived temporary JPEGs and **automatically deleted** immediately after vision processing.
- Single image posts skip frame extraction and are passed directly.

### VisionResult Schema

The visual understanding layer outputs a structured `VisionResult`:

```json
{
  "success": true,
  "media_path": "temp/run/video.mp4",
  "input_type": "video",
  "frames_analyzed": 6,
  "subjects": ["hand", "man", "people"],
  "objects": ["camera", "computer", "paper", "phone", "table", "text"],
  "actions": ["holding", "looking", "standing"],
  "scenes": [],
  "demonstrations": [],
  "observations": [
    "The main subject of the image is a boy in the foreground..."
  ],
  "frame_observations": [...],
  "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
  "frame_extraction_seconds": 0.147,
  "model_load_seconds": 197.33,
  "inference_seconds": 97.40,
  "total_processing_seconds": 294.88
}
```

`VisionResult` is completely independent from SpeechResult and OCR; it contains no OCR or LLM synthesis fields.

### Conservative Visual Observation Rule

Prompts ask the model to report **only what is visibly supported**:
> *"Describe only what is visibly supported by this image. Identify the main subject, visible objects, action, and scene context. If something is uncertain, say uncertain."*

The model is instructed not to guess hidden or unverified context.

### Testing Strategy

- **Offline unit tests** (`pytest tests/test_vision.py`):
  - 100% offline, network-free.
  - Mocks vision analyzer model.
  - Tests image processing, frame sampling, invalid media, frame extraction failure, model load failure, model inference failure, max frames configuration, temp frame cleanup, JSON serialization, and secret leakage safety.
- **Live integration runner** (`scripts/test_vision.py`):
  - Separate networked script for live model testing against real Instagram media.

```bash
# Run 100% offline pytest suite
pytest

# Run live vision test on real extracted Instagram Reel
python scripts/test_vision.py "https://www.instagram.com/p/DcWXVZlMwOB/"
```

### Known Limitations

> **Important:**
> 1. **Processing Time**: Single-video total processing time currently averages ~295s (including initial HuggingFace Hub load and multi-frame sequential VLM inference). Optimization is deferred to a future pass.
> 2. **Benchmark Accuracy**: A single successful vision test proves technical viability on local hardware, but does **NOT** establish the project's long-term target of ≥85% visual understanding accuracy. Reliability will be evaluated during the multi-sample benchmark phase.

---

## Phase 5 — Audio / Speech Understanding

### Stack

- **Engine**: `faster-whisper` (1.2.1)
- **Model**: `base` (CPU int8)
- **Cost**: **₹0 (Free, Local ASR)**
- Distinguishes `no_audio`, `audio_no_speech`, `speech`, and `mixed_or_uncertain`.

---

## Phase 4 — Authenticated Extraction Experiment

- Proven viable using `EXTRACTION_MODE=cookie_file` with Netscape-format cookie files (`secrets/instagram_cookies.txt`).
- Downloaded real media + captions in 5.324s.

---

## Setup & Configuration

```bash
# Activate virtual environment
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Environment options (`.env`):

```env
EXTRACTION_MODE=public
COOKIE_FILE=

ASR_MODEL_SIZE=base
ASR_DEVICE=cpu

VISION_MODEL=HuggingFaceTB/SmolVLM-256M-Instruct
VISION_DEVICE=auto
VISION_MAX_FRAMES=6

OCR_ENGINE=rapidocr
OCR_MAX_FRAMES=6

SYNTHESIS_BACKEND=local
SYNTHESIS_MODEL=Qwen/Qwen2.5-3B-Instruct
SYNTHESIS_DEVICE=cpu
TOKENROUTER_API_KEY=
SYNTHESIS_ENDPOINT=https://api.tokenrouter.com/v1/chat/completions
SYNTHESIS_ENABLED=true
SYNTHESIS_TIMEOUT=60
```

## Running the CLI

```bash
# Full multimodal pipeline (Speech + Vision + OCR + Local Synthesis)
python analyzer.py "https://www.instagram.com/p/DcWXVZlMwOB/"
```

Options:

```bash
python analyzer.py --help
python analyzer.py --skip-synthesis "<url>"   # skip Phase 8 Synthesis
python analyzer.py --synthesis-only "<url>"   # run extraction + synthesis directly
python analyzer.py --ocr-only "<url>"         # run only Phase 7 OCR
python analyzer.py --skip-ocr "<url>"         # skip Phase 7 OCR
python analyzer.py --skip-speech "<url>"      # skip Phase 5 ASR
python analyzer.py --skip-vision "<url>"      # skip Phase 6 Vision
python analyzer.py --keep-media "<url>"       # retain downloaded media
```

## Project Structure

```
.
├── analyzer.py          # CLI entry point (Phases 1-8)
├── config.py            # Configuration + logging
├── requirements.txt
├── .env.example
├── README.md
├── extractor/                 # Phase 2 validation + Phase 3/4 extraction
│   ├── url_validator.py       # Phase 2: offline URL validation
│   ├── instagram_extractor.py # Phase 3: yt-dlp extraction backend
│   ├── models.py              # ExtractionResult, media/mode enums
│   ├── errors.py              # FailureCategory, ExtractionError
│   └── artifacts.py           # per-run temp directory manager
├── processor/                 # Phase 5 (Speech), Phase 6 (Vision), Phase 7 (OCR), Phase 8 (Synthesis)
│   ├── __init__.py
│   ├── models.py              # SpeechResult, VisionResult, OCRResult, OCRTextBlock
│   ├── synthesis_models.py    # MultimodalEvidence, MultimodalAnalysisResult, SynthesisFailureCategory
│   ├── audio.py               # audio extraction (PyAV)
│   ├── speech.py              # ASR abstraction + faster-whisper backend
│   ├── frames.py              # keyframe sampler (PyAV)
│   ├── vision.py              # Vision abstraction + SmolVLM backend
│   ├── ocr.py                 # OCR abstraction + RapidOCR backend + dedup
│   ├── synthesis.py           # Synthesis abstraction + Local Qwen backend (GLM fallback)
│   └── pipeline.py            # process_speech(), process_vision(), process_ocr(), process_synthesis()
├── scripts/
│   ├── test_extraction.py     # live extraction runner
│   ├── test_audio.py          # live ASR runner
│   ├── test_vision.py         # live Vision runner
│   ├── test_ocr.py            # live OCR runner
│   ├── test_synthesis.py      # live cloud Synthesis runner
│   ├── test_local_synthesis.py# live local Qwen Synthesis runner
│   └── validate_local_qwen.py # standalone Qwen validation harness
└── tests/
    ├── test_setup.py          # project + CLI smoke tests (offline)
    ├── test_url_validator.py  # validator unit tests (offline)
    ├── test_extractor.py      # extraction unit tests (offline)
    ├── test_speech.py         # speech unit tests (offline)
    ├── test_vision.py         # vision unit tests (offline)
    ├── test_ocr.py            # OCR unit tests (offline)
    └── test_synthesis.py      # synthesis unit tests (offline)
```

## Status

Phases 1 (environment), 2 (URL validation), 3 (content extraction), 4 (authenticated extraction), 5 (audio / speech understanding), 6 (vision understanding), 7 (OCR / on-screen text), and 8 (multimodal synthesis with permanent local Qwen baseline) are **complete**.

- **Speech**: Local ASR (`faster-whisper` base) extracts & transcribes audio at ₹0 cost.
- **Vision**: Local VLM (`SmolVLM-256M-Instruct`) samples & understands keyframes at ₹0 cost.
- **OCR**: Local OCR (`RapidOCR` / PP-OCRv4 ONNX) detects & reads on-screen text/subtitles with bounding boxes & confidence at ₹0 cost.
- **Synthesis**: Local Multimodal synthesis (`Qwen/Qwen2.5-3B-Instruct` permanent baseline, GLM fallback) grounds metadata, speech, vision, and OCR evidence into structured JSON at ₹0 cost.
- **Remaining**: Phase 9+ (evaluation preparation, 20-URL benchmark, database, Telegram bot) has not been started.


