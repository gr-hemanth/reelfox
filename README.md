# Instagram Content Analyzer

An experimental CLI prototype exploring whether a public Instagram URL can
eventually be turned into accurate, structured, multimodal information.

**This repository is at Phase 6 of that experiment. Speech and Vision
understanding have been integrated locally at ₹0 cost, but overall visual and
multimodal reliability have not been benchmarked.**

## Purpose

The long-term feasibility question is whether this pipeline can be made to
produce trustworthy structured output:

```
Instagram URL
  -> URL validation
  -> Instagram content extraction
  -> Extraction validation
  -> Audio / speech understanding (Phase 5: faster-whisper)
  -> Vision understanding         (Phase 6: local VLM)
  -> OCR / on-screen text         (Phase 7: planned)
  -> Multimodal synthesis         (Phase 8: planned)
  -> Structured JSON              (Phase 9: planned)
  -> Cleanup
  -> Evaluation
```

Each stage is built and validated as its own phase so that a failure can be
attributed to a specific step rather than to the pipeline as a whole.

## Current phase

**Phase 6 - Vision Understanding (complete).** Phases 1 (environment), 2 (URL
validation), 3 (content extraction), 4 (authenticated extraction experiment), and
5 (audio / speech understanding) are also complete.

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
EXTRACTION_MODE=cookie_file
COOKIE_FILE=secrets/instagram_cookies.txt

ASR_MODEL_SIZE=base
ASR_DEVICE=cpu

VISION_MODEL=HuggingFaceTB/SmolVLM-256M-Instruct
VISION_DEVICE=auto
VISION_MAX_FRAMES=6
```

## Running the CLI

```bash
python analyzer.py "https://www.instagram.com/p/DcWXVZlMwOB/"
```

Options:

```bash
python analyzer.py --help
python analyzer.py --skip-speech "<url>"
python analyzer.py --skip-vision "<url>"
python analyzer.py --keep-media "<url>"
```

## Project Structure

```
.
├── analyzer.py          # CLI entry point
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
├── processor/                 # Phase 5 (Speech) & Phase 6 (Vision)
│   ├── __init__.py
│   ├── models.py              # SpeechResult, VisionResult, FrameObservation
│   ├── audio.py               # audio extraction (PyAV)
│   ├── speech.py              # ASR abstraction + faster-whisper backend
│   ├── frames.py              # keyframe sampler (PyAV)
│   ├── vision.py              # Vision abstraction + Moondream2 backend
│   └── pipeline.py            # process_speech() & process_vision()
├── scripts/
│   ├── test_extraction.py     # live extraction runner
│   ├── test_audio.py          # live ASR runner
│   └── test_vision.py         # live Vision runner
└── tests/
    ├── test_setup.py          # project + CLI smoke tests (offline)
    ├── test_url_validator.py  # validator unit tests (offline)
    ├── test_extractor.py      # extraction unit tests (offline)
    ├── test_speech.py         # speech unit tests (offline)
    └── test_vision.py         # vision unit tests (offline)
```

## Status

Phases 1 (environment), 2 (URL validation), 3 (content extraction), 4 (authenticated extraction), 5 (audio / speech understanding), and 6 (vision understanding) are **complete**.

- **Speech**: Local ASR (`faster-whisper` base) extracts & transcribes audio at ₹0 cost.
- **Vision**: Local VLM (`vikhyatk/moondream2`) samples & understands keyframes at ₹0 cost.
- **Remaining**: Phase 7+ (OCR, multimodal LLM synthesis, structured JSON, Telegram, database) has not been started.
