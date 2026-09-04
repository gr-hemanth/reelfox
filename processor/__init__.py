"""Multimodal processing — Phase 5 (Audio/Speech) and Phase 6 (Vision Understanding).

This package provides:

- ``processor.models``: structured results (``SpeechResult``, ``SpeechSegment``,
  ``SpeechClassification``, ``SpeechFailureCategory``, ``VisionResult``,
  ``FrameObservation``, ``VisionFailureCategory``).
- ``processor.audio``: audio stream detection and extraction to temporary WAV.
- ``processor.speech``: ASR abstraction (``BaseSpeechRecognizer``) and the
  ``FasterWhisperRecognizer`` backend.
- ``processor.frames``: keyframe sampling for images and videos.
- ``processor.vision``: Vision abstraction (``BaseVisionAnalyzer``) and the
  ``LocalVisionAnalyzer`` VLM backend.
- ``processor.pipeline``: end-to-end ``process_speech`` and ``process_vision`` functions.

Later phases will add OCR, multimodal synthesis and structured output.
"""

from processor.frames import sample_frames
from processor.models import (
    FrameObservation,
    OCRFailureCategory,
    OCRFrameResult,
    OCRResult,
    OCRTextBlock,
    SpeechClassification,
    SpeechFailureCategory,
    SpeechResult,
    SpeechSegment,
    VisionFailureCategory,
    VisionResult,
)
from processor.ocr import (
    BaseOCRAnalyzer,
    LocalOCRAnalyzer,
    deduplicate_text_blocks,
    normalize_text,
)
from processor.pipeline import (
    process_ocr,
    process_speech,
    process_synthesis,
    process_vision,
)
from processor.benchmark import (
    BenchmarkDecision,
    BenchmarkEvaluation,
    BenchmarkRecord,
    evaluate_benchmark,
    load_ground_truth,
)
from processor.output import sanitize_dict_for_output, save_run_result
from processor.run_result import RunResult, build_stage_metrics, run_pipeline
from processor.speech import BaseSpeechRecognizer, FasterWhisperRecognizer
from processor.synthesis import (
    BaseSynthesizer,
    LocalQwenSynthesizer,
    TokenRouterGLMSynthesizer,
)
from processor.synthesis_models import (
    MultimodalAnalysisResult,
    MultimodalEvidence,
    SynthesisFailureCategory,
)
from processor.vision import BaseVisionAnalyzer, LocalVisionAnalyzer

__all__ = [
    # Models - Speech
    "SpeechClassification",
    "SpeechFailureCategory",
    "SpeechResult",
    "SpeechSegment",
    # Models - Vision
    "VisionResult",
    "FrameObservation",
    "VisionFailureCategory",
    # Models - OCR
    "OCRResult",
    "OCRTextBlock",
    "OCRFrameResult",
    "OCRFailureCategory",
    # Models - Synthesis
    "MultimodalAnalysisResult",
    "MultimodalEvidence",
    "SynthesisFailureCategory",
    # Phase 9 Benchmark & Run Result
    "RunResult",
    "run_pipeline",
    "build_stage_metrics",
    "save_run_result",
    "sanitize_dict_for_output",
    "BenchmarkRecord",
    "BenchmarkEvaluation",
    "BenchmarkDecision",
    "evaluate_benchmark",
    "load_ground_truth",
    # Speech
    "BaseSpeechRecognizer",
    "FasterWhisperRecognizer",
    # Vision
    "BaseVisionAnalyzer",
    "LocalVisionAnalyzer",
    "sample_frames",
    # OCR
    "BaseOCRAnalyzer",
    "LocalOCRAnalyzer",
    "normalize_text",
    "deduplicate_text_blocks",
    # Synthesis
    "BaseSynthesizer",
    "LocalQwenSynthesizer",
    "TokenRouterGLMSynthesizer",
    # Pipeline
    "process_speech",
    "process_vision",
    "process_ocr",
    "process_synthesis",
]


