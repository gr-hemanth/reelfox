"""Instagram content extraction.

* Phase 2: offline URL validation (:mod:`extractor.url_validator`).
* Phase 3: content extraction via yt-dlp (:mod:`extractor.instagram_extractor`),
  with structured results (:mod:`extractor.models`), failure categories
  (:mod:`extractor.errors`) and temporary artifact handling
  (:mod:`extractor.artifacts`).

Later multimodal analysis (OCR, ASR, vision, LLM) is not implemented.
"""

from extractor.artifacts import TempRun
from extractor.errors import ExtractionError, FailureCategory
from extractor.instagram_extractor import (
    BaseExtractor,
    ExtractionOptions,
    YtDlpExtractor,
    classify_error,
)
from extractor.models import (
    ExtractionMode,
    ExtractionResult,
    MediaFile,
    MediaType,
)
from extractor.url_validator import (
    ContentTypeHint,
    ErrorCode,
    ValidationResult,
    validate_instagram_url,
)

__all__ = [
    # Phase 2
    "ContentTypeHint",
    "ErrorCode",
    "ValidationResult",
    "validate_instagram_url",
    # Phase 3
    "BaseExtractor",
    "YtDlpExtractor",
    "ExtractionOptions",
    "ExtractionResult",
    "ExtractionMode",
    "MediaFile",
    "MediaType",
    "ExtractionError",
    "FailureCategory",
    "TempRun",
    "classify_error",
]
