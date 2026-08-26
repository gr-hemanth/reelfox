"""Data models for the extraction layer.

These structures hold *extraction* facts only - what media and metadata came
back from Instagram. They deliberately contain nothing about AI analysis
(transcripts, captions-of-images, summaries); those belong to later phases and
get their own models so the two concerns never blur together.

Everything here is a plain dataclass that serialises cleanly to a dict (and
therefore to JSON) via :meth:`ExtractionResult.as_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ExtractionMode(str, Enum):
    """How the extractor is allowed to authenticate.

    ``PUBLIC`` is the default and requires no Instagram session. ``BROWSER_COOKIES``
    lets yt-dlp borrow a local browser's cookies. ``COOKIE_FILE`` points yt-dlp
    at a local Netscape-format cookie file. No password is ever involved in any
    mode; the app only ever passes a browser name or a file path onward.
    """

    PUBLIC = "public"
    BROWSER_COOKIES = "browser_cookies"
    COOKIE_FILE = "cookie_file"

    @classmethod
    def from_string(cls, value: str | None) -> "ExtractionMode":
        """Parse a configured mode string, defaulting to PUBLIC."""
        if not value:
            return cls.PUBLIC
        normalised = value.strip().lower()
        for mode in cls:
            if mode.value == normalised:
                return mode
        return cls.PUBLIC


class MediaType(str, Enum):
    """Media kinds this project understands.

    ``UNKNOWN`` is a first-class value: when extraction cannot establish the
    type we record ``UNKNOWN`` rather than inventing one.
    """

    REEL = "reel"
    VIDEO = "video"
    CAROUSEL = "carousel"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class MediaFile:
    """One downloaded media artifact on local disk."""

    path: str
    kind: str  # "video" or "image"
    size_bytes: int = 0
    ext: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "ext": self.ext,
        }


@dataclass
class ExtractionResult:
    """The structured outcome of one extraction attempt.

    Success is not all-or-nothing. ``success`` means "the run produced media
    usable by the next stage"; caption and media-type detection are tracked by
    their own boolean flags so a partial result is legible rather than silently
    lossy. A run that retrieved a caption but no media is *not* a success.
    """

    success: bool
    source_url: str
    normalized_url: Optional[str] = None

    # Media -----------------------------------------------------------------
    media_downloaded: bool = False
    media_path: Optional[str] = None
    media_files: list[MediaFile] = field(default_factory=list)

    # Caption ---------------------------------------------------------------
    caption_extracted: bool = False
    caption: Optional[str] = None
    hashtags: list[str] = field(default_factory=list)

    # Type ------------------------------------------------------------------
    media_type_detected: bool = False
    media_type: str = MediaType.UNKNOWN.value
    content_type_hint: Optional[str] = None  # from the Phase 2 validator

    # Everything else yt-dlp gave us that we chose to keep (sanitised).
    metadata: dict[str, Any] = field(default_factory=dict)

    # Failure ---------------------------------------------------------------
    failure_category: Optional[str] = None
    failure_reason: Optional[str] = None
    error_detail: Optional[str] = None

    # Timing ----------------------------------------------------------------
    download_seconds: Optional[float] = None
    extraction_mode: str = ExtractionMode.PUBLIC.value
    cookie_file_configured: bool = False
    run_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary of the whole result."""
        return {
            "success": self.success,
            "source_url": self.source_url,
            "normalized_url": self.normalized_url,
            "media_downloaded": self.media_downloaded,
            "media_path": self.media_path,
            "media_files": [m.as_dict() for m in self.media_files],
            "caption_extracted": self.caption_extracted,
            "caption": self.caption,
            "hashtags": list(self.hashtags),
            "media_type_detected": self.media_type_detected,
            "media_type": self.media_type,
            "content_type_hint": self.content_type_hint,
            "metadata": self.metadata,
            "failure_category": self.failure_category,
            "failure_reason": self.failure_reason,
            "error_detail": self.error_detail,
            "download_seconds": self.download_seconds,
            "extraction_mode": self.extraction_mode,
            "cookie_file_configured": self.cookie_file_configured,
            "run_id": self.run_id,
        }

    def summary_line(self) -> str:
        """A single compact line, handy for benchmark logs."""
        if self.success:
            return (
                f"OK url={self.source_url} type={self.media_type} "
                f"media={self.media_downloaded} caption={self.caption_extracted} "
                f"t={self.download_seconds}"
            )
        return (
            f"FAIL url={self.source_url} category={self.failure_category} "
            f"reason={self.failure_reason}"
        )
