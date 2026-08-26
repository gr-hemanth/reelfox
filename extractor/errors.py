"""Extraction failure categories.

A failed extraction must never masquerade as a success with blank fields, so
every failure carries an explicit, machine-readable category. Callers switch
on :class:`FailureCategory`; humans read the accompanying message.

The categories are ordered from "we never started" to "we could not explain
it". Pick the most specific one the evidence actually supports. In particular,
do not claim ``AUTH_REQUIRED`` or ``RATE_LIMITED`` unless the underlying
error text genuinely points there - a wrong category is worse than
``UNKNOWN`` because it sends the next debugging step in the wrong direction.
"""

from __future__ import annotations

from typing import Final


class FailureCategory:
    """Machine-readable extraction failure categories."""

    #: The URL never passed Phase 2 validation; extraction was not attempted.
    URL_VALIDATION: Final = "URL_VALIDATION"

    #: yt-dlp could not extract information for the URL (generic extractor
    #: failure that no more specific category explains).
    EXTRACTION: Final = "EXTRACTION"

    #: The post exists but no downloadable media was offered.
    MEDIA_UNAVAILABLE: Final = "MEDIA_UNAVAILABLE"

    #: Media came through but the caption could not be read. On its own this
    #: is a *partial* success, not a hard failure (see models.ExtractionResult).
    CAPTION_UNAVAILABLE: Final = "CAPTION_UNAVAILABLE"

    #: The media kind is not one this project supports.
    UNSUPPORTED_MEDIA: Final = "UNSUPPORTED_MEDIA"

    #: Instagram requires a logged-in session for this content.
    AUTH_REQUIRED: Final = "AUTH_REQUIRED"

    #: Instagram throttled the request.
    RATE_LIMITED: Final = "RATE_LIMITED"

    #: A transport-level problem (DNS, connection, timeout).
    NETWORK_ERROR: Final = "NETWORK_ERROR"

    #: Metadata was fine but writing/reading the media file failed, or the
    #: downloaded file did not survive post-download validation.
    DOWNLOAD_ERROR: Final = "DOWNLOAD_ERROR"

    #: yt-dlp returned an info object we could not parse into our model.
    METADATA_ERROR: Final = "METADATA_ERROR"

    #: cookie_file mode: the configured cookie file path does not exist.
    COOKIE_FILE_MISSING: Final = "COOKIE_FILE_MISSING"

    #: cookie_file mode: the cookie file exists but could not be read.
    COOKIE_FILE_UNREADABLE: Final = "COOKIE_FILE_UNREADABLE"

    #: cookie_file mode: the file is empty or yt-dlp rejects its format.
    COOKIE_FILE_INVALID: Final = "COOKIE_FILE_INVALID"

    #: Nothing above fits. Preserve the raw message and move on.
    UNKNOWN: Final = "UNKNOWN"

    ALL: Final = (
        URL_VALIDATION,
        EXTRACTION,
        MEDIA_UNAVAILABLE,
        CAPTION_UNAVAILABLE,
        UNSUPPORTED_MEDIA,
        AUTH_REQUIRED,
        RATE_LIMITED,
        NETWORK_ERROR,
        DOWNLOAD_ERROR,
        METADATA_ERROR,
        COOKIE_FILE_MISSING,
        COOKIE_FILE_UNREADABLE,
        COOKIE_FILE_INVALID,
        UNKNOWN,
    )


class ExtractionError(Exception):
    """Raised internally when an extraction step cannot continue.

    Carries a :class:`FailureCategory` and a safe, human-readable message.
    The extractor catches this and turns it into a failed
    :class:`~extractor.models.ExtractionResult`; it is not meant to escape the
    extraction layer.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.category}] {self.message}"
