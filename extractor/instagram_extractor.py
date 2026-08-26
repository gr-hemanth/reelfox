"""Instagram content extraction, primary backend: yt-dlp.

This is the Phase 3 feasibility experiment. Given a URL that already passed the
Phase 2 offline validator, it attempts to retrieve the media file, caption,
hashtags, media type and whatever metadata yt-dlp exposes - and, crucially,
reports honestly when it cannot.

Design notes
------------
* The rest of the app talks to :class:`BaseExtractor`, not to yt-dlp. A second
  backend can be dropped in later (if the benchmark shows yt-dlp is not enough)
  without touching the CLI.
* yt-dlp is injected as a factory (``ydl_factory``) so unit tests can supply a
  fake and never touch the network. Nothing in this module imports yt-dlp at
  import time.
* Network access lives *only* here. The validator and CLI stay offline.
* Failures are classified into :class:`~extractor.errors.FailureCategory`. We
  never upgrade a guess into ``AUTH_REQUIRED`` or ``RATE_LIMITED`` unless the
  error text supports it.
* Metadata is sanitised before it leaves this module: no cookies, headers,
  formats blob or session values are ever retained, logged or printed.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from extractor.artifacts import TempRun
from extractor.errors import ExtractionError, FailureCategory
from extractor.models import (
    ExtractionMode,
    ExtractionResult,
    MediaFile,
    MediaType,
)
from extractor.url_validator import ContentTypeHint, ValidationResult

logger = logging.getLogger("analyzer.extractor")

# File extensions we recognise, grouped by kind.
_VIDEO_EXTS = frozenset({"mp4", "mkv", "webm", "mov", "m4v", "3gp"})
_IMAGE_EXTS = frozenset({"jpg", "jpeg", "png", "webp", "heic", "gif"})

# Metadata keys worth keeping. Everything else (formats, http_headers,
# cookies, thumbnails payloads, raw urls that may carry tokens) is dropped.
_METADATA_WHITELIST = (
    "id",
    "title",
    "ext",
    "duration",
    "width",
    "height",
    "fps",
    "uploader",
    "uploader_id",
    "channel",
    "channel_id",
    "timestamp",
    "upload_date",
    "like_count",
    "comment_count",
    "view_count",
    "track",
    "artist",
    "extractor",
    "extractor_key",
    "playlist_count",
)

_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")

# yt-dlp colourises its error strings; strip ANSI escapes so reports and the
# benchmark file stay plain text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class ExtractionOptions:
    """Runtime knobs for one extraction call.

    ``mode`` selects the authentication strategy. ``cookies_from_browser``
    names a browser (e.g. ``"chrome"``) and is only consulted in
    ``BROWSER_COOKIES`` mode. ``cookie_file`` is a local Netscape-format cookie
    file path, consulted only in ``COOKIE_FILE`` mode. ``keep_media`` retains
    the temp directory for debugging. ``timeout`` bounds the socket wait.

    The cookie file is never read, parsed, copied or logged by this project;
    only its path is handed to yt-dlp.
    """

    def __init__(
        self,
        mode: ExtractionMode = ExtractionMode.PUBLIC,
        cookies_from_browser: Optional[str] = None,
        cookie_file: Optional[str] = None,
        keep_media: bool = False,
        timeout: int = 60,
    ) -> None:
        self.mode = mode
        self.cookies_from_browser = cookies_from_browser
        self.cookie_file = cookie_file
        self.keep_media = keep_media
        self.timeout = timeout

    @property
    def cookie_file_configured(self) -> bool:
        """True when cookie_file mode has a path set (not whether it exists)."""
        return self.mode == ExtractionMode.COOKIE_FILE and bool(self.cookie_file)


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------


class BaseExtractor:
    """Interface every extraction backend implements."""

    name = "base"

    def extract(
        self,
        validation: ValidationResult,
        options: Optional[ExtractionOptions] = None,
    ) -> ExtractionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_error(message: str) -> str:
    """Map a raw yt-dlp/transport error message to a FailureCategory.

    Deterministic and conservative. When several signals appear we prefer the
    most actionable and specific reading, but we never claim a category the
    text does not support.
    """
    text = (message or "").lower()

    # Cookie-file format problems are a configuration fault, not an Instagram
    # auth rejection. Detect them first so they are never mislabelled
    # AUTH_REQUIRED. These yt-dlp messages are unambiguous.
    if any(
        token in text
        for token in (
            "does not look like a netscape",
            "netscape format cookies",
            "error loading cookies",
            "could not load cookies",
            "failed to load cookies",
        )
    ):
        return FailureCategory.COOKIE_FILE_INVALID

    # Unambiguous throttling signals first.
    if "429" in text or "too many requests" in text:
        return FailureCategory.RATE_LIMITED

    # Authentication. Instagram often emits the combined phrase
    # "rate-limit reached or login required"; login is the actionable path, so
    # we route that to AUTH_REQUIRED while preserving the full message.
    if any(
        token in text
        for token in (
            "login required",
            "log in",
            "logged-in",
            "logged in",
            "sign in",
            "requires authentication",
            "authenticat",  # authenticate / authentication
            "this account is private",
            "private account",
            "you need to log",
            "cookies-from-browser",
            "cookies for the authentication",
            # Instagram's current public-post block for logged-out clients:
            # "Instagram sent an empty media response ... use --cookies..."
            "empty media response",
        )
    ):
        return FailureCategory.AUTH_REQUIRED

    if "rate-limit" in text or "rate limit" in text:
        return FailureCategory.RATE_LIMITED

    if any(
        token in text
        for token in (
            "unable to download webpage",
            "connection",
            "timed out",
            "timeout",
            "getaddrinfo",
            "network is unreachable",
            "name or service not known",
            "connection reset",
            "temporary failure in name resolution",
            "ssl",
        )
    ):
        return FailureCategory.NETWORK_ERROR

    if any(
        token in text
        for token in (
            "not available",
            "unavailable",
            "removed",
            "has been deleted",
            "does not exist",
            "no longer exists",
            "404",
            "not found",
        )
    ):
        return FailureCategory.MEDIA_UNAVAILABLE

    if any(
        token in text
        for token in (
            "unsupported url",
            "no video formats",
            "no media",
            "there is no video",
        )
    ):
        return FailureCategory.UNSUPPORTED_MEDIA

    return FailureCategory.EXTRACTION


# ---------------------------------------------------------------------------
# yt-dlp backend
# ---------------------------------------------------------------------------


class YtDlpExtractor(BaseExtractor):
    """Extraction backed by yt-dlp.

    Parameters
    ----------
    temp_dir:
        Base directory for per-run temporary artifacts (usually ``temp/``).
    ydl_factory:
        Callable ``factory(options_dict) -> ydl`` where ``ydl`` supports the
        context-manager protocol and ``extract_info(url, download=True)``.
        Defaults to the real ``yt_dlp.YoutubeDL``; tests inject a fake.
    """

    name = "yt-dlp"

    def __init__(
        self,
        temp_dir: Path,
        ydl_factory: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.temp_dir = Path(temp_dir)
        self._ydl_factory = ydl_factory

    # -- public API ---------------------------------------------------------

    def extract(
        self,
        validation: ValidationResult,
        options: Optional[ExtractionOptions] = None,
    ) -> ExtractionResult:
        options = options or ExtractionOptions()

        if validation is None or not validation.valid:
            code = getattr(validation, "error_message", None) or "URL is not valid."
            logger.info("Extraction skipped: URL failed validation")
            return ExtractionResult(
                success=False,
                source_url=getattr(validation, "input_url", "") or "",
                normalized_url=getattr(validation, "normalized_url", None),
                content_type_hint=getattr(validation, "content_type_hint", None),
                extraction_mode=options.mode.value,
                failure_category=FailureCategory.URL_VALIDATION,
                failure_reason="URL failed Phase 2 validation.",
                error_detail=str(code),
            )

        # Pre-flight: validate authentication configuration before we create a
        # temp directory or touch the network. A misconfigured cookie file is a
        # config fault, reported with its own category - never AUTH_REQUIRED.
        config_error = self._auth_config_error(options)
        if config_error is not None:
            logger.info(
                "Extraction skipped: %s", config_error.category
            )
            return ExtractionResult(
                success=False,
                source_url=validation.input_url,
                normalized_url=validation.normalized_url,
                content_type_hint=validation.content_type_hint,
                extraction_mode=options.mode.value,
                cookie_file_configured=options.cookie_file_configured,
                failure_category=config_error.category,
                failure_reason=config_error.message,
                error_detail=config_error.message,
            )

        url = validation.normalized_url or validation.input_url
        run = TempRun(self.temp_dir, keep=options.keep_media).create()
        logger.info(
            "Extraction started backend=%s mode=%s run=%s",
            self.name,
            options.mode.value,
            run.run_id,
        )

        started = time.monotonic()
        try:
            info = self._run_ydl(url, run, options)
            result = self._build_result(validation, url, info, run, options)
            result.download_seconds = round(time.monotonic() - started, 3)
            result.run_id = run.run_id
            logger.info(
                "Extraction completed run=%s success=%s type=%s",
                run.run_id,
                result.success,
                result.media_type,
            )
            if not result.success and not options.keep_media:
                run.cleanup()
                logger.debug("Cleaned up temp run after unsuccessful extraction")
            return result
        except ExtractionError as exc:
            logger.warning(
                "Extraction failed run=%s category=%s", run.run_id, exc.category
            )
            if not options.keep_media:
                run.cleanup()
                logger.debug("Cleaned up partial downloads after failure")
            return ExtractionResult(
                success=False,
                source_url=validation.input_url,
                normalized_url=validation.normalized_url,
                content_type_hint=validation.content_type_hint,
                extraction_mode=options.mode.value,
                cookie_file_configured=options.cookie_file_configured,
                run_id=run.run_id,
                download_seconds=round(time.monotonic() - started, 3),
                failure_category=exc.category,
                failure_reason=exc.message,
                error_detail=exc.message,
            )
        except Exception as exc:  # noqa: BLE001 - last-resort net; classify below
            category = classify_error(str(exc))
            logger.warning(
                "Extraction raised run=%s category=%s", run.run_id, category
            )
            if not options.keep_media:
                run.cleanup()
            return ExtractionResult(
                success=False,
                source_url=validation.input_url,
                normalized_url=validation.normalized_url,
                content_type_hint=validation.content_type_hint,
                extraction_mode=options.mode.value,
                cookie_file_configured=options.cookie_file_configured,
                run_id=run.run_id,
                download_seconds=round(time.monotonic() - started, 3),
                failure_category=category,
                failure_reason=_safe_message(str(exc)),
                error_detail=_safe_message(str(exc)),
            )

    # -- internals ----------------------------------------------------------

    def _auth_config_error(
        self, options: ExtractionOptions
    ) -> Optional[ExtractionError]:
        """Validate auth configuration before any network work.

        Returns an :class:`ExtractionError` (with a cookie-file category) if
        ``cookie_file`` mode is selected but the file is unusable, otherwise
        ``None``. Only the file's path/stat is inspected - never its contents,
        so no cookie value can leak through the returned message.
        """
        if options.mode != ExtractionMode.COOKIE_FILE:
            return None

        if not options.cookie_file:
            return ExtractionError(
                FailureCategory.COOKIE_FILE_MISSING,
                "cookie_file mode selected but COOKIE_FILE is not set.",
            )

        path = Path(options.cookie_file)
        if not path.exists() or not path.is_file():
            # The path itself is safe to name; it is not a secret value.
            return ExtractionError(
                FailureCategory.COOKIE_FILE_MISSING,
                f"Configured cookie file was not found: {path}",
            )

        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            return ExtractionError(
                FailureCategory.COOKIE_FILE_UNREADABLE,
                f"Cookie file could not be read: {path} ({exc.strerror or 'error'}).",
            )

        if size <= 0:
            return ExtractionError(
                FailureCategory.COOKIE_FILE_INVALID,
                f"Cookie file is empty: {path}",
            )

        return None

    def _build_ydl_options(self, run: TempRun, options: ExtractionOptions) -> dict:
        """Construct the yt-dlp options dictionary.

        Media lands inside the per-run directory. We fetch metadata alongside
        the download and prefer a single reasonable representation rather than
        the highest-quality multi-format download - correctness over size in
        this phase.
        """
        ydl_opts: dict[str, Any] = {
            "outtmpl": str(run.path / "%(id)s.%(ext)s"),
            "paths": {"home": str(run.path)},
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": options.timeout,
            "retries": 2,
            "ignoreerrors": False,
            "writethumbnail": False,
            "writeinfojson": False,
            "consoletitle": False,
            # Prefer a single already-muxed file where possible; fall back to
            # best available. Avoids fetching multiple format copies.
            "format": "best/bestvideo*+bestaudio/best",
        }

        if options.mode == ExtractionMode.BROWSER_COOKIES and options.cookies_from_browser:
            # yt-dlp reads cookies straight from the local browser store. The
            # browser name is the only thing we pass; cookie values are never
            # read, stored, printed or logged by this project.
            ydl_opts["cookiesfrombrowser"] = (options.cookies_from_browser,)
            logger.info("Browser-cookie mode enabled (browser configured)")
        elif options.mode == ExtractionMode.COOKIE_FILE and options.cookie_file:
            # Only the path is passed. yt-dlp opens the file itself; we never
            # read, parse, copy or log its contents.
            ydl_opts["cookiefile"] = str(options.cookie_file)
            logger.info("Cookie-file mode enabled (cookie file configured)")

        return ydl_opts

    def _make_ydl(self, ydl_opts: dict) -> Any:
        if self._ydl_factory is not None:
            return self._ydl_factory(ydl_opts)
        try:
            import yt_dlp  # imported lazily so unit tests need not install it
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ExtractionError(
                FailureCategory.EXTRACTION,
                "yt-dlp is not installed (run: pip install -r requirements.txt).",
            ) from exc
        return yt_dlp.YoutubeDL(ydl_opts)

    def _run_ydl(self, url: str, run: TempRun, options: ExtractionOptions) -> dict:
        """Invoke yt-dlp and return its info dict."""
        ydl_opts = self._build_ydl_options(run, options)
        logger.info("Invoking yt-dlp for extraction (download=True)")
        ydl = self._make_ydl(ydl_opts)
        with ydl as session:
            info = session.extract_info(url, download=True)
        if not info or not isinstance(info, dict):
            raise ExtractionError(
                FailureCategory.METADATA_ERROR,
                "yt-dlp returned no usable metadata for this URL.",
            )
        return info

    def _build_result(
        self,
        validation: ValidationResult,
        url: str,
        info: dict,
        run: TempRun,
        options: ExtractionOptions,
    ) -> ExtractionResult:
        """Turn a yt-dlp info dict + downloaded files into an ExtractionResult."""
        entries = _entries_of(info)
        media_files, reported_downloads = self._collect_and_validate_media(info, run)

        caption = _extract_caption(info, entries)
        caption_extracted = bool(caption)
        hashtags = _extract_hashtags(caption) if caption else []

        media_type, type_detected = _detect_media_type(
            info, entries, media_files, validation.content_type_hint
        )

        metadata = _sanitise_metadata(info, entries)

        result = ExtractionResult(
            success=False,  # decided below
            source_url=validation.input_url,
            normalized_url=validation.normalized_url,
            content_type_hint=validation.content_type_hint,
            media_files=media_files,
            caption=caption,
            caption_extracted=caption_extracted,
            hashtags=hashtags,
            media_type=media_type,
            media_type_detected=type_detected,
            metadata=metadata,
            extraction_mode=options.mode.value,
            cookie_file_configured=options.cookie_file_configured,
        )

        if media_files:
            result.media_downloaded = True
            result.media_path = media_files[0].path
            # Media is the thing the next stage needs. Its presence defines
            # success; a missing caption is only a partial shortfall.
            result.success = True
            if not caption_extracted:
                result.failure_category = FailureCategory.CAPTION_UNAVAILABLE
                result.failure_reason = "Media retrieved but caption was unavailable."
        else:
            # Metadata may have come through, but with no usable media the run
            # cannot feed the multimodal pipeline, so it is not a success. If
            # yt-dlp reported downloads that then failed validation, that is a
            # DOWNLOAD_ERROR; if it never offered any media, MEDIA_UNAVAILABLE.
            result.media_downloaded = False
            result.success = False
            if reported_downloads:
                result.failure_category = FailureCategory.DOWNLOAD_ERROR
                result.failure_reason = (
                    "yt-dlp reported a download but the file was missing, "
                    "empty or unusable."
                )
            else:
                result.failure_category = FailureCategory.MEDIA_UNAVAILABLE
                result.failure_reason = (
                    "No downloadable media was retrieved for this URL."
                )

        return result

    def _collect_and_validate_media(
        self, info: dict, run: TempRun
    ) -> tuple[list[MediaFile], bool]:
        """Gather downloaded files and keep only those that pass validation.

        Validation: the file exists, is non-empty, readable, and carries a
        recognised media extension. Invalid files are never handed onward.

        Returns ``(valid_media, reported_downloads)`` where ``reported_downloads``
        is True if yt-dlp claimed to have written at least one file - used to
        tell DOWNLOAD_ERROR (claimed but bad) from MEDIA_UNAVAILABLE (nothing
        offered).
        """
        reported = _collect_filepaths(info)
        reported_downloads = bool(reported)
        candidate_paths = list(reported)
        if not candidate_paths:
            # yt-dlp did not report paths; fall back to whatever landed in the
            # isolated run directory.
            candidate_paths = [str(p) for p in run.files()]

        seen: set[str] = set()
        media_files: list[MediaFile] = []
        for raw_path in candidate_paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)

            if not path.exists() or not path.is_file():
                logger.debug("Skipping missing download candidate: %s", path.name)
                continue
            try:
                size = path.stat().st_size
            except OSError:
                logger.debug("Cannot stat download candidate: %s", path.name)
                continue
            if size <= 0:
                logger.debug("Skipping empty download: %s", path.name)
                continue

            ext = path.suffix.lower().lstrip(".")
            kind = _kind_for_ext(ext)
            if kind is None:
                logger.debug("Skipping unrecognised extension: %s", path.name)
                continue
            if not _readable(path):
                logger.debug("Skipping unreadable file: %s", path.name)
                continue

            media_files.append(
                MediaFile(path=str(path), kind=kind, size_bytes=size, ext=ext)
            )

        return media_files, reported_downloads


# ---------------------------------------------------------------------------
# Helpers (module-level, pure)
# ---------------------------------------------------------------------------


def _safe_message(message: str, limit: int = 500) -> str:
    """Trim a raw error message. Never carries secrets: yt-dlp error strings
    do not include cookie values, but we truncate defensively anyway."""
    text = _ANSI_RE.sub("", message or "").strip().replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _entries_of(info: dict) -> list[dict]:
    """Return playlist entries as a list (empty for a single item)."""
    entries = info.get("entries")
    if not entries:
        return []
    return [e for e in entries if isinstance(e, dict)]


def _kind_for_ext(ext: str) -> Optional[str]:
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    return None


def _readable(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _collect_filepaths(info: dict) -> list[str]:
    """Pull every downloaded filepath yt-dlp recorded, across playlist entries."""
    paths: list[str] = []

    def visit(node: dict) -> None:
        for download in node.get("requested_downloads") or []:
            if isinstance(download, dict):
                fp = download.get("filepath") or download.get("_filename")
                if fp:
                    paths.append(fp)
        # Some yt-dlp versions expose the final name at the top level.
        top = node.get("filepath") or node.get("_filename")
        if top:
            paths.append(top)
        for entry in _entries_of(node):
            visit(entry)

    visit(info)
    return paths


def _extract_caption(info: dict, entries: list[dict]) -> Optional[str]:
    """Instagram captions surface as yt-dlp's ``description``/``title``."""
    for source in (info, *entries):
        for key in ("description", "title"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_hashtags(caption: str) -> list[str]:
    """Return hashtags in first-seen order, de-duplicated case-insensitively."""
    seen: set[str] = set()
    tags: list[str] = []
    for match in _HASHTAG_RE.findall(caption or ""):
        lower = match.lower()
        if lower not in seen:
            seen.add(lower)
            tags.append("#" + match)
    return tags


def _detect_media_type(
    info: dict,
    entries: list[dict],
    media_files: list[MediaFile],
    hint: Optional[str],
) -> tuple[str, bool]:
    """Determine the media type from actual extraction data.

    Returns ``(media_type, detected)``. When the data cannot establish a type,
    returns ``("unknown", False)`` rather than guessing. The URL hint only
    refines a *video* into reel vs plain video; it never manufactures a type
    on its own.
    """
    # A real playlist with more than one item is a carousel.
    if len(entries) > 1 or (info.get("playlist_count") or 0) > 1:
        return MediaType.CAROUSEL.value, True
    if len(media_files) > 1:
        return MediaType.CAROUSEL.value, True

    if len(media_files) == 1:
        only = media_files[0]
        if only.kind == "video":
            if hint == ContentTypeHint.REEL:
                return MediaType.REEL.value, True
            if hint == ContentTypeHint.VIDEO:
                return MediaType.VIDEO.value, True
            return MediaType.VIDEO.value, True
        if only.kind == "image":
            return MediaType.IMAGE.value, True

    # No media, or an unrecognised kind: do not invent a type.
    return MediaType.UNKNOWN.value, False


def _sanitise_metadata(info: dict, entries: list[dict]) -> dict[str, Any]:
    """Keep only whitelisted, non-sensitive metadata fields.

    Formats blobs, http_headers, cookies, raw signed URLs and thumbnails are
    intentionally dropped so nothing sensitive is retained, logged or printed.
    """
    meta: dict[str, Any] = {}
    for key in _METADATA_WHITELIST:
        if key in info and info[key] is not None:
            meta[key] = info[key]
    if entries:
        meta["entry_count"] = len(entries)
    return meta
