"""Instagram URL validation.

Phase 2 of the feasibility project. This module answers five questions about
a user-supplied string, using nothing but the string itself:

1. Is it a syntactically valid URL?
2. Does it belong to Instagram?
3. Is it HTTPS?
4. Does its path look like supported Instagram content?
5. What content type, if any, can be inferred from the URL?

The module is strictly offline: it never opens a socket, resolves a name,
follows a redirect or evaluates anything taken from the URL. It only parses
text.

A content type hint is a *hint*. ``/reel/<code>/`` reliably means a reel, but
``/p/<code>/`` may be a single image, a video or a carousel - the URL cannot
tell them apart, so those are reported as ``post_or_unknown`` and resolved
later during extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The only scheme the project accepts.
REQUIRED_SCHEME: Final = "https"

#: Hosts recognised as Instagram. Membership is exact, so lookalike domains
#: such as "instagram.com.example.net" or "notinstagram.com" do not match.
ALLOWED_HOSTS: Final[frozenset] = frozenset(
    {
        "instagram.com",
        "www.instagram.com",
        "m.instagram.com",
        "instagr.am",
        "www.instagr.am",
    }
)

#: Every accepted URL is normalized onto this host.
CANONICAL_HOST: Final = "www.instagram.com"

#: Shortcodes are base64-ish identifiers; Instagram uses A-Z a-z 0-9 _ -.
SHORTCODE_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Usernames may contain letters, digits, periods and underscores.
USERNAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9._]{1,30}$")


class ContentTypeHint:
    """Content type hints inferable from a URL path.

    ``POST_OR_UNKNOWN`` is deliberate: a ``/p/`` URL may hold a single image,
    a video or a carousel, and the URL alone cannot distinguish them.
    """

    REEL: Final = "reel"
    VIDEO: Final = "video"
    POST_OR_UNKNOWN: Final = "post_or_unknown"

    ALL: Final = (REEL, VIDEO, POST_OR_UNKNOWN)


class ErrorCode:
    """Machine-readable validation failure codes."""

    EMPTY_INPUT: Final = "EMPTY_INPUT"
    INVALID_URL: Final = "INVALID_URL"
    INVALID_SCHEME: Final = "INVALID_SCHEME"
    NON_INSTAGRAM_DOMAIN: Final = "NON_INSTAGRAM_DOMAIN"
    UNSUPPORTED_PATH: Final = "UNSUPPORTED_PATH"

    ALL: Final = (
        EMPTY_INPUT,
        INVALID_URL,
        INVALID_SCHEME,
        NON_INSTAGRAM_DOMAIN,
        UNSUPPORTED_PATH,
    )


#: Path prefixes that identify content, mapped to their hint.
_CONTENT_PREFIXES: Final[dict] = {
    "reel": ContentTypeHint.REEL,
    "reels": ContentTypeHint.REEL,
    "p": ContentTypeHint.POST_OR_UNKNOWN,
    "tv": ContentTypeHint.VIDEO,
}

#: Prefixes normalized to a single canonical spelling.
_PREFIX_ALIASES: Final[dict] = {"reels": "reel"}

#: Instagram paths that are understood but out of scope for this project.
_KNOWN_UNSUPPORTED: Final[dict] = {
    "stories": "Stories are not a supported content type for this project.",
    "share": (
        "Instagram share links resolve to real content only by following a "
        "redirect, which this offline validator will not do. Open the link "
        "in a browser and use the resulting /p/, /reel/ or /tv/ URL."
    ),
    "explore": "Explore pages are not individual content items.",
    "accounts": "Account pages are not content items.",
    "directory": "Directory pages are not content items.",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one URL string.

    On success ``normalized_url`` and ``content_type_hint`` are set and both
    error fields are ``None``. On failure the reverse holds.
    """

    valid: bool
    input_url: str = ""
    normalized_url: str | None = None
    content_type_hint: str | None = None
    shortcode: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __bool__(self) -> bool:
        return self.valid

    def as_dict(self) -> dict:
        """Return a plain dictionary, handy for later JSON output."""
        return {
            "valid": self.valid,
            "input_url": self.input_url,
            "normalized_url": self.normalized_url,
            "content_type_hint": self.content_type_hint,
            "shortcode": self.shortcode,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _failure(input_url: str, code: str, message: str) -> ValidationResult:
    return ValidationResult(
        valid=False,
        input_url=input_url,
        error_code=code,
        error_message=message,
    )


def _success(
    input_url: str, normalized_url: str, hint: str, shortcode: str
) -> ValidationResult:
    return ValidationResult(
        valid=True,
        input_url=input_url,
        normalized_url=normalized_url,
        content_type_hint=hint,
        shortcode=shortcode,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _looks_like_bare_instagram_host(candidate: str) -> bool:
    """True when a scheme-less string starts with a known Instagram host."""
    head = candidate.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return head.lower() in ALLOWED_HOSTS


def _split_host_and_port(netloc: str):
    """Return ``(host, port, error)`` for a netloc, without raising."""
    if "@" in netloc:
        return "", None, "URL contains embedded credentials."

    try:
        parsed = urlsplit("//" + netloc)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:  # malformed IPv6 literal, bad port, ...
        return "", None, "URL host could not be parsed ({}).".format(exc)

    return host.lower(), (str(port) if port is not None else None), None


def validate_instagram_url(raw_url) -> ValidationResult:
    """Validate a candidate Instagram URL.

    Never raises for ordinary bad input; failures come back as a
    :class:`ValidationResult` with ``valid=False`` and an error code.
    """
    if raw_url is None:
        return _failure("", ErrorCode.EMPTY_INPUT, "No URL was provided.")

    if not isinstance(raw_url, str):
        return _failure(
            str(raw_url),
            ErrorCode.INVALID_URL,
            "Expected a string, got {}.".format(type(raw_url).__name__),
        )

    original = raw_url
    # Strip surrounding whitespace and the zero-width space that chat clients
    # sometimes paste alongside a link.
    candidate = raw_url.strip().strip("​").strip()

    if not candidate:
        return _failure(original, ErrorCode.EMPTY_INPUT, "No URL was provided.")

    if any(char.isspace() for char in candidate):
        return _failure(original, ErrorCode.INVALID_URL, "URL contains whitespace.")

    # A scheme-less paste is accepted only when it clearly names an Instagram
    # host; anything else stays an invalid URL rather than being guessed at.
    if "://" not in candidate:
        if _looks_like_bare_instagram_host(candidate):
            candidate = REQUIRED_SCHEME + "://" + candidate
        else:
            return _failure(
                original,
                ErrorCode.INVALID_URL,
                "Not a URL: no scheme (expected 'https://...').",
            )

    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        return _failure(
            original, ErrorCode.INVALID_URL, "Malformed URL ({}).".format(exc)
        )

    scheme = parts.scheme.lower()
    if not parts.netloc:
        return _failure(original, ErrorCode.INVALID_URL, "URL has no host.")

    host, port, host_error = _split_host_and_port(parts.netloc)
    if host_error:
        return _failure(original, ErrorCode.INVALID_URL, host_error)
    if not host:
        return _failure(original, ErrorCode.INVALID_URL, "URL has no host.")

    # Domain before scheme: a non-Instagram domain is the more useful
    # complaint, whether it arrived over http or https.
    if host not in ALLOWED_HOSTS:
        return _failure(
            original,
            ErrorCode.NON_INSTAGRAM_DOMAIN,
            "Host '{}' is not an Instagram domain.".format(host),
        )

    if scheme != REQUIRED_SCHEME:
        return _failure(
            original,
            ErrorCode.INVALID_SCHEME,
            "Scheme '{}' is not supported; use https.".format(scheme or "(none)"),
        )

    if port not in (None, "443"):
        return _failure(
            original,
            ErrorCode.INVALID_URL,
            "Unexpected port ':{}' on an Instagram URL.".format(port),
        )

    return _validate_path(original, parts.path)


def _validate_path(original: str, path: str) -> ValidationResult:
    """Classify the path portion of an already-accepted Instagram URL."""
    segments = [segment for segment in path.split("/") if segment]

    if not segments:
        return _failure(
            original,
            ErrorCode.UNSUPPORTED_PATH,
            "URL points at the Instagram home page, not at a post.",
        )

    first = segments[0].lower()

    # Profile-scoped form: /<username>/reel/<shortcode>/
    if (
        first not in _CONTENT_PREFIXES
        and first not in _KNOWN_UNSUPPORTED
        and len(segments) >= 2
        and segments[1].lower() in _CONTENT_PREFIXES
        and USERNAME_PATTERN.match(segments[0])
    ):
        segments = segments[1:]
        first = segments[0].lower()

    if first in _KNOWN_UNSUPPORTED:
        return _failure(original, ErrorCode.UNSUPPORTED_PATH, _KNOWN_UNSUPPORTED[first])

    if first not in _CONTENT_PREFIXES:
        return _failure(
            original,
            ErrorCode.UNSUPPORTED_PATH,
            (
                "Path '/{}/' is not a supported content path. "
                "Expected /reel/, /p/ or /tv/.".format(segments[0])
            ),
        )

    if len(segments) < 2:
        return _failure(
            original,
            ErrorCode.UNSUPPORTED_PATH,
            "Path '/{}/' is missing a content identifier.".format(first),
        )

    shortcode = segments[1]
    if not SHORTCODE_PATTERN.match(shortcode):
        return _failure(
            original,
            ErrorCode.UNSUPPORTED_PATH,
            "'{}' is not a valid Instagram shortcode.".format(shortcode),
        )

    # Trailing sub-pages such as /p/<code>/liked_by/ are not content URLs.
    if len(segments) > 2:
        extra = "/".join(segments[2:])
        return _failure(
            original,
            ErrorCode.UNSUPPORTED_PATH,
            "Unexpected trailing path segment '/{}/'.".format(extra),
        )

    prefix = _PREFIX_ALIASES.get(first, first)
    normalized = urlunsplit(
        (REQUIRED_SCHEME, CANONICAL_HOST, "/{}/{}/".format(prefix, shortcode), "", "")
    )
    return _success(original, normalized, _CONTENT_PREFIXES[first], shortcode)
