"""Tests for the Phase 2 Instagram URL validator.

Covers accepted patterns, rejected patterns, normalization, content type
hints, error code determinism, and the guarantee that validation is purely
offline.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extractor.url_validator import (  # noqa: E402
    ContentTypeHint,
    ErrorCode,
    ValidationResult,
    validate_instagram_url,
)

REEL_URL = "https://www.instagram.com/reel/Cx1_ab-Z/"
POST_URL = "https://www.instagram.com/p/Cx1_ab-Z/"
TV_URL = "https://www.instagram.com/tv/Cx1_ab-Z/"


# ---------------------------------------------------------------------------
# Valid URLs
# ---------------------------------------------------------------------------


def test_valid_reel_url():
    result = validate_instagram_url(REEL_URL)
    assert result.valid
    assert result.error_code is None
    assert result.error_message is None
    assert result.normalized_url == REEL_URL
    assert result.content_type_hint == ContentTypeHint.REEL
    assert result.shortcode == "Cx1_ab-Z"


def test_valid_post_url_is_only_a_hint():
    result = validate_instagram_url(POST_URL)
    assert result.valid
    assert result.content_type_hint == ContentTypeHint.POST_OR_UNKNOWN
    assert result.normalized_url == POST_URL


def test_valid_tv_url():
    result = validate_instagram_url(TV_URL)
    assert result.valid
    assert result.content_type_hint == ContentTypeHint.VIDEO


def test_missing_trailing_slash_is_added():
    result = validate_instagram_url("https://www.instagram.com/reel/Cx1_ab-Z")
    assert result.valid
    assert result.normalized_url == REEL_URL


def test_query_parameters_are_stripped():
    result = validate_instagram_url(
        "https://www.instagram.com/reel/Cx1_ab-Z/?igsh=MzRlODBiNWFlZA%3D%3D&utm_source=x"
    )
    assert result.valid
    assert result.normalized_url == REEL_URL


def test_fragment_is_stripped():
    result = validate_instagram_url("https://www.instagram.com/p/Cx1_ab-Z/#comments")
    assert result.valid
    assert result.normalized_url == POST_URL


def test_host_case_is_normalized():
    result = validate_instagram_url("https://WWW.Instagram.COM/reel/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == REEL_URL


def test_shortcode_case_is_preserved():
    """Shortcodes are case sensitive and must survive normalization intact."""
    result = validate_instagram_url("https://www.instagram.com/p/AbCdEf/")
    assert result.valid
    assert result.shortcode == "AbCdEf"
    assert result.normalized_url == "https://www.instagram.com/p/AbCdEf/"


def test_bare_host_without_www_is_normalized():
    result = validate_instagram_url("https://instagram.com/reel/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == REEL_URL


def test_mobile_host_is_accepted():
    result = validate_instagram_url("https://m.instagram.com/p/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == POST_URL


def test_reels_plural_is_normalized_to_reel():
    result = validate_instagram_url("https://www.instagram.com/reels/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == REEL_URL
    assert result.content_type_hint == ContentTypeHint.REEL


def test_profile_scoped_reel_url_is_accepted():
    result = validate_instagram_url("https://www.instagram.com/nasa/reel/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == REEL_URL
    assert result.content_type_hint == ContentTypeHint.REEL


def test_surrounding_whitespace_is_tolerated():
    result = validate_instagram_url("  " + REEL_URL + "\n")
    assert result.valid
    assert result.normalized_url == REEL_URL


def test_scheme_less_instagram_paste_is_accepted():
    result = validate_instagram_url("www.instagram.com/reel/Cx1_ab-Z/")
    assert result.valid
    assert result.normalized_url == REEL_URL


# ---------------------------------------------------------------------------
# Invalid URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", "\t\n", None])
def test_empty_input_is_rejected(value):
    result = validate_instagram_url(value)
    assert not result.valid
    assert result.error_code == ErrorCode.EMPTY_INPUT
    assert result.normalized_url is None
    assert result.content_type_hint is None


@pytest.mark.parametrize(
    "value",
    [
        "not-a-url",
        "hello world",
        "12345abc",
        "://missing-scheme",
        "https://",
        "instagram",
    ],
)
def test_non_url_strings_are_rejected(value):
    result = validate_instagram_url(value)
    assert not result.valid
    assert result.error_code == ErrorCode.INVALID_URL


def test_non_instagram_domain_is_rejected():
    result = validate_instagram_url("https://youtube.com/watch?v=test")
    assert not result.valid
    assert result.error_code == ErrorCode.NON_INSTAGRAM_DOMAIN


@pytest.mark.parametrize(
    "value",
    [
        "https://www.instagram.com.evil.example/p/Cx1_ab-Z/",
        "https://notinstagram.com/p/Cx1_ab-Z/",
        "https://instagram.com.co/p/Cx1_ab-Z/",
        "https://evil.example/www.instagram.com/p/Cx1_ab-Z/",
    ],
)
def test_lookalike_domains_are_rejected(value):
    result = validate_instagram_url(value)
    assert not result.valid
    assert result.error_code == ErrorCode.NON_INSTAGRAM_DOMAIN


def test_credentials_in_url_are_rejected():
    """https://www.instagram.com@evil.example/ style spoofing must not pass."""
    result = validate_instagram_url("https://www.instagram.com@evil.example/p/Ab/")
    assert not result.valid
    assert result.error_code in (ErrorCode.INVALID_URL, ErrorCode.NON_INSTAGRAM_DOMAIN)


def test_http_is_rejected_because_project_requires_https():
    result = validate_instagram_url("http://www.instagram.com/reel/Cx1_ab-Z/")
    assert not result.valid
    assert result.error_code == ErrorCode.INVALID_SCHEME


@pytest.mark.parametrize("scheme", ["ftp", "file", "javascript"])
def test_other_schemes_are_rejected(scheme):
    result = validate_instagram_url(f"{scheme}://www.instagram.com/p/Cx1_ab-Z/")
    assert not result.valid
    assert result.error_code == ErrorCode.INVALID_SCHEME


def test_unexpected_port_is_rejected():
    result = validate_instagram_url("https://www.instagram.com:8443/p/Cx1_ab-Z/")
    assert not result.valid
    assert result.error_code == ErrorCode.INVALID_URL


@pytest.mark.parametrize(
    "value",
    [
        "https://www.instagram.com/",
        "https://www.instagram.com/nasa/",
        "https://www.instagram.com/stories/nasa/123456/",
        "https://www.instagram.com/explore/tags/space/",
        "https://www.instagram.com/accounts/login/",
        "https://www.instagram.com/p/",
        "https://www.instagram.com/reel/",
        "https://www.instagram.com/p/Cx1_ab-Z/liked_by/",
    ],
)
def test_unsupported_instagram_paths_are_rejected(value):
    result = validate_instagram_url(value)
    assert not result.valid
    assert result.error_code == ErrorCode.UNSUPPORTED_PATH


def test_share_links_are_rejected_with_guidance():
    result = validate_instagram_url("https://www.instagram.com/share/reel/AbCdEf/")
    assert not result.valid
    assert result.error_code == ErrorCode.UNSUPPORTED_PATH
    assert "redirect" in result.error_message


def test_invalid_shortcode_characters_are_rejected():
    result = validate_instagram_url("https://www.instagram.com/p/A%20b$/")
    assert not result.valid
    assert result.error_code == ErrorCode.UNSUPPORTED_PATH


# ---------------------------------------------------------------------------
# Contract guarantees
# ---------------------------------------------------------------------------


def test_error_codes_are_from_the_declared_set():
    for value in ["", "not-a-url", "https://youtube.com/", REEL_URL.replace("https", "http")]:
        result = validate_instagram_url(value)
        assert result.error_code in ErrorCode.ALL


def test_error_codes_are_deterministic():
    for _ in range(5):
        assert validate_instagram_url("https://vimeo.com/12345").error_code == (
            ErrorCode.NON_INSTAGRAM_DOMAIN
        )


def test_result_is_immutable_and_truthy_by_validity():
    ok = validate_instagram_url(REEL_URL)
    bad = validate_instagram_url("not-a-url")
    assert bool(ok) is True
    assert bool(bad) is False
    with pytest.raises(Exception):
        ok.valid = False  # frozen dataclass


def test_result_as_dict_round_trips_all_fields():
    payload = validate_instagram_url(REEL_URL).as_dict()
    assert set(payload) == {
        "valid",
        "input_url",
        "normalized_url",
        "content_type_hint",
        "shortcode",
        "error_code",
        "error_message",
    }
    assert payload["content_type_hint"] == ContentTypeHint.REEL


def test_failure_never_carries_a_normalized_url():
    for value in ["", "not-a-url", "https://youtube.com/x", "https://www.instagram.com/"]:
        result = validate_instagram_url(value)
        assert result.normalized_url is None
        assert result.content_type_hint is None
        assert result.error_message


@pytest.mark.parametrize(
    "value",
    [
        12345,
        3.14,
        b"https://www.instagram.com/p/Ab/",
        ["https://www.instagram.com/p/Ab/"],
        {"url": "x"},
        object(),
        "https://www.instagram.com/p/" + "A" * 500 + "/",
        "https://" + "a" * 5000 + ".com/p/Ab/",
        "https://www.instagram.com/p/\x00/",
        "https://www.instagram.com/p/ünïcødé/",
        "https://[::1]/p/Ab/",
        "https://192.168.0.1/p/Ab/",
        "//www.instagram.com/p/Ab/",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
    ],
)
def test_hostile_input_never_raises(value):
    result = validate_instagram_url(value)
    assert isinstance(result, ValidationResult)
    assert not result.valid
    assert result.error_code in ErrorCode.ALL


def test_validation_makes_no_network_requests(monkeypatch):
    """Any socket use during validation fails the test loudly."""

    def blow_up(*args, **kwargs):
        raise AssertionError("validator attempted a network call")

    monkeypatch.setattr(socket, "socket", blow_up)
    monkeypatch.setattr(socket, "create_connection", blow_up)
    monkeypatch.setattr(socket, "getaddrinfo", blow_up)
    monkeypatch.setattr(socket, "gethostbyname", blow_up)

    for value in [REEL_URL, POST_URL, TV_URL, "not-a-url", "https://youtube.com/x"]:
        validate_instagram_url(value)


def test_validator_module_imports_no_network_libraries():
    """Guard against a future edit reaching for requests/urllib.request."""
    source = (PROJECT_ROOT / "extractor" / "url_validator.py").read_text(
        encoding="utf-8"
    )
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    forbidden = ("requests", "urllib.request", "httpx", "socket", "http.client", "yt_dlp")
    for line in imports:
        for name in forbidden:
            assert name not in line, f"unexpected import in {line!r}"
