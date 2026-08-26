"""Offline unit tests for the Phase 3 extraction layer.

These tests never touch Instagram. A fake yt-dlp factory is injected into
``YtDlpExtractor``; it writes files into the real per-run temp directory and
returns an info dict, so the whole download/validate/classify path runs without
a network.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extractor import (  # noqa: E402
    ExtractionMode,
    ExtractionOptions,
    ExtractionResult,
    FailureCategory,
    MediaType,
    YtDlpExtractor,
    classify_error,
    validate_instagram_url,
)
from extractor.instagram_extractor import (  # noqa: E402
    _extract_hashtags,
    _safe_message,
    _sanitise_metadata,
)

REEL_URL = "https://www.instagram.com/reel/Cx1_ab-Z/"
POST_URL = "https://www.instagram.com/p/Cx1_ab-Z/"


# ---------------------------------------------------------------------------
# Fake yt-dlp
# ---------------------------------------------------------------------------


class FakeYDL:
    """Minimal yt-dlp double.

    On ``extract_info`` it either raises the configured exception or writes the
    configured files into the run's home directory and returns ``info`` with
    matching ``requested_downloads`` filepaths.
    """

    def __init__(self, opts, info, files, raise_exc):
        self.opts = opts
        self._info = info
        self._files = files
        self._raise = raise_exc
        home = (opts.get("paths") or {}).get("home")
        self.home = Path(home) if home else Path(".")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        if self._raise is not None:
            raise self._raise
        if self._info is None:
            return None
        written = []
        for name, content in self._files:
            path = self.home / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            written.append(str(path))
        info = dict(self._info)
        if written and "requested_downloads" not in info and "entries" not in info:
            info["requested_downloads"] = [{"filepath": p} for p in written]
        return info


def make_factory(info=None, files=(), raise_exc=None):
    """Return a ydl_factory that yields a configured FakeYDL."""
    info = info if info is not None else {"id": "Cx1_ab-Z", "description": ""}

    def factory(opts):
        return FakeYDL(opts, info, list(files), raise_exc)

    return factory


def valid(url=REEL_URL):
    result = validate_instagram_url(url)
    assert result.valid
    return result


def make_extractor(tmp_path, **factory_kwargs):
    return YtDlpExtractor(temp_dir=tmp_path, ydl_factory=make_factory(**factory_kwargs))


# ---------------------------------------------------------------------------
# Success and partial success
# ---------------------------------------------------------------------------


def test_successful_reel_extraction(tmp_path):
    info = {"id": "Cx1_ab-Z", "description": "A trip #travel #Sunset", "duration": 12}
    extractor = make_extractor(tmp_path, info=info, files=[("Cx1_ab-Z.mp4", b"\x00" * 2048)])
    result = extractor.extract(valid(REEL_URL))

    assert result.success is True
    assert result.media_downloaded is True
    assert result.media_path and result.media_path.endswith(".mp4")
    assert Path(result.media_path).exists()
    assert result.caption_extracted is True
    assert result.caption == "A trip #travel #Sunset"
    assert result.hashtags == ["#travel", "#Sunset"]
    assert result.media_type == MediaType.REEL.value
    assert result.media_type_detected is True
    assert result.failure_category is None
    assert result.download_seconds is not None


def test_post_with_single_video_is_video_not_guessed_from_url(tmp_path):
    info = {"id": "x", "description": "hi"}
    extractor = make_extractor(tmp_path, info=info, files=[("x.mp4", b"d" * 1024)])
    result = extractor.extract(valid(POST_URL))
    assert result.success is True
    # URL hint was post_or_unknown; actual file decides "video".
    assert result.media_type == MediaType.VIDEO.value
    assert result.media_type_detected is True


def test_single_image_post(tmp_path):
    info = {"id": "x", "description": "a photo #art"}
    extractor = make_extractor(tmp_path, info=info, files=[("x.jpg", b"img" * 512)])
    result = extractor.extract(valid(POST_URL))
    assert result.success is True
    assert result.media_type == MediaType.IMAGE.value
    assert result.hashtags == ["#art"]


def test_carousel_detected_from_multiple_entries(tmp_path):
    info = {
        "id": "car",
        "description": "swipe #carousel",
        "entries": [
            {"id": "a", "requested_downloads": [{"filepath": "PLACEHOLDER_A"}]},
            {"id": "b", "requested_downloads": [{"filepath": "PLACEHOLDER_B"}]},
        ],
    }
    # Files must exist; rewrite placeholders after write via a custom factory.
    def factory(opts):
        home = Path(opts["paths"]["home"])
        (home / "a.jpg").write_bytes(b"a" * 600)
        (home / "b.jpg").write_bytes(b"b" * 600)
        live_info = {
            "id": "car",
            "description": "swipe #carousel",
            "entries": [
                {"id": "a", "requested_downloads": [{"filepath": str(home / "a.jpg")}]},
                {"id": "b", "requested_downloads": [{"filepath": str(home / "b.jpg")}]},
            ],
        }
        return FakeYDL(opts, live_info, [], None)

    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=factory)
    result = extractor.extract(valid(POST_URL))
    assert result.success is True
    assert result.media_type == MediaType.CAROUSEL.value
    assert len(result.media_files) == 2
    assert result.hashtags == ["#carousel"]


def test_missing_caption_is_partial_success(tmp_path):
    info = {"id": "x", "description": "   "}  # blank caption
    extractor = make_extractor(tmp_path, info=info, files=[("x.mp4", b"v" * 1024)])
    result = extractor.extract(valid(REEL_URL))
    assert result.success is True
    assert result.media_downloaded is True
    assert result.caption_extracted is False
    assert result.caption is None
    assert result.failure_category == FailureCategory.CAPTION_UNAVAILABLE


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_missing_media_is_not_success(tmp_path):
    """Caption/metadata present but no media => not a success."""
    info = {"id": "x", "description": "caption only"}
    extractor = make_extractor(tmp_path, info=info, files=[])  # nothing written
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.media_downloaded is False
    assert result.failure_category == FailureCategory.MEDIA_UNAVAILABLE


def test_empty_downloaded_file_is_rejected(tmp_path):
    info = {"id": "x", "description": "hi"}
    extractor = make_extractor(tmp_path, info=info, files=[("x.mp4", b"")])  # empty
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.media_downloaded is False
    # yt-dlp reported a download that failed validation.
    assert result.failure_category == FailureCategory.DOWNLOAD_ERROR


def test_missing_downloaded_file_reported_by_ydl(tmp_path):
    """yt-dlp claims a filepath that does not exist on disk."""
    info = {
        "id": "x",
        "description": "hi",
        "requested_downloads": [{"filepath": str(tmp_path / "nope" / "ghost.mp4")}],
    }
    extractor = YtDlpExtractor(
        temp_dir=tmp_path, ydl_factory=make_factory(info=info, files=[])
    )
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.failure_category == FailureCategory.DOWNLOAD_ERROR


def test_unsupported_extension_yields_no_media(tmp_path):
    info = {"id": "x", "description": "hi"}
    extractor = make_extractor(tmp_path, info=info, files=[("x.txt", b"not media")])
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.media_type == MediaType.UNKNOWN.value
    assert result.media_type_detected is False


def test_ydl_returns_no_info_is_metadata_error(tmp_path):
    extractor = YtDlpExtractor(
        temp_dir=tmp_path, ydl_factory=make_factory(info=None, files=[])
    )
    # Force info to be None by making the fake return None.
    def factory(opts):
        return FakeYDL(opts, None, [], None)

    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=factory)
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.failure_category == FailureCategory.METADATA_ERROR


def test_invalid_validation_yields_url_validation_failure(tmp_path):
    bad = validate_instagram_url("https://youtube.com/watch?v=x")
    assert not bad.valid
    extractor = make_extractor(tmp_path)
    result = extractor.extract(bad)
    assert result.success is False
    assert result.failure_category == FailureCategory.URL_VALIDATION


@pytest.mark.parametrize(
    "message, expected",
    [
        ("ERROR: Requested content is not available, login required", FailureCategory.AUTH_REQUIRED),
        ("Please log in to view this account", FailureCategory.AUTH_REQUIRED),
        ("This account is private", FailureCategory.AUTH_REQUIRED),
        ("HTTP Error 429: Too Many Requests", FailureCategory.RATE_LIMITED),
        ("rate-limit reached", FailureCategory.RATE_LIMITED),
        ("Unable to download webpage: <urlopen error timed out>", FailureCategory.NETWORK_ERROR),
        ("getaddrinfo failed", FailureCategory.NETWORK_ERROR),
        ("The post is unavailable", FailureCategory.MEDIA_UNAVAILABLE),
        ("HTTP Error 404: Not Found", FailureCategory.MEDIA_UNAVAILABLE),
        ("Unsupported URL: https://example.com", FailureCategory.UNSUPPORTED_MEDIA),
        (
            "[Instagram] X: Instagram sent an empty media response. Check if "
            "this post is accessible ... use --cookies-from-browser or --cookies "
            "for the authentication.",
            FailureCategory.AUTH_REQUIRED,
        ),
        ("some totally novel failure", FailureCategory.EXTRACTION),
    ],
)
def test_error_classification(message, expected):
    assert classify_error(message) == expected


def test_ambiguous_ig_message_routes_to_auth(tmp_path):
    """IG's combined 'rate-limit reached or login required' -> AUTH_REQUIRED."""
    assert classify_error("rate-limit reached or login required") == (
        FailureCategory.AUTH_REQUIRED
    )


def test_auth_error_raised_by_ydl_is_classified(tmp_path):
    extractor = make_extractor(
        tmp_path, raise_exc=Exception("ERROR: login required to view")
    )
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    assert result.failure_category == FailureCategory.AUTH_REQUIRED


def test_network_error_raised_by_ydl_is_classified(tmp_path):
    extractor = make_extractor(
        tmp_path, raise_exc=Exception("Unable to download webpage: timed out")
    )
    result = extractor.extract(valid(REEL_URL))
    assert result.failure_category == FailureCategory.NETWORK_ERROR
    assert result.error_detail


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_failure_cleans_up_temp_run(tmp_path):
    extractor = make_extractor(tmp_path, raise_exc=Exception("boom network timeout"))
    result = extractor.extract(valid(REEL_URL))
    assert result.success is False
    # No leftover run directories.
    assert list(tmp_path.iterdir()) == []


def test_missing_media_cleans_up_temp_run(tmp_path):
    extractor = make_extractor(tmp_path, files=[])
    extractor.extract(valid(REEL_URL))
    assert list(tmp_path.iterdir()) == []


def test_success_keeps_media_until_caller_cleans(tmp_path):
    extractor = make_extractor(
        tmp_path, info={"id": "x", "description": "hi"}, files=[("x.mp4", b"v" * 1024)]
    )
    result = extractor.extract(valid(REEL_URL))
    assert result.success is True
    assert Path(result.media_path).exists()
    assert len(list(tmp_path.iterdir())) == 1  # run dir retained


def test_keep_media_retains_dir_even_on_failure(tmp_path):
    extractor = make_extractor(tmp_path, raise_exc=Exception("boom"))
    result = extractor.extract(valid(REEL_URL), ExtractionOptions(keep_media=True))
    assert result.success is False
    assert len(list(tmp_path.iterdir())) == 1


def test_runs_are_isolated_from_each_other(tmp_path):
    extractor = make_extractor(
        tmp_path, info={"id": "x", "description": "hi"}, files=[("x.mp4", b"v" * 1024)]
    )
    r1 = extractor.extract(valid(REEL_URL))
    r2 = extractor.extract(valid(REEL_URL))
    assert r1.run_id != r2.run_id
    assert Path(r1.media_path).parent != Path(r2.media_path).parent


# ---------------------------------------------------------------------------
# Serialisation, metadata, secrets
# ---------------------------------------------------------------------------


def test_result_serialises_to_dict(tmp_path):
    extractor = make_extractor(
        tmp_path, info={"id": "x", "description": "hi #a"}, files=[("x.mp4", b"v" * 1024)]
    )
    payload = extractor.extract(valid(REEL_URL)).as_dict()
    import json

    text = json.dumps(payload)  # must be JSON-serialisable
    assert '"success": true' in text
    assert set(payload) >= {
        "success",
        "source_url",
        "media_downloaded",
        "media_path",
        "caption_extracted",
        "caption",
        "hashtags",
        "media_type_detected",
        "media_type",
        "metadata",
        "failure_category",
        "failure_reason",
    }


def test_metadata_is_sanitised_of_sensitive_fields():
    raw = {
        "id": "abc",
        "title": "t",
        "duration": 5,
        "http_headers": {"Cookie": "sessionid=SECRET"},
        "cookies": "sessionid=SECRET",
        "formats": [{"url": "https://cdn/signed?token=SECRET"}],
        "url": "https://cdn/signed?token=SECRET",
        "thumbnails": [{"url": "https://x"}],
    }
    meta = _sanitise_metadata(raw, [])
    assert meta == {"id": "abc", "title": "t", "duration": 5}
    assert "http_headers" not in meta
    assert "cookies" not in meta
    assert "formats" not in meta
    assert "url" not in meta


def test_secrets_never_appear_in_result_or_logs(tmp_path, caplog):
    """A cookie value hidden in yt-dlp metadata must not reach result or logs."""
    secret = "sessionid=TOPSECRET123"
    info = {
        "id": "x",
        "description": "hi",
        "http_headers": {"Cookie": secret},
        "cookies": secret,
    }
    extractor = make_extractor(tmp_path, info=info, files=[("x.mp4", b"v" * 1024)])
    with caplog.at_level(logging.DEBUG):
        result = extractor.extract(valid(REEL_URL))

    blob = str(result.as_dict())
    assert secret not in blob
    assert "TOPSECRET123" not in blob
    assert secret not in caplog.text
    assert "TOPSECRET123" not in caplog.text


def test_safe_message_strips_ansi_and_truncates():
    coloured = "\x1b[0;31mERROR:\x1b[0m something failed"
    cleaned = _safe_message(coloured)
    assert "\x1b" not in cleaned
    assert cleaned.startswith("ERROR: something failed")
    assert _safe_message("x" * 900).endswith("...")


def test_hashtag_parsing_dedupes_case_insensitively():
    tags = _extract_hashtags("look #Travel #travel #SUNSET end #travel")
    assert tags == ["#Travel", "#SUNSET"]


def test_browser_cookie_mode_passes_only_browser_name(tmp_path):
    """browser_cookies mode sets cookiesfrombrowser to the browser name only."""
    captured = {}

    def factory(opts):
        captured.update(opts)
        home = Path(opts["paths"]["home"])
        (home / "x.mp4").write_bytes(b"v" * 1024)
        return FakeYDL(
            opts,
            {"id": "x", "description": "hi", "requested_downloads": [{"filepath": str(home / "x.mp4")}]},
            [],
            None,
        )

    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=factory)
    options = ExtractionOptions(
        mode=ExtractionMode.BROWSER_COOKIES, cookies_from_browser="firefox"
    )
    result = extractor.extract(valid(REEL_URL), options)
    assert result.success is True
    assert captured.get("cookiesfrombrowser") == ("firefox",)


# ---------------------------------------------------------------------------
# cookie_file mode (Phase 4)
# ---------------------------------------------------------------------------

# A tiny synthetic Netscape cookie file with entirely fake values. This is the
# ONLY cookie data used in tests; no real credentials are involved.
FAKE_SESSION_VALUE = "FAKE_SESSION_VALUE"
NETSCAPE_FIXTURE = (
    "# Netscape HTTP Cookie File\n"
    ".instagram.com\tTRUE\t/\tTRUE\t9999999999\tsessionid\t" + FAKE_SESSION_VALUE + "\n"
    ".instagram.com\tTRUE\t/\tTRUE\t9999999999\tds_user_id\tFAKE_USER_ID\n"
)


def write_cookie_fixture(path: Path) -> Path:
    path.write_text(NETSCAPE_FIXTURE, encoding="utf-8")
    return path


def _capturing_factory(captured: dict, files=(("x.mp4", b"v" * 1024),)):
    """A ydl_factory that records the options it was built with."""

    def factory(opts):
        captured.clear()
        captured.update(opts)
        home = Path(opts["paths"]["home"])
        written = []
        for name, content in files:
            (home / name).write_bytes(content)
            written.append(str(home / name))
        info = {
            "id": "x",
            "description": "hi #tag",
            "requested_downloads": [{"filepath": p} for p in written],
        }
        return FakeYDL(opts, info, [], None)

    return factory


def test_public_mode_sets_no_cookie_options(tmp_path):
    captured: dict = {}
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=_capturing_factory(captured))
    result = extractor.extract(valid(REEL_URL), ExtractionOptions(mode=ExtractionMode.PUBLIC))
    assert result.success is True
    assert "cookiefile" not in captured
    assert "cookiesfrombrowser" not in captured
    assert result.cookie_file_configured is False


def test_browser_cookie_mode_still_configures(tmp_path):
    captured: dict = {}
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=_capturing_factory(captured))
    options = ExtractionOptions(
        mode=ExtractionMode.BROWSER_COOKIES, cookies_from_browser="firefox"
    )
    result = extractor.extract(valid(REEL_URL), options)
    assert result.success is True
    assert captured.get("cookiesfrombrowser") == ("firefox",)
    assert "cookiefile" not in captured


def test_cookie_file_mode_accepts_valid_fixture_and_passes_path(tmp_path):
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    captured: dict = {}
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=_capturing_factory(captured))
    options = ExtractionOptions(
        mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie)
    )
    result = extractor.extract(valid(REEL_URL), options)
    assert result.success is True
    # yt-dlp received the PATH, not the contents.
    assert captured.get("cookiefile") == str(cookie)
    assert result.cookie_file_configured is True


def test_cookie_file_missing_is_reported_without_network(tmp_path):
    missing = tmp_path / "nope" / "instagram_cookies.txt"
    extractor = YtDlpExtractor(
        temp_dir=tmp_path, ydl_factory=make_factory()  # would write files if called
    )
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(missing))
    result = extractor.extract(valid(REEL_URL), options)
    assert result.success is False
    assert result.failure_category == FailureCategory.COOKIE_FILE_MISSING
    # No temp run directory was created (pre-flight failed before network).
    assert list(tmp_path.iterdir()) == []


def test_cookie_file_mode_without_path_is_missing(tmp_path):
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=make_factory())
    result = extractor.extract(
        valid(REEL_URL), ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=None)
    )
    assert result.failure_category == FailureCategory.COOKIE_FILE_MISSING


def test_empty_cookie_file_is_invalid(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=make_factory())
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(empty))
    result = extractor.extract(valid(REEL_URL), options)
    assert result.failure_category == FailureCategory.COOKIE_FILE_INVALID


def test_unreadable_cookie_file_is_reported(tmp_path, monkeypatch):
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    real_open = Path.open

    def deny(self, *args, **kwargs):
        if self == cookie:
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny)
    extractor = YtDlpExtractor(temp_dir=tmp_path, ydl_factory=make_factory())
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie))
    result = extractor.extract(valid(REEL_URL), options)
    assert result.failure_category == FailureCategory.COOKIE_FILE_UNREADABLE


def test_ydl_rejecting_cookie_format_is_invalid_not_auth(tmp_path):
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    extractor = make_extractor(
        tmp_path,
        raise_exc=Exception("'cookies.txt' does not look like a Netscape format cookies file"),
    )
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie))
    result = extractor.extract(valid(REEL_URL), options)
    assert result.failure_category == FailureCategory.COOKIE_FILE_INVALID
    assert result.failure_category != FailureCategory.AUTH_REQUIRED


def test_cookie_values_never_appear_in_logs_or_result(tmp_path, caplog):
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    extractor = make_extractor(
        tmp_path, info={"id": "x", "description": "hi"}, files=[("x.mp4", b"v" * 1024)]
    )
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie))
    with caplog.at_level(logging.DEBUG):
        result = extractor.extract(valid(REEL_URL), options)
    assert FAKE_SESSION_VALUE not in str(result.as_dict())
    assert FAKE_SESSION_VALUE not in caplog.text


def test_cookie_values_never_appear_in_failure_message(tmp_path):
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    extractor = make_extractor(tmp_path, raise_exc=Exception("some failure with the session"))
    options = ExtractionOptions(mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie))
    result = extractor.extract(valid(REEL_URL), options)
    assert FAKE_SESSION_VALUE not in (result.failure_reason or "")
    assert FAKE_SESSION_VALUE not in (result.error_detail or "")


def test_cookie_file_is_never_copied_into_temp(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    cookie = write_cookie_fixture(tmp_path / "cookies.txt")
    extractor = YtDlpExtractor(
        temp_dir=base,
        ydl_factory=make_factory(info={"id": "x", "description": "hi"}, files=[("x.mp4", b"v" * 1024)]),
    )
    options = ExtractionOptions(
        mode=ExtractionMode.COOKIE_FILE, cookie_file=str(cookie), keep_media=True
    )
    result = extractor.extract(valid(REEL_URL), options)
    assert result.success is True
    # No file anywhere under the temp base contains the fake session value.
    for path in base.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert FAKE_SESSION_VALUE.encode() not in data


def test_no_network_import_in_validator():
    """Phase 2 validator must remain import-clean of network libraries."""
    source = (PROJECT_ROOT / "extractor" / "url_validator.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            for name in ("requests", "urllib.request", "httpx", "socket", "yt_dlp"):
                assert name not in line
