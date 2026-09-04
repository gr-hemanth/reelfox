"""Project and CLI smoke tests.

Fully offline. The one test that exercises the valid-URL path injects a fake
extractor so pytest never touches Instagram.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import analyzer  # noqa: E402
import config  # noqa: E402
from extractor import ExtractionResult, ValidationResult  # noqa: E402

VALID_URL = "https://www.instagram.com/reel/example/"
FOREIGN_URL = "https://youtube.com/watch?v=test"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Execute analyzer.py in a subprocess."""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "analyzer.py"), *args],
        capture_output=True,
        text=True,
    )


class _FakeExtractor:
    """Stand-in for YtDlpExtractor that returns a canned success result."""

    def __init__(self, *args, **kwargs):
        pass

    def extract(self, validation: ValidationResult, options=None) -> ExtractionResult:
        return ExtractionResult(
            success=True,
            source_url=validation.input_url,
            normalized_url=validation.normalized_url,
            content_type_hint=validation.content_type_hint,
            media_downloaded=True,
            media_path="temp/run/abc.mp4",
            caption_extracted=True,
            caption="hello #world",
            hashtags=["#world"],
            media_type_detected=True,
            media_type="reel",
            run_id="run",
        )


def test_modules_import():
    assert analyzer.__doc__
    assert config.PROJECT_NAME == "Instagram Content Analyzer"
    assert config.PHASE_NUMBER == 9
    assert config.PHASE == "Benchmark Infrastructure"




def test_expected_modules_exist():
    for name in ("__init__", "url_validator", "instagram_extractor", "models", "errors"):
        assert (PROJECT_ROOT / "extractor" / f"{name}.py").is_file()


def test_config_loads_without_secrets():
    settings = config.get_config()
    assert settings.log_level in config.VALID_LOG_LEVELS
    assert settings.extraction_mode in config.VALID_EXTRACTION_MODES
    assert settings.base_dir == PROJECT_ROOT


def test_environment_check_passes():
    ok, problems = config.get_config().check_environment()
    assert ok, f"environment problems: {problems}"


def test_parser_accepts_optional_url_and_flags():
    parser = analyzer.build_parser()
    assert parser.parse_args([]).url is None
    assert parser.parse_args([VALID_URL]).url == VALID_URL
    assert parser.parse_args(["--keep-media", VALID_URL]).keep_media is True


def test_run_with_valid_url_uses_extractor(monkeypatch, capsys):
    monkeypatch.setattr(analyzer, "YtDlpExtractor", _FakeExtractor)
    from processor.synthesis_models import MultimodalAnalysisResult
    monkeypatch.setattr(
        "processor.pipeline.process_synthesis",
        lambda *a, **kw: MultimodalAnalysisResult(
            success=True,
            summary="Fake summary",
            key_points=["Fake point"],
            core_takeaway="Fake takeaway",
            confidence=0.9,
            evidence_used={"caption": True, "speech": False, "vision": False, "ocr": False},
        ),
    )
    exit_code = analyzer.run([VALID_URL])
    output = capsys.readouterr().out
    assert exit_code == analyzer.EXIT_OK
    assert "URL validation: PASS" in output
    assert "Extraction: PASS" in output
    assert "Media downloaded: PASS" in output
    assert "reel" in output
    assert analyzer.FOOTER in output



def test_run_with_invalid_url_stops_before_extraction(capsys):
    exit_code = analyzer.run([FOREIGN_URL])
    output = capsys.readouterr().out
    assert exit_code == analyzer.EXIT_INVALID_URL
    assert "URL validation: FAIL" in output
    assert "NON_INSTAGRAM_DOMAIN" in output
    # Extraction section never printed.
    assert "Extraction:" not in output


def test_run_without_url_prints_usage(capsys):
    exit_code = analyzer.run([])
    output = capsys.readouterr().out
    assert exit_code == analyzer.EXIT_USAGE
    assert "Usage:" in output


def test_empty_string_url_is_a_validation_failure(capsys):
    exit_code = analyzer.run([""])
    output = capsys.readouterr().out
    assert exit_code == analyzer.EXIT_INVALID_URL
    assert "EMPTY_INPUT" in output


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_cli_flags_exit_cleanly(flag):
    result = run_cli(flag)
    assert result.returncode == 0, result.stderr


def test_cli_subprocess_with_rejected_url_is_offline():
    """A rejected URL never reaches extraction, so the subprocess stays offline."""
    result = run_cli("not-a-url")
    assert result.returncode == analyzer.EXIT_INVALID_URL, result.stderr
    assert "URL validation: FAIL" in result.stdout
    assert "INVALID_URL" in result.stdout


def test_cli_stdout_stays_clean_of_logging():
    """Logging goes to stderr so stdout remains machine-friendly."""
    result = run_cli("--verbose", "not-a-url")
    assert result.returncode == analyzer.EXIT_INVALID_URL, result.stderr
    assert "DEBUG" not in result.stdout
    assert "|" not in result.stdout
