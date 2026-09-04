"""Configuration handling for the Instagram Content Analyzer.

Phase 1 scope: read a small set of values from the environment (optionally
from a local ``.env`` file), expose them through one immutable object, and
set up logging.

Design intent: later phases add fields to :class:`Config` and read them from
``.env``. Nothing else in the project should call ``os.environ`` directly, so
the configuration surface stays in one file.

No secret is ever hardcoded here. Keys live in ``.env`` (git-ignored); the
committed ``.env.example`` documents which keys exist.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at runtime
    load_dotenv = None


# Project root: the directory containing this file.
BASE_DIR = Path(__file__).resolve().parent

PROJECT_NAME = "Instagram Content Analyzer"
PHASE = "Multimodal Synthesis"
PHASE_NUMBER = 8

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

#: Extraction authentication modes. "public" needs no Instagram session;
#: "browser_cookies" is a hook for later use of a local browser's cookies.
VALID_EXTRACTION_MODES = ("public", "browser_cookies", "cookie_file")

_DEFAULTS = {
    "LOG_LEVEL": "INFO",
    "OUTPUT_DIR": "output",
    "TEMP_DIR": "temp",
    "EXTRACTION_MODE": "public",
    "COOKIES_FROM_BROWSER": "",
    "COOKIE_FILE": "",
    # Phase 5 ASR defaults.
    "ASR_MODEL_SIZE": "base",
    "ASR_DEVICE": "cpu",
    "ASR_COMPUTE_TYPE": "int8",
    # Phase 6 Vision defaults.
    "VISION_MODEL": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "VISION_DEVICE": "auto",
    "VISION_MAX_FRAMES": "6",
    # Phase 7 OCR defaults.
    "OCR_ENGINE": "rapidocr",
    "OCR_MAX_FRAMES": "6",
    # Phase 8 Multimodal Synthesis defaults.
    "SYNTHESIS_BACKEND": "local",
    "SYNTHESIS_MODEL": "Qwen/Qwen2.5-3B-Instruct",
    "SYNTHESIS_DEVICE": "cpu",
    "TOKENROUTER_API_KEY": "",
    "SYNTHESIS_ENDPOINT": "https://api.tokenrouter.com/v1/chat/completions",
    "SYNTHESIS_ENABLED": "true",
    "SYNTHESIS_TIMEOUT": "60",
}


def _env(key: str) -> str:
    """Return an environment value, falling back to the built-in default."""
    value = os.environ.get(key, "").strip()
    return value or _DEFAULTS[key]


def _resolve(path_value: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


@dataclass(frozen=True)
class Config:
    """Runtime configuration.

    Later phases extend this with extraction, model and API settings.
    """

    project_name: str = PROJECT_NAME
    phase: str = PHASE
    phase_number: int = PHASE_NUMBER

    base_dir: Path = BASE_DIR
    output_dir: Path = field(default_factory=lambda: _resolve(_DEFAULTS["OUTPUT_DIR"]))
    temp_dir: Path = field(default_factory=lambda: _resolve(_DEFAULTS["TEMP_DIR"]))

    log_level: str = _DEFAULTS["LOG_LEVEL"]

    # Phase 3 / 4 extraction settings.
    extraction_mode: str = _DEFAULTS["EXTRACTION_MODE"]
    cookies_from_browser: str = ""
    # Path only; the cookie file's contents are never read or stored here.
    cookie_file: str = ""

    # Phase 5 ASR configuration.
    asr_model_size: str = _DEFAULTS["ASR_MODEL_SIZE"]
    asr_device: str = _DEFAULTS["ASR_DEVICE"]
    asr_compute_type: str = _DEFAULTS["ASR_COMPUTE_TYPE"]

    # Phase 6 Vision configuration.
    vision_model: str = _DEFAULTS["VISION_MODEL"]
    vision_device: str = _DEFAULTS["VISION_DEVICE"]
    vision_max_frames: int = int(_DEFAULTS["VISION_MAX_FRAMES"])

    # Phase 7 OCR configuration.
    ocr_engine: str = _DEFAULTS["OCR_ENGINE"]
    ocr_max_frames: int = int(_DEFAULTS["OCR_MAX_FRAMES"])

    # Phase 8 Multimodal Synthesis configuration.
    synthesis_backend: str = _DEFAULTS["SYNTHESIS_BACKEND"]
    synthesis_model: str = _DEFAULTS["SYNTHESIS_MODEL"]
    synthesis_device: str = _DEFAULTS["SYNTHESIS_DEVICE"]
    tokenrouter_api_key: str = field(default="", repr=False)
    synthesis_endpoint: str = _DEFAULTS["SYNTHESIS_ENDPOINT"]
    synthesis_enabled: bool = True
    synthesis_timeout: float = 60.0

    # Placeholders for later phases. Empty string means "not configured".
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @classmethod
    def load(cls, dotenv_path: Path | None = None) -> "Config":
        """Build a :class:`Config` from the environment and ``.env`` file."""
        if load_dotenv is not None:
            load_dotenv(dotenv_path or (BASE_DIR / ".env"), override=False)

        log_level = _env("LOG_LEVEL").upper()
        if log_level not in VALID_LOG_LEVELS:
            log_level = _DEFAULTS["LOG_LEVEL"]

        extraction_mode = _env("EXTRACTION_MODE").lower()
        if extraction_mode not in VALID_EXTRACTION_MODES:
            extraction_mode = _DEFAULTS["EXTRACTION_MODE"]

        cookie_file_raw = os.environ.get("COOKIE_FILE", "").strip()
        cookie_file = str(_resolve(cookie_file_raw)) if cookie_file_raw else ""

        try:
            max_frames = int(_env("VISION_MAX_FRAMES"))
        except ValueError:
            max_frames = 6

        try:
            ocr_max_frames = int(_env("OCR_MAX_FRAMES"))
        except ValueError:
            ocr_max_frames = 6

        synthesis_enabled_raw = _env("SYNTHESIS_ENABLED").lower()
        synthesis_enabled = synthesis_enabled_raw in ("true", "1", "yes")

        try:
            synthesis_timeout = float(_env("SYNTHESIS_TIMEOUT"))
        except ValueError:
            synthesis_timeout = 60.0

        return cls(
            output_dir=_resolve(_env("OUTPUT_DIR")),
            temp_dir=_resolve(_env("TEMP_DIR")),
            log_level=log_level,
            extraction_mode=extraction_mode,
            cookies_from_browser=os.environ.get("COOKIES_FROM_BROWSER", "").strip(),
            cookie_file=cookie_file,
            asr_model_size=_env("ASR_MODEL_SIZE").lower(),
            asr_device=_env("ASR_DEVICE").lower(),
            asr_compute_type=_env("ASR_COMPUTE_TYPE").lower(),
            vision_model=_env("VISION_MODEL"),
            vision_device=_env("VISION_DEVICE").lower(),
            vision_max_frames=max_frames,
            ocr_engine=_env("OCR_ENGINE"),
            ocr_max_frames=ocr_max_frames,
            synthesis_backend=_env("SYNTHESIS_BACKEND").lower() or _DEFAULTS["SYNTHESIS_BACKEND"],
            synthesis_model=_env("SYNTHESIS_MODEL") or _DEFAULTS["SYNTHESIS_MODEL"],
            synthesis_device=_env("SYNTHESIS_DEVICE").lower() or _DEFAULTS["SYNTHESIS_DEVICE"],
            tokenrouter_api_key=os.environ.get("TOKENROUTER_API_KEY", "").strip(),
            synthesis_endpoint=_env("SYNTHESIS_ENDPOINT"),
            synthesis_enabled=synthesis_enabled,
            synthesis_timeout=synthesis_timeout,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    def ensure_directories(self) -> None:
        """Create the working directories if they do not exist yet."""
        for directory in (self.output_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def check_environment(self) -> tuple[bool, list[str]]:
        """Verify the basics Phase 1 depends on.

        Returns ``(ok, problems)``. ``problems`` is empty when ``ok`` is True.
        """
        problems: list[str] = []

        if load_dotenv is None:
            problems.append(
                "python-dotenv is not installed (run: pip install -r requirements.txt)"
            )

        try:
            self.ensure_directories()
        except OSError as exc:
            problems.append(f"cannot create working directories: {exc}")

        for label, directory in (("output", self.output_dir), ("temp", self.temp_dir)):
            if not directory.is_dir():
                problems.append(f"{label} directory missing: {directory}")

        return (not problems, problems)


def configure_logging(level: str = _DEFAULTS["LOG_LEVEL"]) -> logging.Logger:
    """Set up simple console logging and return the project logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("analyzer")


def get_config() -> Config:
    """Convenience wrapper used by the CLI and by tests."""
    return Config.load()
