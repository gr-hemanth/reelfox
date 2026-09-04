"""Dedicated output writer for end-to-end run results (Phase 9).

Writes one structured JSON record per complete run to output/runs/<run_id>.json.
Guarantees zero leakage of raw media binaries, cookies, or API credentials.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from processor.run_result import RunResult

logger = logging.getLogger("analyzer.processor.output")

# Deny-list of keys that must NEVER be written to output run records
_FORBIDDEN_KEY_SUBSTRINGS = (
    "cookie",
    "password",
    "secret",
    "api_key",
    "token",
    "auth",
)


def sanitize_dict_for_output(obj: Any) -> Any:
    """Recursively scrub any sensitive authentication values or raw binary data."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(sub in k_lower for sub in _FORBIDDEN_KEY_SUBSTRINGS):
                # Allow safe booleans / configured status flags
                if k in ("cookie_file_configured", "cookies_from_browser"):
                    cleaned[k] = bool(v) if k == "cookie_file_configured" else (str(v) if v else "")
                else:
                    cleaned[k] = "[REDACTED]"
            elif isinstance(v, (bytes, bytearray)):
                cleaned[k] = f"[RAW_BYTES: {len(v)} bytes]"
            else:
                cleaned[k] = sanitize_dict_for_output(v)
        return cleaned
    elif isinstance(obj, list):
        return [sanitize_dict_for_output(item) for item in obj]
    elif isinstance(obj, (bytes, bytearray)):
        return f"[RAW_BYTES: {len(obj)} bytes]"
    return obj


def save_run_result(
    run_result: Union[RunResult, Dict[str, Any]],
    output_dir: Union[Path, str] = "output/runs",
) -> Path:
    """Write one structured JSON record per complete run.

    Parameters
    ----------
    run_result:
        A :class:`RunResult` instance or a raw dictionary.
    output_dir:
        Target directory (default: ``output/runs``).

    Returns
    -------
    Path
        Absolute path to the written JSON record.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(run_result, RunResult):
        raw_dict = run_result.as_dict()
        run_id = run_result.run_id
    elif isinstance(run_result, dict):
        raw_dict = dict(run_result)
        run_id = str(raw_dict.get("run_id") or f"run_{id(run_result)}")
    else:
        raise TypeError(f"Unsupported run_result type: {type(run_result)}")

    cleaned_dict = sanitize_dict_for_output(raw_dict)
    target_path = out_dir / f"{run_id}.json"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dict, f, indent=2, ensure_ascii=False)

    logger.debug("Saved run result to: %s", target_path)
    return target_path
