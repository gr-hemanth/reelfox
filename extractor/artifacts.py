"""Temporary artifact management for extraction runs.

Every extraction gets its own isolated directory under ``temp/`` so that
concurrent or repeated runs never overwrite one another. Media downloaded
there is temporary by policy: nothing in this project promises long-term
storage.

Lifecycle is deliberately caller-controlled. On a *failed* run the extractor
removes its own partial downloads immediately. On a *successful* run the media
must survive so the later multimodal pipeline (Phase 4+) can read it, so the
caller owns cleanup - typically ``with TempRun(...) as run: ...`` or an
explicit ``run.cleanup()`` once the file has been consumed.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("analyzer.extractor.artifacts")


class TempRun:
    """An isolated per-run temporary directory.

    Usable as a context manager. By default the directory is deleted on exit;
    set ``keep=True`` to retain artifacts for debugging.
    """

    def __init__(self, base_dir: Path, keep: bool = False) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / self.run_id
        self.keep = keep
        self._created = False

    def create(self) -> "TempRun":
        """Create the run directory (idempotent)."""
        if not self._created:
            self.path.mkdir(parents=True, exist_ok=True)
            self._created = True
            logger.debug("Created temp run directory %s", self.path)
        return self

    def cleanup(self) -> None:
        """Remove the run directory and everything in it."""
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            logger.debug("Cleaned up temp run directory %s", self.path)
        self._created = False

    def files(self) -> list[Path]:
        """Return every regular file currently in the run directory."""
        if not self.path.exists():
            return []
        return sorted(p for p in self.path.rglob("*") if p.is_file())

    def __enter__(self) -> "TempRun":
        return self.create()

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep:
            self.cleanup()
