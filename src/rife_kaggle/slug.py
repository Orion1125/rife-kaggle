"""Slug helpers — Kaggle ids must be lowercase, hyphenated, <=50 chars."""

from __future__ import annotations

import re
import time
from pathlib import Path

_SLUG_OK = re.compile(r"[^a-z0-9-]")
MAX_LEN = 40


def make_slug(video_path: Path, suffix: str | None = None) -> str:
    """Build a deterministic-ish slug for a video upload.

    Format: ``<basename>-<unix-ts>`` clipped to MAX_LEN. The timestamp keeps
    runs distinct; without it Kaggle would refuse the second upload of a
    video with the same name.
    """
    base = _slugify(video_path.stem)
    stamp = str(int(time.time()))
    parts = [base, stamp]
    if suffix:
        parts.append(_slugify(suffix))
    out = "-".join(p for p in parts if p)
    return out[:MAX_LEN].rstrip("-") or f"video-{stamp}"


def _slugify(value: str) -> str:
    lowered = value.lower().replace("_", "-").replace(" ", "-")
    cleaned = _SLUG_OK.sub("", lowered)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned
