# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Append-only verbatim feedback store backed by newline-delimited JSON.
Each entry preserves the exact comment as submitted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path("data/processed/feedback.jsonl")


def save(comment: str, context: dict | None = None) -> dict:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "comment": comment,
        "context": context or {},
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Feedback saved: %d chars", len(comment))
    return entry


def load_all() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    entries = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
