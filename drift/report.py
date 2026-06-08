# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Persists drift reports to disk as JSON for audit trail and dashboarding.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from drift.detector import DriftReport

logger = logging.getLogger(__name__)


def save_report(report: DriftReport, report_dir: str) -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = path / f"drift_{ts}.json"

    payload = {
        "timestamp": ts,
        "triggered": report.triggered,
        "drifted_features": report.drifted_features(),
        "results": [
            {
                "feature": r.feature,
                "test": r.test,
                "statistic": r.statistic,
                "pvalue": r.pvalue,
                "drifted": r.drifted,
                "detail": r.detail,
            }
            for r in report.results
        ],
    }

    filename.write_text(json.dumps(payload, indent=2))
    logger.info("Drift report saved to %s", filename)
    return filename


def load_latest_report(report_dir: str) -> dict | None:
    path = Path(report_dir)
    files = sorted(path.glob("drift_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())
