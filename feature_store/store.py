# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Parquet-backed feature store.

Two partitions:
  - reference: the snapshot used to train the current champion model (drift baseline)
  - current:   the live accumulating feature set (new rows appended here)

The store is append-only. Retraining snapshots the current partition into reference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, store_path: str, reference_path: str) -> None:
        self.store_path = Path(store_path)
        self.reference_path = Path(reference_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)

    # ── writes ──────────────────────────────────────────────────────────────

    def append(self, df: pd.DataFrame) -> None:
        """Append new feature rows to the current partition."""
        if self.store_path.exists():
            existing = pd.read_parquet(self.store_path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()
        combined.to_parquet(self.store_path, index=False)
        logger.info("Feature store: %d total rows after append", len(combined))

    def snapshot_as_reference(self) -> None:
        """
        Promote current feature store to reference baseline.
        Called immediately before a retrain so drift comparisons
        use the data the new model was trained on.
        """
        if not self.store_path.exists():
            raise FileNotFoundError("No current feature store to snapshot.")
        current = pd.read_parquet(self.store_path)
        current.to_parquet(self.reference_path, index=False)
        logger.info("Snapshotted %d rows as drift reference", len(current))

    def replace(self, df: pd.DataFrame) -> None:
        """Overwrite the current partition (used after full retrain with fresh data)."""
        df.to_parquet(self.store_path, index=False)
        logger.info("Feature store replaced with %d rows", len(df))

    # ── reads ────────────────────────────────────────────────────────────────

    def load_current(self) -> pd.DataFrame:
        if not self.store_path.exists():
            raise FileNotFoundError("Feature store is empty — run ingestion first.")
        return pd.read_parquet(self.store_path)

    def load_reference(self) -> pd.DataFrame:
        if not self.reference_path.exists():
            raise FileNotFoundError(
                "No reference snapshot found — train a model first to establish baseline."
            )
        return pd.read_parquet(self.reference_path)

    def new_row_count(self) -> int:
        """Rows in current that exceed the reference snapshot size."""
        current_len = len(pd.read_parquet(self.store_path)) if self.store_path.exists() else 0
        ref_len = len(pd.read_parquet(self.reference_path)) if self.reference_path.exists() else 0
        return max(0, current_len - ref_len)
