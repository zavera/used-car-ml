# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Data ingestion layer.
Reads raw CSV drops from data/raw/, validates schema, and hands off to preprocessing.
New data sources (scrapers, APIs) plug in here by implementing the DataSource protocol.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.preprocessing import full_clean_pipeline

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "price", "year", "manufacturer", "condition", "cylinders",
    "fuel", "odometer", "title_status", "transmission", "drive",
    "size", "type", "paint_color",
}


class DataSource(Protocol):
    def fetch(self) -> pd.DataFrame: ...


class CsvDataSource:
    """Loads the most recent CSV from the raw data directory."""

    def __init__(self, raw_path: str, filename_pattern: str = "vehicles_*.csv") -> None:
        self.raw_path = Path(raw_path)
        self.filename_pattern = filename_pattern

    def fetch(self) -> pd.DataFrame:
        files = sorted(glob.glob(str(self.raw_path / self.filename_pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching {self.filename_pattern} in {self.raw_path}"
            )
        latest = files[-1]
        logger.info("Loading raw data from %s", latest)
        return pd.read_csv(latest, low_memory=False)


def validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing expected columns: {missing}")


def ingest(
    source: DataSource,
    encoding_tables: dict | None = None,
    global_mean: float = 0.0,
    price_floor: float = 1000,
    odometer_floor: float = 500,
) -> pd.DataFrame:
    """
    Fetch raw data, validate, clean, and return a feature-ready DataFrame.
    encoding_tables=None → compute fresh tables (use for initial training).
    encoding_tables=<dict> → apply frozen tables (use for incremental / inference).
    """
    raw = source.fetch()
    logger.info("Fetched %d raw rows", len(raw))
    validate_schema(raw)
    cleaned = full_clean_pipeline(
        raw,
        encoding_tables=encoding_tables,
        global_mean=global_mean,
        price_floor=price_floor,
        odometer_floor=odometer_floor,
    )
    logger.info("Cleaned to %d rows", len(cleaned))
    return cleaned
