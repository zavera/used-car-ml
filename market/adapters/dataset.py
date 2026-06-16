# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Dataset adapter — computes price percentiles from the training CSV.

Matches on: manufacturer + year (±2) + condition.
Falls back to manufacturer + year (±3) if fewer than 5 matches.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from market.adapters.base import PricingAdapter, PriceEstimate

logger = logging.getLogger(__name__)

PRICE_MIN = 1000
PRICE_MAX = 150_000
MIN_MATCH  = 5


class DatasetAdapter(PricingAdapter):
    name = "Training Dataset"

    def __init__(self, raw_path: str, filename_pattern: str):
        self._raw_path = raw_path
        self._pattern = filename_pattern
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        files = sorted(glob.glob(os.path.join(self._raw_path, self._pattern)))
        if not files:
            raise FileNotFoundError(f"No CSV found at {self._raw_path}/{self._pattern}")
        df = pd.read_csv(files[-1], usecols=["price", "year", "manufacturer", "condition", "state"])
        df = df.dropna(subset=["price", "year", "manufacturer"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
        df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
        df["manufacturer"] = df["manufacturer"].str.lower().str.strip()
        df["condition"]    = df["condition"].str.lower().str.strip()
        df["state"]        = df["state"].str.lower().str.strip().fillna("")
        self._df = df
        logger.info("DatasetAdapter loaded %d rows from %s", len(df), files[-1])
        return self._df

    def get_estimate(self, vehicle: dict) -> PriceEstimate:
        try:
            df = self._load()
        except Exception as exc:
            return PriceEstimate(source=self.name, price=0, count=0, available=False, note=str(exc)[:80])

        year         = int(vehicle.get("year", 0))
        manufacturer = str(vehicle.get("manufacturer", "")).lower().strip()
        condition    = str(vehicle.get("condition", "")).lower().strip()
        state        = str(vehicle.get("state", "")).lower().strip()
        use_state    = state and state != "unknown" and state in df["state"].values

        def _filter(yr_window: int, use_condition: bool, state_filter: bool) -> pd.Series:
            mask = (
                (df["manufacturer"] == manufacturer)
                & df["year"].between(year - yr_window, year + yr_window)
            )
            if use_condition and condition:
                mask &= df["condition"] == condition
            if state_filter and use_state:
                mask &= df["state"] == state
            return df.loc[mask, "price"].dropna()

        # Try narrow: same state + condition + year ±2
        subset = _filter(2, True, True) if use_state else _filter(2, True, False)
        # Fall back: drop state filter but keep condition
        if len(subset) < MIN_MATCH:
            subset = _filter(2, True, False)
        # Fall back: drop condition too
        if len(subset) < MIN_MATCH:
            subset = _filter(3, False, False)

        if len(subset) < 3:
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note="Too few matches for a reliable estimate")

        p25 = int(np.percentile(subset, 25))
        p75 = int(np.percentile(subset, 75))
        region_note = f" in {state.upper()}" if use_state and len(_filter(2, True, True)) >= MIN_MATCH else ""
        return PriceEstimate(
            source=self.name,
            price=float(np.median(subset)),
            count=len(subset),
            note=f"p25 ${p25:,} – p75 ${p75:,}{region_note}",
        )
