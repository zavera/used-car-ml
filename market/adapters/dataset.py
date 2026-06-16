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
        df = pd.read_csv(files[-1], usecols=["price", "year", "manufacturer", "model", "condition", "state"])
        df = df.dropna(subset=["price", "year", "manufacturer"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
        df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
        df["manufacturer"] = df["manufacturer"].str.lower().str.strip()
        df["condition"]    = df["condition"].str.lower().str.strip()
        df["state"]        = df["state"].str.lower().str.strip().fillna("")
        df["model"]        = df["model"].astype(str).str.lower().str.strip()
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
        model        = str(vehicle.get("model", "")).lower().strip()
        use_state    = state and state != "unknown" and state in df["state"].values
        use_model    = model and model in df["model"].values

        def _filter(yr_window: int, use_condition: bool, state_filter: bool, model_filter: bool) -> pd.Series:
            mask = (
                (df["manufacturer"] == manufacturer)
                & df["year"].between(year - yr_window, year + yr_window)
            )
            if use_condition and condition:
                mask &= df["condition"] == condition
            if state_filter and use_state:
                mask &= df["state"] == state
            if model_filter and use_model:
                mask &= df["model"] == model
            return df.loc[mask, "price"].dropna()

        # Tiers from most to least specific: model+state -> model -> state -> manufacturer only
        used_model = False
        used_state = False
        subset = pd.Series(dtype=float)
        for m_filt, s_filt in [(True, True), (True, False), (False, True), (False, False)]:
            candidate = _filter(2, True, s_filt, m_filt)
            if len(candidate) >= MIN_MATCH:
                subset, used_model, used_state = candidate, m_filt and use_model, s_filt and use_state
                break
            subset = candidate  # keep the widest attempt as fallback if none hit MIN_MATCH
        # Final fallback: drop condition, widen year window
        if len(subset) < MIN_MATCH:
            subset = _filter(3, False, False, False)
            used_model = used_state = False

        if len(subset) < 3:
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note="Too few matches for a reliable estimate")

        p25 = int(np.percentile(subset, 25))
        p75 = int(np.percentile(subset, 75))
        tags = []
        if used_model: tags.append(model.title())
        if used_state: tags.append(state.upper())
        tag_note = f" ({', '.join(tags)})" if tags else ""
        return PriceEstimate(
            source=self.name,
            price=float(np.median(subset)),
            count=len(subset),
            note=f"p25 ${p25:,} – p75 ${p75:,}{tag_note}",
        )
