# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Market aggregator — collects estimates from all active adapters,
computes the overall market median, and signals a price drift when
|model_price - market_median| > drift_threshold.
"""
from __future__ import annotations

import glob
import logging
import os
import statistics
from dataclasses import dataclass, field

from market.adapters.base import PriceEstimate, PricingAdapter

logger = logging.getLogger(__name__)

_instance: "MarketAggregator | None" = None


@dataclass
class MarketComparison:
    model_price: float
    sources: list[PriceEstimate] = field(default_factory=list)
    market_median: float | None = None
    delta: float | None = None
    market_drift_triggered: bool = False
    drift_threshold: float = 1000.0


class MarketAggregator:
    def __init__(self, adapters: list[PricingAdapter], drift_threshold: float = 1000.0):
        self._adapters       = adapters
        self._drift_threshold = drift_threshold

    def compare(self, vehicle: dict, model_price: float) -> MarketComparison:
        estimates: list[PriceEstimate] = []
        for adapter in self._adapters:
            try:
                est = adapter.get_estimate(vehicle)
                estimates.append(est)
            except Exception as exc:
                logger.warning("Adapter %s raised unexpectedly: %s", adapter.name, exc)

        available_prices = [e.price for e in estimates if e.available and e.price > 0]

        market_median: float | None = None
        delta:         float | None = None
        drift = False

        if available_prices:
            market_median = statistics.median(available_prices)
            delta         = abs(model_price - market_median)
            drift         = delta > self._drift_threshold

        return MarketComparison(
            model_price=model_price,
            sources=estimates,
            market_median=market_median,
            delta=delta,
            market_drift_triggered=drift,
            drift_threshold=self._drift_threshold,
        )


def get_aggregator() -> MarketAggregator:
    """Singleton — built once from config on first call."""
    global _instance
    if _instance is not None:
        return _instance

    from config_loader import load_config
    from market.adapters.dataset import DatasetAdapter
    from market.adapters.ebay import EbayAdapter
    from market.adapters.enterprise import EnterpriseAdapter

    cfg    = load_config()
    mc_cfg = cfg.get("market_compare", {})
    ac_cfg = mc_cfg.get("adapters", {})

    adapters: list[PricingAdapter] = []

    if ac_cfg.get("dataset", {}).get("enabled", True):
        adapters.append(DatasetAdapter(
            raw_path=cfg["data"]["raw_path"],
            filename_pattern=cfg["data"]["filename_pattern"],
        ))

    if ac_cfg.get("ebay", {}).get("enabled", True):
        app_id = mc_cfg.get("ebay_app_id", "YOUR_EBAY_APP_ID_HERE")
        adapters.append(EbayAdapter(app_id=app_id))

    if ac_cfg.get("enterprise", {}).get("enabled", False):
        adapters.append(EnterpriseAdapter())

    threshold = float(mc_cfg.get("price_drift_threshold", 1000.0))
    _instance = MarketAggregator(adapters=adapters, drift_threshold=threshold)
    logger.info("MarketAggregator initialised with %d adapter(s)", len(adapters))
    return _instance
