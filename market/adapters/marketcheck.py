# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
MarketCheck adapter — queries the MarketCheck used-car listings API.

Free tier: 1,000 calls/month at marketcheck.com (sign up → API Keys).
Returns median price across active listings matching year/make/model.

Set MARKETCHECK_API_KEY in .env or config.yaml → market_compare.adapters.marketcheck.api_key.
"""
from __future__ import annotations

import logging
import statistics

import httpx

from market.adapters.base import PricingAdapter, PriceEstimate

logger = logging.getLogger(__name__)

_BASE_URL  = "https://mc-api.marketcheck.com/v2/search/car/active"
_PRICE_MIN = 1_000
_PRICE_MAX = 150_000
_PLACEHOLDER = "YOUR_MARKETCHECK_API_KEY_HERE"


class MarketCheckAdapter(PricingAdapter):
    name = "MarketCheck"

    def __init__(self, api_key: str):
        self._api_key = (api_key or "").strip()

    def _unconfigured(self) -> bool:
        return not self._api_key or self._api_key == _PLACEHOLDER or len(self._api_key) < 8

    def get_estimate(self, vehicle: dict) -> PriceEstimate:
        if self._unconfigured():
            return PriceEstimate(
                source=self.name, price=0, count=0, available=False,
                note="MarketCheck API key not set — sign up free at marketcheck.com",
            )

        year         = str(vehicle.get("year", "")).strip()
        manufacturer = str(vehicle.get("manufacturer", "")).strip().lower()
        model        = str(vehicle.get("model", "")).strip().lower()

        params: dict[str, str | int] = {
            "api_key": self._api_key,
            "car_type": "used",
            "rows": 25,
            "start": 0,
        }
        if year:
            params["year"] = year
        if manufacturer:
            params["make"] = manufacturer
        if model:
            params["model"] = model

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            listings = data.get("listings") or []
            prices: list[float] = []
            for listing in listings:
                price = listing.get("price") or listing.get("dp_price")
                if price:
                    val = float(price)
                    if _PRICE_MIN < val < _PRICE_MAX:
                        prices.append(val)

            if not prices:
                total = data.get("num_found", 0)
                note = f"No priced listings (total found: {total})" if total else "No listings found"
                return PriceEstimate(source=self.name, price=0, count=0, available=False, note=note)

            return PriceEstimate(
                source=self.name,
                price=statistics.median(prices),
                count=len(prices),
                note=f"{len(prices)} active listings · median ${statistics.median(prices):,.0f}",
            )

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                note = "Invalid API key — check marketcheck.com dashboard"
            elif status == 429:
                note = "Rate limit reached — free tier: 1,000 calls/month"
            else:
                note = f"HTTP {status}"
            logger.warning("MarketCheck HTTP error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False, note=note)
        except Exception as exc:
            logger.warning("MarketCheck adapter error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note=str(exc)[:80])
