# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Tavily adapter — uses Tavily Search to pull real-time used car pricing
from credible sources (KBB, Edmunds, CarGurus, AutoTrader, Cars.com).

One Tavily query returns price signals from multiple sites simultaneously,
giving broad market coverage without per-site scraping.

Set TAVILY_API_KEY in .env or config.yaml → market_compare.adapters.tavily.api_key.
"""
from __future__ import annotations

import logging
import re
import statistics

from market.adapters.base import PricingAdapter, PriceEstimate

logger = logging.getLogger(__name__)

_PRICE_MIN  = 1_000
_PRICE_MAX  = 150_000
_PLACEHOLDER = "YOUR_TAVILY_API_KEY_HERE"

# Sites Tavily will prefer when searching
_TRUSTED_DOMAINS = [
    "kbb.com",
    "edmunds.com",
    "cargurus.com",
    "autotrader.com",
    "cars.com",
]


class TavilyAdapter(PricingAdapter):
    name = "Tavily (KBB/Edmunds/CarGurus)"

    def __init__(self, api_key: str):
        self._api_key = (api_key or "").strip()

    def _unconfigured(self) -> bool:
        return not self._api_key or self._api_key == _PLACEHOLDER or len(self._api_key) < 10

    def get_estimate(self, vehicle: dict) -> PriceEstimate:
        if self._unconfigured():
            return PriceEstimate(
                source=self.name, price=0, count=0, available=False,
                note="Tavily API key not set — add TAVILY_API_KEY to .env",
            )

        year         = str(vehicle.get("year", "")).strip()
        manufacturer = str(vehicle.get("manufacturer", "")).strip()
        model        = str(vehicle.get("model", "")).strip()
        vtype        = str(vehicle.get("type", "")).strip()
        odometer     = vehicle.get("odometer")

        vehicle_str = f"{year} {manufacturer} {model}".strip() if model else f"{year} {manufacturer} {vtype}".strip()
        mileage_hint = f" {int(odometer):,} miles" if odometer else ""
        query = f"{vehicle_str} used car value price{mileage_hint} site:kbb.com OR site:edmunds.com OR site:cargurus.com OR site:autotrader.com OR site:cars.com"

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self._api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=8,
                include_domains=_TRUSTED_DOMAINS,
            )

            prices = _extract_prices(response)

            if not prices:
                return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                     note="No price signals found in search results")

            sources_hit = _source_names(response)
            return PriceEstimate(
                source=self.name,
                price=statistics.median(prices),
                count=len(prices),
                note=f"{len(prices)} prices from: {', '.join(sources_hit)}",
            )

        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "unauthorized" in msg.lower():
                note = "Invalid Tavily API key"
            elif "429" in msg or "rate" in msg.lower():
                note = "Tavily rate limit reached"
            else:
                note = msg[:80]
            logger.warning("Tavily adapter error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False, note=note)


def _extract_prices(response: dict) -> list[float]:
    """Pull dollar amounts from Tavily result snippets and answer field."""
    prices: list[float] = []
    texts: list[str] = []

    if response.get("answer"):
        texts.append(response["answer"])

    for result in response.get("results", []):
        if result.get("content"):
            texts.append(result["content"])

    for text in texts:
        # Match $12,500 or $12500 with optional range (- $23,000)
        dollar_matches = re.findall(
            r'\$\s*(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\s*[-–]\s*\$\s*(\d{1,3}(?:,\d{3})+|\d{4,6}))?',
            text,
        )
        for low, high in dollar_matches:
            for val_str in (low, high):
                if val_str:
                    val = float(val_str.replace(",", ""))
                    if _PRICE_MIN < val < _PRICE_MAX:
                        prices.append(val)

        # Match bare numbers in listing context: "Sport; 89K mi. 16,995" or "Used ... 14,500"
        # Only capture standalone 5-digit comma-separated numbers not preceded by other digits
        bare_matches = re.findall(r'(?<!\d)(\d{2},\d{3})(?!\d)', text)
        for val_str in bare_matches:
            val = float(val_str.replace(",", ""))
            if _PRICE_MIN < val < _PRICE_MAX:
                prices.append(val)

    return prices


def _source_names(response: dict) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for result in response.get("results", []):
        url = result.get("url", "")
        for domain in _TRUSTED_DOMAINS:
            short = domain.split(".")[0]
            if domain in url and short not in seen:
                seen.add(short)
                names.append(short.upper() if short == "kbb" else short.title())
    return names or ["web"]
