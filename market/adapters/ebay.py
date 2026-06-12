# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
eBay Motors adapter — queries completed/sold listings via the eBay Finding API.

Requires an eBay developer App ID (free at developer.ebay.com).
Set ebay_app_id in config/config.yaml under market_compare.
The adapter returns unavailable gracefully when the key is a placeholder.
"""
from __future__ import annotations

import logging
import statistics

import httpx

from market.adapters.base import PricingAdapter, PriceEstimate

logger = logging.getLogger(__name__)

_FINDING_URL    = "https://svcs.ebay.com/services/search/FindingService/v1"
_MOTORS_CAT     = "6001"   # eBay Motors → Cars & Trucks
_PRICE_MIN      = 1_000
_PRICE_MAX      = 150_000
_PLACEHOLDER    = "YOUR_EBAY_APP_ID_HERE"


class EbayAdapter(PricingAdapter):
    name = "eBay Motors"

    def __init__(self, app_id: str):
        self._app_id = (app_id or "").strip()

    def _unconfigured(self) -> bool:
        return (
            not self._app_id
            or self._app_id == _PLACEHOLDER
            or len(self._app_id) < 10
        )

    def get_estimate(self, vehicle: dict) -> PriceEstimate:
        if self._unconfigured():
            return PriceEstimate(
                source=self.name, price=0, count=0, available=False,
                note="eBay App ID not configured — add to config.yaml"
            )

        year         = vehicle.get("year", "")
        manufacturer = vehicle.get("manufacturer", "")
        vtype        = vehicle.get("type", "")
        keywords     = f"{year} {manufacturer} {vtype}".strip()

        params = {
            "OPERATION-NAME":               "findCompletedItems",
            "SERVICE-VERSION":              "1.0.0",
            "SECURITY-APPNAME":             self._app_id,
            "RESPONSE-DATA-FORMAT":         "JSON",
            "keywords":                     keywords,
            "categoryId":                   _MOTORS_CAT,
            "itemFilter(0).name":           "SoldItemsOnly",
            "itemFilter(0).value":          "true",
            "paginationInput.entriesPerPage": "25",
        }

        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(_FINDING_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = (
                data
                .get("findCompletedItemsResponse", [{}])[0]
                .get("searchResult",               [{}])[0]
                .get("item",                        [])
            )

            prices = []
            for item in items:
                try:
                    raw = item["sellingStatus"][0]["convertedCurrentPrice"][0]["__value__"]
                    price = float(raw)
                    if _PRICE_MIN < price < _PRICE_MAX:
                        prices.append(price)
                except (KeyError, IndexError, ValueError):
                    continue

            if not prices:
                return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                     note="No sold listings found")

            return PriceEstimate(
                source=self.name,
                price=statistics.median(prices),
                count=len(prices),
            )

        except httpx.HTTPStatusError as exc:
            logger.warning("eBay API HTTP error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            logger.warning("eBay adapter error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note=str(exc)[:60])
