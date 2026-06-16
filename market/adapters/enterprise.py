# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Enterprise Car Sales adapter — submits vehicle details to the Enterprise
value-your-vehicle tool and returns their offer price.

Requires `playwright` (`pip install playwright && playwright install chromium`).
Falls back gracefully if Playwright is not installed or the page changes.

Enterprise form requires: Year, Make, Model, Mileage, ZIP, Color, Condition.
We derive Model from type (e.g. "sedan" → generic placeholder) since our
dataset has body type rather than model name.
"""
from __future__ import annotations

import logging

from market.adapters.base import PricingAdapter, PriceEstimate

logger = logging.getLogger(__name__)

ENTERPRISE_URL = "https://www.enterprisecarsales.com/value-your-vehicle/"

# Map our body types to a common model search term Enterprise understands
_TYPE_TO_MODEL: dict[str, str] = {
    "sedan":       "Camry",
    "SUV":         "Explorer",
    "pickup":      "F-150",
    "truck":       "Silverado",
    "coupe":       "Mustang",
    "hatchback":   "Civic",
    "wagon":       "Outback",
    "van":         "Transit",
    "convertible": "Mustang",
    "mini-van":    "Odyssey",
    "offroad":     "Wrangler",
    "other":       "Camry",
}

# Map state abbreviations → representative ZIP (major city)
_STATE_TO_ZIP: dict[str, str] = {
    "ak": "99501", "al": "35203", "ar": "72201", "az": "85001",
    "ca": "90001", "co": "80201", "ct": "06101", "dc": "20001",
    "de": "19901", "fl": "32099", "ga": "30301", "hi": "96801",
    "id": "83701", "il": "60601", "ma": "02101", "nc": "27601",
    "ny": "10001", "or": "97201", "pa": "17101", "tx": "73301",
    "wa": "98001", "wi": "53201",
}


class EnterpriseAdapter(PricingAdapter):
    name = "Enterprise Car Sales"

    def get_estimate(self, vehicle: dict) -> PriceEstimate:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            return PriceEstimate(
                source=self.name, price=0, count=0, available=False,
                note="playwright not installed — run: pip install playwright && playwright install chromium",
            )

        year         = str(vehicle.get("year", ""))
        manufacturer = str(vehicle.get("manufacturer", "")).strip().title()
        vtype        = str(vehicle.get("type", "sedan"))
        odometer     = str(int(vehicle.get("odometer", 0)))
        state        = str(vehicle.get("state", "")).lower().strip()
        zip_code     = str(vehicle.get("zip_code", "")).strip()
        condition    = str(vehicle.get("condition", "good")).lower()
        color        = str(vehicle.get("paint_color", "white")).strip().title()

        # Resolve ZIP
        if not zip_code or len(zip_code) != 5:
            zip_code = _STATE_TO_ZIP.get(state, "10001")

        # Map condition to Enterprise tiers
        _cond_map = {
            "new": "Excellent", "like new": "Excellent",
            "excellent": "Excellent", "good": "Very Good",
            "fair": "Good", "salvage": "Poor",
        }
        ent_condition = _cond_map.get(condition, "Very Good")

        real_model = str(vehicle.get("model", "")).strip().title()
        guessed_model = real_model or _TYPE_TO_MODEL.get(vtype, "Camry")

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ))
                page.goto(ENTERPRISE_URL, timeout=15_000, wait_until="domcontentloaded")

                # Step 1: Year
                year_sel = page.locator("select[name*='year'], select[id*='year']").first
                year_sel.select_option(year)
                page.wait_for_timeout(400)

                # Step 2: Make
                make_sel = page.locator("select[name*='make'], select[id*='make']").first
                make_sel.select_option(label=manufacturer, timeout=5000)
                page.wait_for_timeout(400)

                # Step 3: Model (best guess from body type)
                model_sel = page.locator("select[name*='model'], select[id*='model']").first
                try:
                    model_sel.select_option(label=guessed_model, timeout=2000)
                except PWTimeout:
                    opts = model_sel.locator("option").all()
                    if len(opts) > 1:
                        opts[1].evaluate("el => el.selected = true")
                page.wait_for_timeout(300)

                # Step 4: Trim — pick first available
                trim_sel = page.locator("select[name*='trim'], select[id*='trim']").first
                try:
                    trim_opts = trim_sel.locator("option").all()
                    if len(trim_opts) > 1:
                        trim_sel.select_option(index=1)
                except Exception:
                    pass
                page.wait_for_timeout(300)

                # Step 5: Mileage
                mile_input = page.locator("input[name*='mileage'], input[id*='mileage'], input[placeholder*='ileage']").first
                mile_input.fill(odometer, timeout=3000)
                page.wait_for_timeout(200)

                # Step 6: ZIP
                zip_input = page.locator("input[name*='zip'], input[id*='zip'], input[placeholder*='ZIP']").first
                zip_input.fill(zip_code, timeout=3000)
                page.wait_for_timeout(200)

                # Step 7: Submit / Next
                submit = page.locator("button[type='submit'], input[type='submit']").first
                submit.click()
                page.wait_for_timeout(2000)

                # Extract price — common patterns: $XX,XXX
                import re
                text = page.inner_text("body")
                prices = re.findall(r'\$\s*(\d{1,3}(?:,\d{3})+)', text)
                prices = [int(p.replace(",", "")) for p in prices if 500 < int(p.replace(",", "")) < 150_000]

                browser.close()

                if not prices:
                    return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                         note="No price found on Enterprise page")

                return PriceEstimate(
                    source=self.name,
                    price=float(prices[0]),
                    count=1,
                    note=f"ZIP {zip_code} · {'model: ' if real_model else 'model est.: '}{guessed_model}",
                )

        except PWTimeout:
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note="Enterprise page timed out")
        except Exception as exc:
            logger.warning("EnterpriseAdapter error: %s", exc)
            return PriceEstimate(source=self.name, price=0, count=0, available=False,
                                 note=str(exc)[:80])
