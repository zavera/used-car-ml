# Copyright (c) 2026 Callisto Tech — see LICENSE

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def load_config(path: str = "config/config.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    ebay_app_id = os.getenv("EBAY_APP_ID")
    if ebay_app_id:
        cfg.setdefault("market_compare", {})["ebay_app_id"] = ebay_app_id
    return cfg
