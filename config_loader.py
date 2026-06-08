# Copyright (c) 2026 Callisto Tech — see LICENSE

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
