# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Application entry point.
Starts the scheduler, then hands off to uvicorn.
"""

from __future__ import annotations

import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from config_loader import load_config
from orchestration.scheduler import start_scheduler
from serving.api import app

if __name__ == "__main__":
    cfg = load_config()
    start_scheduler()
    uvicorn.run(
        app,
        host=cfg["serving"]["host"],
        port=cfg["serving"]["port"],
        log_level="info",
    )
