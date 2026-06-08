# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Canonical feature definitions derived from the Used-Car-Study notebook.
Single source of truth for encoding maps, feature lists, and derived columns.
All transforms here must be deterministic and stateless.
"""

from __future__ import annotations

REFERENCE_YEAR = 2025

# Columns dropped at ingest — carry no signal or are too granular
DROP_COLUMNS = ["id", "VIN", "region", "state", "model"]

# Ordinal encoding maps — order reflects quality/value progression
CONDITION_MAP = {
    "salvage": 1,
    "fair": 2,
    "good": 3,
    "excellent": 4,
    "like new": 5,
    "new": 6,
    "Unknown": 0,
}

CYLINDERS_MAP = {
    "3 cylinders": 3,
    "4 cylinders": 4,
    "5 cylinders": 5,
    "6 cylinders": 6,
    "8 cylinders": 8,
    "10 cylinders": 10,
    "12 cylinders": 12,
    "other": 0,
    "Unknown": 0,
}

# Binary: clean title = 1, anything else = 0
CLEAN_TITLE_VALUES = {"clean"}

# Categorical features encoded via mean-target encoding (computed from training data)
MEAN_TARGET_ENCODE_COLS = ["manufacturer", "fuel", "drive", "size", "type", "paint_color"]

# Final feature columns fed to the model (post-encoding)
FEATURE_COLUMNS = [
    "car_age",
    "mileage_per_year",
    "title_status_encoded",
    "condition_encoded",
    "cylinders_encoded",
    "manufacturer_encoded",
    "fuel_encoded",
    "drive_encoded",
    "size_encoded",
    "type_encoded",
    "paint_color_encoded",
]

TARGET_COLUMN = "price"
LOG_TARGET_COLUMN = "log_price"

# Drift monitoring: which features use PSI (continuous) vs chi2 (categorical)
CONTINUOUS_DRIFT_FEATURES = ["car_age", "mileage_per_year", "cylinders_encoded"]
CATEGORICAL_DRIFT_FEATURES = ["fuel", "drive", "type", "condition", "title_status"]
