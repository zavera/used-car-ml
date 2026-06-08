# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Loads the champion model artifact and runs inference.
Handles both ridge (sklearn Pipeline) and poly (dict bundle) formats.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from data.preprocessing import (
    apply_mean_target_encoding,
    encode_ordinals,
    engineer_features,
)
from feature_store.definitions import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

CHAMPION_PATH = "models/champion.pkl"


class Predictor:
    def __init__(self, champion_path: str = CHAMPION_PATH) -> None:
        self._path = Path(champion_path)
        self._artifact: dict | None = None
        self.reload()

    def reload(self) -> None:
        if not self._path.exists():
            logger.warning("No champion model found at %s", self._path)
            self._artifact = None
            return
        with open(self._path, "rb") as f:
            self._artifact = pickle.load(f)
        logger.info("Loaded champion model '%s'", self._artifact["model_name"])

    def predict(self, raw_input: dict) -> dict:
        """
        raw_input fields: year, manufacturer, condition, cylinders, fuel,
                          odometer, title_status, drive, size, type, paint_color
        Returns: { "predicted_price": float, "model": str }
        """
        if self._artifact is None:
            raise RuntimeError("No model loaded. Run training first.")

        import pandas as pd
        df = pd.DataFrame([raw_input])

        df = encode_ordinals(df)
        df = engineer_features(df)
        df = apply_mean_target_encoding(
            df,
            self._artifact["encoding_tables"],
            self._artifact["global_mean"],
        )

        features = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[features].values

        model_name = self._artifact["model_name"]
        model = self._artifact["model"]

        if model_name == "poly":
            X_s = model["scaler"].transform(X)
            X_p = model["poly"].transform(X_s)
            X_sel = model["sfs"].transform(X_p)
            log_price = model["lr"].predict(X_sel)[0]
        else:
            log_price = model.predict(X)[0]

        price = float(np.expm1(log_price))
        return {"predicted_price": round(price, 2), "model": model_name}
