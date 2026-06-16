# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Stateless cleaning and encoding functions.
All functions take a DataFrame and return a new DataFrame — no mutation.
Mean-target encoding tables are passed in explicitly so this module
never touches disk or the feature store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_store.definitions import (
    CLEAN_TITLE_VALUES,
    CONDITION_MAP,
    CYLINDERS_MAP,
    DROP_COLUMNS,
    LOG_TARGET_COLUMN,
    MEAN_TARGET_ENCODE_COLS,
    REFERENCE_YEAR,
    TARGET_COLUMN,
)


def drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols)


def fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("state", "model"):
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()
    cat_cols = df.select_dtypes(include="object").columns
    return df.assign(**{c: df[c].fillna("Unknown") for c in cat_cols})


def remove_sparse_rows(df: pd.DataFrame, max_unknowns: int = 5) -> pd.DataFrame:
    unknown_count = (df == "Unknown").sum(axis=1)
    return df[unknown_count <= max_unknowns].copy()


def filter_outliers(
    df: pd.DataFrame,
    price_floor: float = 1000,
    odometer_floor: float = 500,
) -> pd.DataFrame:
    df = df[df[TARGET_COLUMN] >= price_floor]
    if "odometer" in df.columns:
        df = df[df["odometer"] >= odometer_floor]
    df = df.dropna(subset=["year"])
    return df.copy()


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition_encoded"] = out["condition"].map(CONDITION_MAP).fillna(0).astype(int)
    out["cylinders_encoded"] = out["cylinders"].map(CYLINDERS_MAP).fillna(0).astype(int)
    out["title_status_encoded"] = (
        out["title_status"].str.lower().isin(CLEAN_TITLE_VALUES).astype(int)
    )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["car_age"] = REFERENCE_YEAR - out["year"].astype(int)
    # Guard: avoid division by zero for brand-new cars
    out["car_age"] = out["car_age"].clip(lower=1)
    out["mileage_per_year"] = out["odometer"] / out["car_age"]
    return out


def apply_mean_target_encoding(
    df: pd.DataFrame,
    encoding_tables: dict[str, pd.Series],
    global_mean: float,
) -> pd.DataFrame:
    """
    encoding_tables: {col_name: Series(index=category, values=log_mean_price)}
    global_mean: fallback for unseen categories
    """
    out = df.copy()
    for col in MEAN_TARGET_ENCODE_COLS:
        table = encoding_tables.get(col)
        if table is None:
            out[f"{col}_encoded"] = global_mean
        else:
            out[f"{col}_encoded"] = out[col].map(table).fillna(global_mean)
    return out


def compute_encoding_tables(df: pd.DataFrame) -> tuple[dict[str, pd.Series], float]:
    """Compute mean-target encoding tables from labelled training data."""
    log_price = np.log1p(df[TARGET_COLUMN])
    global_mean = float(log_price.mean())
    tables: dict[str, pd.Series] = {}
    for col in MEAN_TARGET_ENCODE_COLS:
        if col in df.columns:
            tables[col] = df.groupby(col).apply(
                lambda g: np.log1p(g[TARGET_COLUMN]).mean()
            )
    return tables, global_mean


def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LOG_TARGET_COLUMN] = np.log1p(out[TARGET_COLUMN])
    return out


def full_clean_pipeline(
    df: pd.DataFrame,
    encoding_tables: dict[str, pd.Series] | None = None,
    global_mean: float = 0.0,
    price_floor: float = 1000,
    odometer_floor: float = 500,
) -> pd.DataFrame:
    """
    End-to-end cleaning for a raw vehicles DataFrame.
    Pass encoding_tables=None when building training data (tables computed internally).
    Pass encoding_tables=<dict> when processing inference or new data against frozen tables.
    """
    df = drop_uninformative(df)
    df = fill_categoricals(df)
    df = remove_sparse_rows(df)
    df = filter_outliers(df, price_floor, odometer_floor)
    df = encode_ordinals(df)
    df = engineer_features(df)

    if encoding_tables is None:
        encoding_tables, global_mean = compute_encoding_tables(df)

    df = apply_mean_target_encoding(df, encoding_tables, global_mean)
    df = log_transform_target(df)
    return df
