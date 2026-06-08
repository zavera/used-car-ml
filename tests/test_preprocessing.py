# Copyright (c) 2026 Callisto Tech — see LICENSE

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import (
    encode_ordinals,
    engineer_features,
    filter_outliers,
    fill_categoricals,
    remove_sparse_rows,
    log_transform_target,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "price": [15000, 8000, 500, 25000],
        "year": [2018, 2015, 2020, 2010],
        "odometer": [50000, 80000, 200, 120000],
        "condition": ["good", None, "excellent", "fair"],
        "cylinders": ["4 cylinders", "6 cylinders", None, "8 cylinders"],
        "title_status": ["clean", "rebuilt", "clean", "lien"],
        "manufacturer": ["toyota", "ford", "honda", "chevrolet"],
        "fuel": ["gas", "gas", "electric", "gas"],
        "drive": ["fwd", "4wd", "fwd", "rwd"],
        "size": ["mid-size", "full-size", "compact", "full-size"],
        "type": ["sedan", "pickup", "sedan", "truck"],
        "paint_color": ["white", "black", "silver", "blue"],
    })


def test_fill_categoricals_replaces_none_with_unknown():
    df = fill_categoricals(_base_df())
    assert (df["condition"] == "Unknown").sum() == 1
    assert (df["cylinders"] == "Unknown").sum() == 1


def test_filter_outliers_removes_low_price():
    df = fill_categoricals(_base_df())
    filtered = filter_outliers(df, price_floor=1000, odometer_floor=500)
    assert (filtered["price"] < 1000).sum() == 0


def test_filter_outliers_removes_low_odometer():
    df = fill_categoricals(_base_df())
    filtered = filter_outliers(df, price_floor=1000, odometer_floor=500)
    assert (filtered["odometer"] < 500).sum() == 0


def test_encode_ordinals_clean_title():
    df = fill_categoricals(_base_df())
    df = filter_outliers(df)
    encoded = encode_ordinals(df)
    assert encoded.loc[encoded["title_status"] == "clean", "title_status_encoded"].iloc[0] == 1
    assert encoded.loc[encoded["title_status"] == "rebuilt", "title_status_encoded"].iloc[0] == 0


def test_encode_ordinals_condition_order():
    df = fill_categoricals(_base_df())
    df = filter_outliers(df)
    encoded = encode_ordinals(df)
    good_val = encoded.loc[encoded["condition"] == "good", "condition_encoded"].iloc[0]
    fair_val = encoded.loc[encoded["condition"] == "fair", "condition_encoded"].iloc[0]
    assert good_val > fair_val


def test_engineer_features_car_age():
    df = fill_categoricals(_base_df())
    df = filter_outliers(df)
    df = encode_ordinals(df)
    out = engineer_features(df)
    assert "car_age" in out.columns
    assert (out["car_age"] >= 1).all()


def test_engineer_features_mileage_per_year():
    df = fill_categoricals(_base_df())
    df = filter_outliers(df)
    df = encode_ordinals(df)
    out = engineer_features(df)
    assert "mileage_per_year" in out.columns
    assert (out["mileage_per_year"] >= 0).all()


def test_log_transform_target_invertible():
    df = fill_categoricals(_base_df())
    df = filter_outliers(df)
    out = log_transform_target(df)
    reconstructed = np.expm1(out["log_price"])
    np.testing.assert_allclose(reconstructed.values, df["price"].values, rtol=1e-5)
