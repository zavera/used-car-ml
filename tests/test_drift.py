# Copyright (c) 2026 Callisto Tech — see LICENSE

import numpy as np
import pandas as pd
import pytest

from drift.detector import detect, _psi


def _make_df(car_age_mean: float, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "car_age": rng.normal(car_age_mean, 2, n).clip(1, 25),
        "mileage_per_year": rng.normal(12000, 3000, n).clip(0),
        "cylinders_encoded": rng.choice([4, 6, 8], n),
        "fuel": rng.choice(["gas", "gas", "gas", "electric", "diesel"], n),
        "drive": rng.choice(["fwd", "4wd", "rwd"], n),
        "type": rng.choice(["sedan", "SUV", "pickup", "truck"], n),
        "condition": rng.choice(["good", "excellent", "fair"], n),
        "title_status": rng.choice(["clean", "rebuilt"], n, p=[0.9, 0.1]),
    })


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(42)
    data = rng.normal(7, 2, 2000)
    psi = _psi(data[:1000], data[1000:])
    assert psi < 0.1, f"Expected PSI < 0.10 for identical distribution, got {psi:.4f}"


def test_psi_shifted_distribution_exceeds_threshold():
    rng = np.random.default_rng(42)
    ref = rng.normal(7, 2, 1000)
    cur = rng.normal(14, 2, 1000)  # strongly shifted
    psi = _psi(ref, cur)
    assert psi > 0.2, f"Expected PSI > 0.20 for shifted distribution, got {psi:.4f}"


def test_detect_no_drift_identical_data():
    ref = _make_df(car_age_mean=7, seed=1)
    cur = _make_df(car_age_mean=7, seed=2)
    report = detect(ref, cur, psi_threshold=0.2, ks_pvalue_threshold=0.05)
    assert not report.triggered


def test_detect_drift_triggered_on_large_shift():
    ref = _make_df(car_age_mean=5, n=2000, seed=1)
    cur = _make_df(car_age_mean=15, n=2000, seed=2)  # older cars now
    report = detect(ref, cur, psi_threshold=0.2, ks_pvalue_threshold=0.05)
    assert report.triggered
    assert "car_age" in report.drifted_features()


def test_drift_report_summary_contains_feature_names():
    ref = _make_df(7, seed=10)
    cur = _make_df(7, seed=11)
    report = detect(ref, cur)
    summary = report.summary()
    assert "car_age" in summary
    assert "fuel" in summary
