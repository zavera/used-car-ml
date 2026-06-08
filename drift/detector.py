# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Drift detection engine.

Three tests used depending on feature type:
  - PSI  (Population Stability Index) — continuous features, industry standard for model monitoring
  - KS   (Kolmogorov-Smirnov)        — continuous features, distribution shape change
  - Chi² (chi-squared)               — categorical features, frequency distribution shift

PSI thresholds (industry convention):
  < 0.10  → stable
  0.10–0.20 → moderate drift, watch
  > 0.20  → major drift, retrain triggered
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from feature_store.definitions import (
    CATEGORICAL_DRIFT_FEATURES,
    CONTINUOUS_DRIFT_FEATURES,
)

logger = logging.getLogger(__name__)

PSI_BINS = 10


@dataclass
class FeatureDriftResult:
    feature: str
    test: str
    statistic: float
    pvalue: float | None
    drifted: bool
    detail: str = ""


@dataclass
class DriftReport:
    triggered: bool
    results: list[FeatureDriftResult] = field(default_factory=list)

    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.results if r.drifted]

    def summary(self) -> str:
        lines = [f"Drift triggered: {self.triggered}"]
        for r in self.results:
            flag = "DRIFT" if r.drifted else "OK"
            lines.append(f"  [{flag}] {r.feature} ({r.test}): stat={r.statistic:.4f}")
        return "\n".join(lines)


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two continuous distributions."""
    combined = np.concatenate([reference, current])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)
    breakpoints[0] -= 1e-9
    breakpoints[-1] += 1e-9

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = (ref_counts / len(reference)).clip(min=1e-6)
    cur_pct = (cur_counts / len(current)).clip(min=1e-6)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def _chi2_test(
    reference: pd.Series, current: pd.Series
) -> tuple[float, float]:
    all_cats = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts().reindex(all_cats, fill_value=0)
    cur_counts = current.value_counts().reindex(all_cats, fill_value=0)

    # Scale current to same total as reference to avoid size bias
    scale = len(reference) / max(len(current), 1)
    cur_scaled = (cur_counts * scale).round().astype(int).clip(lower=0)

    result = stats.chi2_contingency(
        pd.DataFrame({"ref": ref_counts, "cur": cur_scaled})
    )
    return float(result.statistic), float(result.pvalue)


def detect(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
    chi2_pvalue_threshold: float = 0.05,
) -> DriftReport:
    """
    Compare current feature distribution against the reference (training) snapshot.
    Returns a DriftReport — triggered=True means retrain should be initiated.
    """
    results: list[FeatureDriftResult] = []
    any_drift = False

    for feat in CONTINUOUS_DRIFT_FEATURES:
        if feat not in reference.columns or feat not in current.columns:
            logger.warning("Drift feature %s not found in data — skipping", feat)
            continue

        ref_vals = reference[feat].dropna().to_numpy()
        cur_vals = current[feat].dropna().to_numpy()

        psi_val = _psi(ref_vals, cur_vals)
        psi_drifted = psi_val > psi_threshold

        ks_stat, ks_pvalue = _ks_test(ref_vals, cur_vals)
        ks_drifted = ks_pvalue < ks_pvalue_threshold

        drifted = psi_drifted or ks_drifted
        if drifted:
            any_drift = True

        results.append(FeatureDriftResult(
            feature=feat,
            test="PSI+KS",
            statistic=psi_val,
            pvalue=ks_pvalue,
            drifted=drifted,
            detail=f"PSI={psi_val:.4f} KS_p={ks_pvalue:.4f}",
        ))

    for feat in CATEGORICAL_DRIFT_FEATURES:
        if feat not in reference.columns or feat not in current.columns:
            logger.warning("Drift feature %s not found in data — skipping", feat)
            continue

        try:
            chi2_stat, chi2_pvalue = _chi2_test(reference[feat], current[feat])
        except Exception as exc:
            logger.warning("Chi² failed for %s: %s", feat, exc)
            continue

        drifted = chi2_pvalue < chi2_pvalue_threshold
        if drifted:
            any_drift = True

        results.append(FeatureDriftResult(
            feature=feat,
            test="chi2",
            statistic=chi2_stat,
            pvalue=chi2_pvalue,
            drifted=drifted,
            detail=f"chi2={chi2_stat:.4f} p={chi2_pvalue:.4f}",
        ))

    return DriftReport(triggered=any_drift, results=results)
