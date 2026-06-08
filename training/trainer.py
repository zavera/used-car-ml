# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Model training with MLflow experiment tracking.

Two models from the original study:
  - Ridge regression  (baseline, interpretable)
  - Polynomial degree-3 + SequentialFeatureSelector (champion by MSE)

The winning model is serialized to models/champion.pkl and registered in MLflow.
Encoding tables and scaler are co-serialized so inference is self-contained.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from feature_store.definitions import FEATURE_COLUMNS, LOG_TARGET_COLUMN

logger = logging.getLogger(__name__)

MODEL_VERSION_FILE = "models/version.txt"


@dataclass
class TrainingArtifacts:
    model: Any
    scaler: StandardScaler
    encoding_tables: dict
    global_mean: float
    selected_features: list[str]
    train_mse: float
    test_mse: float
    model_name: str
    run_id: str


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def _split(df: pd.DataFrame, test_size: float, random_state: int):
    features = _available_features(df)
    X = df[features].values
    y = df[LOG_TARGET_COLUMN].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state), features


def train_ridge(
    df: pd.DataFrame,
    alpha_grid: list[float],
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[Pipeline, list[str], float, float]:
    (X_train, X_test, y_train, y_test), features = _split(df, test_size, random_state)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    param_grid = {"ridge__alpha": alpha_grid}
    gs = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    train_mse = float(np.mean((best.predict(X_train) - y_train) ** 2))
    test_mse = float(np.mean((best.predict(X_test) - y_test) ** 2))
    logger.info("Ridge best alpha=%s  train_mse=%.4f  test_mse=%.4f",
                gs.best_params_["ridge__alpha"], train_mse, test_mse)
    return best, features, train_mse, test_mse


def train_poly(
    df: pd.DataFrame,
    degree: int = 3,
    n_features_to_select: int = 4,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[Pipeline, list[str], float, float]:
    (X_train, X_test, y_train, y_test), features = _split(df, test_size, random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_p = poly.fit_transform(X_train_s)
    X_test_p = poly.transform(X_test_s)

    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=n_features_to_select, cv=3)
    sfs.fit(X_train_p, y_train)

    X_train_sel = sfs.transform(X_train_p)
    X_test_sel = sfs.transform(X_test_p)

    lr.fit(X_train_sel, y_train)
    train_mse = float(np.mean((lr.predict(X_train_sel) - y_train) ** 2))
    test_mse = float(np.mean((lr.predict(X_test_sel) - y_test) ** 2))
    logger.info("Poly degree=%d n_sel=%d  train_mse=%.4f  test_mse=%.4f",
                degree, n_features_to_select, train_mse, test_mse)

    # Bundle into a dict so inference can replay the same transform chain
    bundle = {"scaler": scaler, "poly": poly, "sfs": sfs, "lr": lr}
    return bundle, features, train_mse, test_mse


def run_training(
    df: pd.DataFrame,
    encoding_tables: dict,
    global_mean: float,
    config: dict,
    mlflow_tracking_uri: str,
    experiment_name: str,
    champion: str = "poly",
    registry_path: str = "models",
) -> TrainingArtifacts:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    tc = config["training"]

    with mlflow.start_run() as run:
        # Ridge
        ridge_model, features, ridge_train_mse, ridge_test_mse = train_ridge(
            df,
            alpha_grid=tc["ridge"]["alpha_grid"],
            test_size=tc["test_size"],
            random_state=tc["random_state"],
            cv_folds=tc["cv_folds"],
        )
        mlflow.log_metrics({"ridge_train_mse": ridge_train_mse, "ridge_test_mse": ridge_test_mse})

        # Poly
        poly_bundle, _, poly_train_mse, poly_test_mse = train_poly(
            df,
            degree=tc["poly"]["degree"],
            n_features_to_select=tc["poly"]["n_features_to_select"],
            test_size=tc["test_size"],
            random_state=tc["random_state"],
        )
        mlflow.log_metrics({"poly_train_mse": poly_train_mse, "poly_test_mse": poly_test_mse})
        mlflow.log_param("champion", champion)

        chosen_model = poly_bundle if champion == "poly" else ridge_model
        chosen_train_mse = poly_train_mse if champion == "poly" else ridge_train_mse
        chosen_test_mse = poly_test_mse if champion == "poly" else ridge_test_mse

        # Persist champion
        Path(registry_path).mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": chosen_model,
            "model_name": champion,
            "encoding_tables": encoding_tables,
            "global_mean": global_mean,
            "features": features,
        }
        artifact_path = Path(registry_path) / "champion.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)

        # Track version
        Path(MODEL_VERSION_FILE).write_text(run.info.run_id)
        mlflow.log_artifact(str(artifact_path))

        logger.info("Champion '%s' saved. run_id=%s", champion, run.info.run_id)

    return TrainingArtifacts(
        model=chosen_model,
        scaler=poly_bundle["scaler"] if champion == "poly" else ridge_model.named_steps["scaler"],
        encoding_tables=encoding_tables,
        global_mean=global_mean,
        selected_features=features,
        train_mse=chosen_train_mse,
        test_mse=chosen_test_mse,
        model_name=champion,
        run_id=run.info.run_id,
    )
