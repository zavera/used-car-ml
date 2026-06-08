# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
APScheduler-based orchestration.

Two jobs:
  1. drift_check_job  — runs on configurable cron (default every 6h)
     Ingests new data → computes drift vs reference → if triggered, runs retrain
  2. retrain_job      — runs on configurable cron (default Sundays 2am)
     Full retrain regardless of drift (keeps model fresh on new patterns)

run_pipeline_once() is the shared logic invoked by both jobs and the /pipeline/run API.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config_loader import load_config
from data.ingestion import CsvDataSource, ingest
from data.preprocessing import compute_encoding_tables
from drift.detector import detect
from drift.report import save_report
from feature_store.store import FeatureStore
from training.trainer import run_training

logger = logging.getLogger(__name__)


def run_pipeline_once(force_retrain: bool = False) -> None:
    cfg = load_config()

    store = FeatureStore(
        store_path=cfg["feature_store"]["store_path"],
        reference_path=cfg["feature_store"]["reference_path"],
    )

    # ── 1. Ingest new data ──────────────────────────────────────────────────
    source = CsvDataSource(
        raw_path=cfg["data"]["raw_path"],
        filename_pattern=cfg["data"]["filename_pattern"],
    )
    try:
        new_df = ingest(source)
    except FileNotFoundError as exc:
        logger.warning("Ingestion skipped: %s", exc)
        return

    store.append(new_df)

    # ── 2. Drift check ──────────────────────────────────────────────────────
    min_new = cfg["drift"]["min_new_rows"]
    if store.new_row_count() < min_new and not force_retrain:
        logger.info("Not enough new rows (%d < %d) for drift check", store.new_row_count(), min_new)
        return

    try:
        reference = store.load_reference()
        current = store.load_current()
    except FileNotFoundError:
        logger.info("No reference snapshot — triggering initial training")
        _retrain(cfg, store, new_df)
        return

    report = detect(
        reference=reference,
        current=current,
        psi_threshold=cfg["drift"]["psi_threshold"],
        ks_pvalue_threshold=cfg["drift"]["ks_pvalue_threshold"],
        chi2_pvalue_threshold=cfg["drift"]["chi2_pvalue_threshold"],
    )
    save_report(report, cfg["drift"]["report_path"])
    logger.info(report.summary())

    if report.triggered or force_retrain:
        logger.info("Drift detected or force_retrain=True — initiating retrain")
        _retrain(cfg, store, current)
    else:
        logger.info("No significant drift detected — model unchanged")


def _retrain(cfg: dict, store: FeatureStore, df) -> None:
    from serving.api import predictor  # late import to avoid circular deps

    encoding_tables, global_mean = compute_encoding_tables(df)

    store.snapshot_as_reference()

    run_training(
        df=df,
        encoding_tables=encoding_tables,
        global_mean=global_mean,
        config=cfg,
        mlflow_tracking_uri=cfg["model"]["mlflow_tracking_uri"],
        experiment_name=cfg["model"]["experiment_name"],
        champion=cfg["model"]["champion"],
        registry_path=cfg["model"]["registry_path"],
    )

    # Hot-reload the serving layer with the new model
    try:
        predictor.reload()
        logger.info("Serving layer reloaded with new champion")
    except Exception as exc:
        logger.warning("Could not reload predictor in-process: %s", exc)


def start_scheduler() -> BackgroundScheduler:
    cfg = load_config()
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=run_pipeline_once,
        trigger=CronTrigger.from_crontab(cfg["scheduler"]["drift_check_cron"]),
        id="drift_check",
        name="Drift check + conditional retrain",
        replace_existing=True,
    )

    scheduler.add_job(
        func=lambda: run_pipeline_once(force_retrain=True),
        trigger=CronTrigger.from_crontab(cfg["scheduler"]["retrain_cron"]),
        id="full_retrain",
        name="Weekly full retrain",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started. drift_check='%s'  retrain='%s'",
        cfg["scheduler"]["drift_check_cron"],
        cfg["scheduler"]["retrain_cron"],
    )
    return scheduler
