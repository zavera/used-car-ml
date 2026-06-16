# Callisto Tech — Claude Code Project Instructions

## Ownership
This codebase is the exclusive property of Callisto Tech, authored by Ambreen Zaver.
All code, model architecture, pipelines, and components are fully proprietary.

## Commits
- Author: Ambreen Zaver <zaver.ambreen@gmail.com>
- NEVER add `Co-Authored-By: Claude` or any AI attribution to commit messages.
- Commit messages should reflect Callisto Tech authorship only.

---

## Project: Used Car ML Platform

A production-grade ML system for used car price prediction. It ingests vehicle listing data,
trains regression models, serves predictions via FastAPI, and continuously monitors for
data drift and market price drift — retraining automatically when either is detected.

### Target Market
This product is sold to **Manheim auction users** — dealers and bidders who need fast, data-backed price guidance at wholesale vehicle auctions. Predictions should help users decide bid ceilings quickly. The market comparison (Training Dataset, eBay, Enterprise Car Sales) gives them a cross-reference against retail prices so they understand margin. Keep UX fast and focused — auction users are making decisions under time pressure.

### Core Philosophy
- **Self-healing model**: the system retrains itself when it detects drift; humans don't manage model freshness manually
- **Market-anchored pricing**: model predictions are always cross-checked against real market data (eBay + training dataset); if they diverge by >$1,000 the model retrains
- **No manual intervention**: drift checks, ingestion, and retraining all run on a cron schedule or trigger automatically from the API
- **MLflow tracks everything**: every training run is logged; `models/champion.pkl` is always the production artifact

### Architecture

```
data/raw/vehicles_YYYYMMDD.csv
        │
        ▼
data/ingestion.py ──► feature_store/store.py (parquet)
        │
        ▼
drift/detector.py  (PSI + KS + chi²)
        │ drift triggered?
        ▼
training/trainer.py ──► models/champion.pkl  ──► serving/predictor.py
                                                        │
                                              POST /predict
                                              POST /market/compare ──► market/aggregator.py
                                                                              │
                                                                    EbayAdapter + DatasetAdapter
                                                                              │
                                                                    |delta| > $1,000 → retrain
```

### Key Components

| Module | Purpose |
|---|---|
| `serving/api.py` | FastAPI app; endpoints: `/predict`, `/market/compare`, `/pipeline/run`, `/drift/latest`, `/feedback/*`, `/health` |
| `serving/predictor.py` | Loads `models/champion.pkl`; hot-reloads after retrain |
| `training/trainer.py` | Trains Ridge (baseline) and Poly-degree-3+SFS (champion); logs to MLflow |
| `drift/detector.py` | PSI for categoricals, KS for continuous; triggers retrain when thresholds exceeded |
| `market/aggregator.py` | Compares model price vs market median; triggers retrain if delta > threshold |
| `market/adapters/ebay.py` | eBay Finding API (findCompletedItems); auto-routes SBX App IDs to sandbox endpoint |
| `market/adapters/dataset.py` | Uses training dataset as a market price reference |
| `orchestration/scheduler.py` | APScheduler: drift check every 6h, forced retrain Sundays 2am |
| `feedback/store.py` + `narrative.py` | Collects user comments; Groq generates a narrative summary |
| `config_loader.py` | Loads `config/config.yaml`; overrides `ebay_app_id` from `EBAY_APP_ID` env var |

### Models
- **Champion**: Polynomial degree-3 + SequentialFeatureSelector (4 features) — best MSE
- **Baseline**: Ridge regression with GridSearchCV alpha tuning
- Both are trained and logged each run; `config.yaml → model.champion` controls which is served
- Target is `log(price)`; predictor exponentiates back to dollars

### Retrain Triggers (any one is sufficient)
1. **Feature drift**: PSI > 0.2 on categoricals, KS p-value < 0.05 on continuous
2. **Market drift**: `|model_price - market_median| > $1,000` (configurable in `config.yaml`)
3. **Scheduled**: Sundays 2am (force retrain regardless of drift)
4. **Manual**: `POST /pipeline/run?force=true`

### eBay Integration
- Credentials live in `.env` (never committed); loaded by `python-dotenv` at startup
- App ID `Callisto-callisto-SBX-*` = sandbox; adapter auto-detects and routes to sandbox endpoint
- Sandbox returns no listings (expected) — use production keys for live data
- When production keys are ready: update `EBAY_APP_ID` in `.env`, no code changes needed

### Environment
- Python virtualenv at `.venv/` — always use `.venv/bin/python3` / `.venv/bin/pip`
- Config: `config/config.yaml` (thresholds, cron schedules, paths)
- Secrets: `.env` (eBay keys, any API keys) — already in `.gitignore`
- MLflow UI: `sqlite:///mlruns/mlflow.db`

### Conventions
- All new market data sources implement `market/adapters/base.py → PricingAdapter`
- Drift reports are JSON, saved to `data/processed/drift_reports/`
- Feature definitions (column names, log target) live in `feature_store/definitions.py` — single source of truth
- Copyright header on every file: `# Copyright (c) 2026 Callisto Tech — see LICENSE`
