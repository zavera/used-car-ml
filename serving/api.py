# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
FastAPI serving layer.

Endpoints:
  POST /predict          — price prediction from vehicle attributes
  GET  /health           — model loaded + last drift check status
  GET  /drift/latest     — last drift report
  POST /pipeline/run     — manually trigger ingest + drift check + retrain if needed
  POST /feedback         — save verbatim user comment
  GET  /feedback/all     — return all raw feedback entries
  GET  /feedback/narrative — Groq-generated narrative from feedback history
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from drift.report import load_latest_report
from serving.predictor import Predictor

logger = logging.getLogger(__name__)
app = FastAPI(title="Callisto Tech — Used Car Price API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
predictor = Predictor()


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


class VehicleInput(BaseModel):
    year: int = Field(..., ge=1900, le=2026, example=2018)
    manufacturer: str = Field(..., example="toyota")
    condition: Literal["new", "like new", "excellent", "good", "fair", "salvage"] = "good"
    cylinders: Literal[
        "3 cylinders", "4 cylinders", "5 cylinders", "6 cylinders",
        "8 cylinders", "10 cylinders", "12 cylinders", "other"
    ] = "4 cylinders"
    fuel: Literal["gas", "diesel", "hybrid", "electric", "other"] = "gas"
    odometer: float = Field(..., ge=0, example=55000)
    title_status: Literal["clean", "rebuilt", "lien", "missing", "parts only", "salvage"] = "clean"
    transmission: Literal["automatic", "manual", "other"] = "automatic"
    drive: Literal["4wd", "fwd", "rwd"] = "fwd"
    size: Literal["full-size", "mid-size", "compact", "sub-compact"] = "mid-size"
    type: Literal[
        "sedan", "SUV", "pickup", "truck", "coupe", "hatchback",
        "wagon", "van", "convertible", "mini-van", "offroad", "bus", "other"
    ] = "sedan"
    paint_color: str = Field(default="white", example="silver")


class PredictionResponse(BaseModel):
    predicted_price: float
    model: str


@app.post("/predict", response_model=PredictionResponse)
def predict(vehicle: VehicleInput):
    try:
        result = predictor.predict(vehicle.model_dump())
        return PredictionResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.get("/health")
def health():
    loaded = predictor._artifact is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": predictor._artifact["model_name"] if loaded else None,
    }


@app.get("/drift/latest")
def drift_latest():
    from config_loader import load_config
    cfg = load_config()
    report = load_latest_report(cfg["drift"]["report_path"])
    if report is None:
        return {"message": "No drift reports yet"}
    return report


@app.post("/pipeline/run")
def pipeline_run(force: bool = False):
    """Manually trigger the full pipeline (ingest → drift → retrain if needed).
    Pass ?force=true to retrain regardless of drift."""
    from orchestration.scheduler import run_pipeline_once
    import threading
    threading.Thread(target=lambda: run_pipeline_once(force_retrain=force), daemon=True).start()
    return {"message": f"Pipeline triggered in background (force_retrain={force})"}


@app.post("/feedback")
def submit_feedback(body: dict):
    comment = (body.get("comment") or "").strip()
    if not comment:
        raise HTTPException(status_code=400, detail="comment is required")
    if len(comment) > 2000:
        raise HTTPException(status_code=400, detail="comment exceeds 2000 characters")
    from feedback.store import save
    entry = save(comment, context=body.get("context"))
    return {"saved": True, "ts": entry["ts"]}


@app.get("/feedback/all")
def get_feedback():
    from feedback.store import load_all
    return {"entries": load_all()}


@app.get("/feedback/narrative")
def get_narrative():
    from feedback.store import load_all
    from feedback.narrative import generate
    entries = load_all()
    if not entries:
        return {"narrative": None, "count": 0}
    try:
        narrative = generate()
        return {"narrative": narrative, "count": len(entries)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Narrative generation failed")
        raise HTTPException(status_code=500, detail="Narrative generation failed")


@app.post("/market/compare")
def market_compare(vehicle: VehicleInput):
    """
    Run the model prediction then compare against market sources (training dataset + eBay).
    If |model_price − market_median| > price_drift_threshold, retrain is triggered in background.
    """
    try:
        result = predictor.predict(vehicle.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    model_price = result["predicted_price"]

    from market.aggregator import get_aggregator
    aggregator  = get_aggregator()
    comparison  = aggregator.compare(vehicle.model_dump(), model_price)

    if comparison.market_drift_triggered:
        logger.info(
            "Market price drift: model=$%.0f market_median=$%.0f delta=$%.0f — triggering retrain",
            model_price, comparison.market_median, comparison.delta,
        )
        import threading
        from orchestration.scheduler import run_pipeline_once
        threading.Thread(target=lambda: run_pipeline_once(force_retrain=True), daemon=True).start()

    return {
        "model_prediction":         round(model_price, 2),
        "model_name":               result["model"],
        "sources": [
            {
                "name":      e.source,
                "price":     round(e.price, 2) if e.available else None,
                "count":     e.count,
                "available": e.available,
                "note":      e.note,
            }
            for e in comparison.sources
        ],
        "market_median":            round(comparison.market_median, 2) if comparison.market_median is not None else None,
        "delta":                    round(comparison.delta, 2) if comparison.delta is not None else None,
        "market_drift_triggered":   comparison.market_drift_triggered,
        "drift_threshold":          comparison.drift_threshold,
    }


@app.post("/model/reload")
def model_reload():
    predictor.reload()
    return {"message": "Model reloaded", "loaded": predictor._artifact is not None}
