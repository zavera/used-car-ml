# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
FastAPI serving layer.

Endpoints:
  POST /predict          — price prediction from vehicle attributes
  GET  /health           — model loaded + last drift check status
  GET  /drift/latest     — last drift report
  POST /pipeline/run     — manually trigger ingest + drift check + retrain if needed
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drift.report import load_latest_report
from serving.predictor import Predictor

logger = logging.getLogger(__name__)
app = FastAPI(title="Callisto Tech — Used Car Price API", version="1.0.0")
predictor = Predictor()


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
def pipeline_run():
    """Manually trigger the full pipeline (ingest → drift → retrain if needed)."""
    from orchestration.scheduler import run_pipeline_once
    import threading
    threading.Thread(target=run_pipeline_once, daemon=True).start()
    return {"message": "Pipeline triggered in background"}


@app.post("/model/reload")
def model_reload():
    predictor.reload()
    return {"message": "Model reloaded", "loaded": predictor._artifact is not None}
