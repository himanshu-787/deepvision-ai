"""Settings (from environment variables) and shared dependencies."""

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from deepvision.inference.predictor import Predictor


@dataclass(frozen=True)
class Settings:
    model_path: Path
    max_upload_mb: float
    conf_threshold: float
    device: str


def load_settings() -> Settings:
    """Read configuration from the environment (re-read at every app start)."""
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "artifacts/final/model.pt")),
        max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "5")),
        conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.5")),
        device=os.getenv("DEVICE", "cpu"),
    )


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH and the server logs.")
    return predictor
