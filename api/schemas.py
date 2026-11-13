"""Request/response schemas (Pydantic validates and documents them)."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = Field(description="True when confidence is below the configured threshold")
    top_k: List[ClassScore]
    inference_ms: float
    architecture: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClassesResponse(BaseModel):
    classes: List[str]


class ModelInfoResponse(BaseModel):
    architecture: str
    image_size: int
    num_classes: int
    class_names: List[str]
    device: str
    confidence_threshold: float
    training_run: Optional[str] = None
    validation_accuracy: Optional[float] = None
    git_commit: Optional[str] = None
