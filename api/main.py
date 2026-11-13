"""DeepVision AI inference API.

Run locally:  uvicorn api.main:app --reload     (interactive docs at /docs)
"""

import io
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from api.dependencies import Settings, get_predictor, get_settings, load_settings
from api.schemas import ClassesResponse, HealthResponse, ModelInfoResponse, PredictionResponse
from deepvision.inference.predictor import Predictor
from deepvision.utils.logger import get_logger

logger = get_logger("deepvision.api")
Image.MAX_IMAGE_PIXELS = 25_000_000          # reject decompression bombs
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.predictor = None
    app.state.model_info = {}
    if settings.model_path.is_file():
        try:
            app.state.predictor = Predictor(settings.model_path, settings.device, settings.conf_threshold)
            logger.info("Loaded model %s (%s)", settings.model_path, app.state.predictor.architecture)
        except Exception:
            logger.exception("Failed to load model from %s", settings.model_path)
    else:
        logger.warning("Model file not found at %s; /predict will return 503", settings.model_path)
    info_file = settings.model_path.with_name("model_info.json")
    if info_file.is_file():
        app.state.model_info = json.loads(info_file.read_text())
    yield


app = FastAPI(
    title="DeepVision AI",
    version="1.0.0",
    description="CIFAR-10 image classification service (PyTorch). Upload an image to /predict.",
    lifespan=lifespan,
)


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> RGB-ready PIL image, with explicit errors for bad input."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return ImageOps.exif_transpose(image)
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image has too many pixels.")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Liveness: the process is up (always 200)."""
    return HealthResponse(status="ok", model_loaded=request.app.state.predictor is not None)


@app.get("/ready", tags=["ops"])
def ready(predictor: Predictor = Depends(get_predictor)) -> dict:
    """Readiness: 200 only when the model is loaded."""
    return {"status": "ready"}


@app.get("/classes", response_model=ClassesResponse, tags=["model"])
def classes(predictor: Predictor = Depends(get_predictor)) -> ClassesResponse:
    return ClassesResponse(classes=predictor.class_names)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request, predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    meta = request.app.state.model_info
    return ModelInfoResponse(
        architecture=predictor.architecture, image_size=predictor.image_size,
        num_classes=len(predictor.class_names), class_names=predictor.class_names,
        device=str(predictor.device), confidence_threshold=predictor.threshold,
        training_run=meta.get("run_name"), validation_accuracy=meta.get("validation_accuracy"),
        git_commit=meta.get("git_commit"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    file: UploadFile = File(..., description="JPEG, PNG or WebP image"),
    top_k: int = Query(3, ge=1, le=10),
    predictor: Predictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type {file.content_type!r}. "
                                                    f"Use one of {sorted(ALLOWED_CONTENT_TYPES)}.")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    data = file.file.read(max_bytes + 1)                 # never read more than the limit
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    result = predictor.predict(decode_image(data), top_k=top_k)
    logger.info("predicted=%s confidence=%.3f low_conf=%s latency_ms=%.1f bytes=%d",   # no filenames/content logged
                result["predicted_class"], result["confidence"], result["low_confidence"],
                result["inference_ms"], len(data))
    return PredictionResponse(architecture=predictor.architecture, **result)
