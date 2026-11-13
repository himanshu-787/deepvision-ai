FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch (much smaller than the CUDA build). Cached as its own layer.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Install our package (non-editable) and the API code
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY api ./api

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=/models/model.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
