"""API integration tests (tiny model, no network, no GPU)."""

from fastapi.testclient import TestClient

from deepvision.data.dataset import CLASS_NAMES
from tests.helpers import make_png_bytes


def post_image(client, content=None, ctype="image/png", params=None, name="x.png"):
    return client.post("/predict", params=params,
                       files={"file": (name, content if content is not None else make_png_bytes(), ctype)})


def test_health_ready_and_classes(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/ready").status_code == 200
    assert client.get("/classes").json()["classes"] == CLASS_NAMES


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["architecture"] == "simple_cnn" and body["num_classes"] == 10


def test_predict_success(client):
    r = post_image(client)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CLASS_NAMES and len(body["top_k"]) == 3
    assert 0.0 <= body["confidence"] <= 1.0


def test_top_k_parameter(client):
    assert len(post_image(client, params={"top_k": 5}).json()["top_k"]) == 5
    assert post_image(client, params={"top_k": 0}).status_code == 422
    assert post_image(client, params={"top_k": 11}).status_code == 422


def test_rejects_wrong_content_type(client):
    assert post_image(client, b"hello", "text/plain", name="a.txt").status_code == 415


def test_rejects_corrupt_image(client):
    assert post_image(client, b"this is not an image").status_code == 400


def test_rejects_empty_file(client):
    assert post_image(client, b"").status_code == 400


def test_missing_file_field_is_422(client):
    assert client.post("/predict").status_code == 422


def test_rejects_oversized_upload(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_checkpoint))
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")          # about 1 KB
    from api.main import app
    with TestClient(app) as c:
        assert post_image(c).status_code == 413


def test_service_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert c.get("/ready").status_code == 503
        assert post_image(c).status_code == 503
