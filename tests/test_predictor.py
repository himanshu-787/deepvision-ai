"""Predictor behaviour, including train/serve preprocessing parity."""

import numpy as np
import pytest
import torch
from PIL import Image

from deepvision.data.transforms import get_eval_transform
from deepvision.inference.predictor import Predictor


def random_image(size=(100, 80), mode="RGB"):
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).convert(mode)


def test_predict_output_structure(tiny_checkpoint):
    result = Predictor(tiny_checkpoint).predict(random_image(), top_k=3)
    assert len(result["top_k"]) == 3
    assert result["predicted_class"] == result["top_k"][0]["label"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert sum(item["probability"] for item in result["top_k"]) <= 1.0 + 1e-5


def test_probabilities_sum_to_one_when_all_classes_requested(tiny_checkpoint):
    result = Predictor(tiny_checkpoint).predict(random_image(), top_k=10)
    assert sum(item["probability"] for item in result["top_k"]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_handles_other_image_modes(tiny_checkpoint, mode):
    assert "predicted_class" in Predictor(tiny_checkpoint).predict(random_image(mode=mode))


def test_prediction_is_deterministic(tiny_checkpoint):
    p, img = Predictor(tiny_checkpoint), random_image()
    assert p.predict(img)["top_k"] == p.predict(img)["top_k"]


def test_low_confidence_flag_follows_threshold(tiny_checkpoint):
    img = random_image()
    assert Predictor(tiny_checkpoint, confidence_threshold=0.0).predict(img)["low_confidence"] is False
    assert Predictor(tiny_checkpoint, confidence_threshold=1.01).predict(img)["low_confidence"] is True


def test_preprocessing_matches_training_pipeline(tiny_checkpoint):
    """Serving must apply exactly the same transform as evaluation did."""
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    expected = get_eval_transform(image_size=32, stats="cifar")(img)
    assert torch.allclose(Predictor(tiny_checkpoint).transform(img), expected)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "nope.pt")
