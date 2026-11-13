"""Model shapes, parameter counts, freezing and BatchNorm behaviour."""

import pytest
import torch

from deepvision.models.cnn import ImprovedCNN, SimpleCNN
from deepvision.models.factory import build_model
from deepvision.models.resnet import ResNet18Classifier
from deepvision.utils.model_info import count_parameters


def test_simple_cnn_shape_and_params():
    model = SimpleCNN()
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    assert count_parameters(model) == (620_362, 620_362)


def test_improved_cnn_shape_and_params():
    model = ImprovedCNN()
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    assert count_parameters(model) == (620_586, 620_586)


def test_eval_mode_is_deterministic_with_dropout():
    model = ImprovedCNN(dropout=0.5).eval()
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(model(x), model(x))


def test_resnet_head_and_freezing():
    model = ResNet18Classifier(num_classes=10, pretrained=False)
    assert model(torch.randn(4, 3, 64, 64)).shape == (4, 10)
    model.set_trainable(["fc"])
    assert count_parameters(model)[1] == 512 * 10 + 10


def test_frozen_batchnorm_stays_in_eval_mode():
    model = ResNet18Classifier(pretrained=False)
    model.set_trainable(["fc"])
    model.train()
    assert model.backbone.layer4[0].bn1.training is False
    assert model.backbone.fc.training is True
    model.set_trainable(["fc", "layer4"])
    model.train()
    assert model.backbone.layer4[0].bn1.training is True
    assert model.backbone.layer1[0].bn1.training is False


def test_invalid_stage_and_model_name():
    with pytest.raises(ValueError):
        ResNet18Classifier(pretrained=False).stage_parameters("layer9")
    with pytest.raises(ValueError):
        build_model("does_not_exist")
