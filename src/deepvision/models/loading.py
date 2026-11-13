"""Rebuild a model from a checkpoint written by our training code."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from deepvision.models.factory import build_model


def load_model_from_checkpoint(path: Union[str, Path]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Return (model in eval mode on CPU, checkpoint 'extra' metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    extra = dict(ckpt.get("extra", {}))
    kwargs = dict(extra.get("model_kwargs", {}))
    if "pretrained" in kwargs:
        kwargs["pretrained"] = False          # weights come from the checkpoint; no download needed

    class_names = extra.get("class_names")
    if not class_names:
        from deepvision.data.dataset import CLASS_NAMES
        class_names = CLASS_NAMES
    extra["class_names"] = list(class_names)

    model = build_model(extra.get("model_name", "simple_cnn"), num_classes=len(class_names), **kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, extra
