"""Load a trained checkpoint and classify PIL images."""

import time
from pathlib import Path
from typing import Any, Dict, Union

import torch
from PIL import Image
from torchvision import transforms

from deepvision.data.transforms import get_eval_transform
from deepvision.models.loading import load_model_from_checkpoint


class Predictor:
    """Stateful inference wrapper: load once, predict many times.

    Preprocessing = resize to 32x32 (CIFAR-native size) followed by the SAME evaluation
    transform used in training (optional upscaling + normalization).
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu", confidence_threshold: float = 0.5):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        model, extra = load_model_from_checkpoint(path)
        self.model = model.to(self.device).eval()
        self.class_names = list(extra["class_names"])
        self.architecture = str(extra.get("model_name", "unknown"))
        self.image_size = int(extra.get("image_size", 32))
        self.stats = extra.get("stats", "cifar")
        self.threshold = float(confidence_threshold)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            get_eval_transform(image_size=self.image_size, stats=self.stats),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        start = time.perf_counter()
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)   # (1,3,S,S)
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu()            # (num_classes,)
        values, indices = probs.topk(min(top_k, len(self.class_names)))
        top = [{"label": self.class_names[i], "probability": float(v)}
               for v, i in zip(values.tolist(), indices.tolist())]
        return {
            "predicted_class": top[0]["label"],
            "confidence": top[0]["probability"],
            "low_confidence": top[0]["probability"] < self.threshold,
            "top_k": top,
            "inference_ms": (time.perf_counter() - start) * 1000.0,
        }
