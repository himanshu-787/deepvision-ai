"""ONNX export, verification and dynamic quantization."""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn


def export_onnx(model: nn.Module, image_size: int, out_path: Union[str, Path], opset: int = 17) -> Path:
    """Export to ONNX with a dynamic batch dimension. Input (B,3,S,S) -> output logits (B,C)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    try:   # newer PyTorch versions default to a different exporter; ask for the classic one
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:   # older PyTorch has no `dynamo` argument
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    return out_path


def verify_onnx(model: nn.Module, onnx_path: Union[str, Path], sample: torch.Tensor) -> Dict[str, float]:
    """Check the file is valid and compare its outputs with PyTorch on a real batch."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.no_grad():
        ref = model.eval().cpu()(sample).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    return {
        "max_abs_diff": float(np.abs(ref - out).max()),
        "argmax_agreement": float((ref.argmax(1) == out.argmax(1)).mean()),
    }


def quantize_onnx_dynamic(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """int8 weight quantization with ONNX Runtime. May be unsupported for some conv models."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return Path(dst)
