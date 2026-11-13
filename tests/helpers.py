"""Shared test helpers."""

import io

import numpy as np
from PIL import Image


def make_png_bytes(size=(64, 48), seed: int = 0) -> bytes:
    """A random-noise RGB PNG (incompressible, so its size is predictable)."""
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
