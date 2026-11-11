"""Device selection."""

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return 'cuda' if available (and not overridden), otherwise 'cpu'."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
