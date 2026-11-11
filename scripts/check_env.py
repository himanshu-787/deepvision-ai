"""Environment check for DeepVision AI.

Run from the project root:
    python scripts/check_env.py
"""

import platform
import sys
import time

import torch
import torchvision


def select_device() -> torch.device:
    """Return the GPU if PyTorch can use one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    print("=" * 55)
    print("DEEPVISION AI - ENVIRONMENT CHECK")
    print("=" * 55)

    # 1. Software versions
    print(f"OS:                 {platform.platform()}")
    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"Torchvision:        {torchvision.__version__}")

    # 2. Device information
    device = select_device()
    print(f"\nCUDA available:     {torch.cuda.is_available()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU name:           {props.name}")
        print(f"GPU memory (GB):    {props.total_memory / 1024**3:.1f}")
        print(f"CUDA build version: {torch.version.cuda}")
    else:
        print("Running on CPU (no usable NVIDIA GPU detected).")
    print(f"Selected device:    {device}")

    # 3. Basic tensor operation
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    print(f"\nTensor test: {x.tolist()} * 2 = {y.tolist()}")
    assert y.tolist() == [2.0, 4.0, 6.0], "Basic tensor math failed!"

    # 4. A real computation on the selected device
    size = 2048
    a = torch.randn(size, size, device=device)  # a 2048 x 2048 random matrix
    b = torch.randn(size, size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU runs asynchronously; wait before timing
    start = time.perf_counter()
    c = a @ b  # matrix multiplication
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Matrix multiply ({size}x{size}) on {device}: {elapsed_ms:.1f} ms")
    print(f"Result tensor shape: {tuple(c.shape)}")

    print("\nEnvironment verification completed successfully!")


if __name__ == "__main__":
    main()