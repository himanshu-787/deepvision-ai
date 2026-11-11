"""Phase 2.1 - tensors: creation, shapes, ops, devices, NumPy bridge."""

import numpy as np
import torch

# Creation
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros = torch.zeros(2, 3)
rand = torch.randn(2, 3)            # normal distribution
print("dtype:", a.dtype, "| shape:", a.shape, "| ndim:", a.ndim)

# Math (element-wise vs matrix)
print("elementwise a*a:\n", a * a)
print("matmul a@a:\n", a @ a)

# Indexing & slicing
print("first row:", a[0], "| last column:", a[:, -1])

# Reshaping
t = torch.arange(12)                # tensor([0..11])
print("reshape (3,4):\n", t.reshape(3, 4))
print("reshape (2,-1) shape:", t.reshape(2, -1).shape)

# Broadcasting
print("broadcast:\n", torch.ones(2, 3) + torch.tensor([10.0, 20.0, 30.0]))

# Image batches use (Batch, Channels, Height, Width) = "NCHW"
batch = torch.randn(8, 3, 32, 32)
print("image batch shape (B,C,H,W):", batch.shape)
# A single image from a NumPy/PIL array is (H, W, C) -> permute to (C, H, W)
hwc = torch.randn(32, 32, 3)
print("permuted:", hwc.permute(2, 0, 1).shape)

# NumPy bridge (CPU tensors share memory with the array!)
arr = np.array([1.0, 2.0, 3.0])
tn = torch.from_numpy(arr)
arr[0] = 99.0
print("shared memory effect:", tn)   # tensor([99., 2., 3.])

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_device = torch.randn(3, 3).to(device)
print("tensor device:", on_device.device)

# Common bug demo: dtype mismatch
try:
    torch.tensor([1, 2, 3]) @ torch.tensor([1.0, 2.0, 3.0])
except RuntimeError as err:
    print("dtype error caught:", str(err)[:80])
