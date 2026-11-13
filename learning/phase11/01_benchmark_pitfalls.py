"""Phase 11 - benchmarking pitfalls and what quantization does to numbers."""

import time

import torch

from deepvision.models.cnn import SimpleCNN

model = SimpleCNN().eval()

# ---- 1. Warm-up: the first calls are not representative ----
x = torch.randn(1, 3, 32, 32)
times = []
with torch.inference_mode():
    for _ in range(30):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
print("first 3 calls (ms):", [round(t, 3) for t in times[:3]])
print("median of last 20 (ms):", round(sorted(times[-20:])[10], 3))

# ---- 2. Batch size trades latency for throughput ----
for bs in (1, 8, 32, 128):
    xb = torch.randn(bs, 3, 32, 32)
    with torch.inference_mode():
        model(xb)                                   # warm-up
        t0 = time.perf_counter()
        for _ in range(10):
            model(xb)
        ms = (time.perf_counter() - t0) * 1000 / 10
    print(f"batch {bs:3d}: {ms:8.2f} ms/batch | {bs / (ms / 1000):9.0f} img/s")

# ---- 3. GPU timing without synchronize is wrong (only if you have CUDA) ----
if torch.cuda.is_available():
    m, xg = model.cuda(), torch.randn(128, 3, 32, 32, device="cuda")
    with torch.inference_mode():
        m(xg); torch.cuda.synchronize()
        t0 = time.perf_counter(); m(xg)
        wrong = (time.perf_counter() - t0) * 1000       # measures only the launch
        torch.cuda.synchronize()
        t0 = time.perf_counter(); m(xg); torch.cuda.synchronize()
        right = (time.perf_counter() - t0) * 1000
    print(f"GPU without sync: {wrong:.3f} ms (misleading) | with sync: {right:.3f} ms")

# ---- 4. Quantization by hand: float32 -> uint8 -> back ----
w = torch.randn(1000) * 0.5
scale = ((w.max() - w.min()) / 255).item()
zero_point = round((-w.min() / scale).item())
q = torch.clamp(torch.round(w / scale) + zero_point, 0, 255)      # integers 0..255
deq = (q - zero_point) * scale
print(f"\nquantization: scale={scale:.5f} zero_point={zero_point}")
print(f"max rounding error {(w - deq).abs().max().item():.5f} (bounded by scale/2 = {scale / 2:.5f})")
print(f"storage: {w.numel() * 4} bytes -> {w.numel()} bytes (4x smaller)")
