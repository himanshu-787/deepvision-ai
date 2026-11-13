# Inference benchmark: r18_head

Machine: {'processor': 'AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD', 'cpu_count': 12, 'torch_threads': 6, 'cuda_device': None, 'torch': '2.15.0+cpu'}

Latency is model-only (no image decoding, no HTTP). Accuracy is on the validation split.

| variant | size_MB | val_acc | batch | p50_ms | p95_ms | img/s |
|---|---|---|---|---|---|---|
| torch_fp32_cpu | 42.691 | 0.7338 | 1 | 8.393 | 9.176 | 118.2 |
| torch_fp32_cpu | 42.691 | 0.7338 | 32 | 63.652 | 81.024 | 493.1 |
| onnx_fp32_cpu | 42.647 | 0.7338 | 1 | 2.506 | 3.138 | 379.9 |
| onnx_fp32_cpu | 42.647 | 0.7338 | 32 | 32.663 | 43.523 | 930.1 |
