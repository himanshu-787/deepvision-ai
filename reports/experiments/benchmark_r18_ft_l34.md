# Inference benchmark: r18_ft_l34

Machine: {'processor': 'AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD', 'cpu_count': 12, 'torch_threads': 6, 'cuda_device': None, 'torch': '2.15.0+cpu'}

Latency is model-only (no image decoding, no HTTP). Accuracy is on the validation split.

| variant | size_MB | val_acc | batch | p50_ms | p95_ms | img/s |
|---|---|---|---|---|---|---|
| torch_fp32_cpu | 42.691 | 0.8812 | 1 | 8.671 | 10.199 | 113.1 |
| torch_fp32_cpu | 42.691 | 0.8812 | 32 | 58.341 | 62.836 | 549.1 |
| onnx_fp32_cpu | 42.647 | 0.8812 | 1 | 2.456 | 2.978 | 404.8 |
| onnx_fp32_cpu | 42.647 | 0.8812 | 32 | 30.622 | 33.904 | 1031.1 |
| onnx_int8_cpu | 10.719 | 0.8798 | 1 | 3.839 | 4.331 | 262.9 |
| onnx_int8_cpu | 10.719 | 0.8798 | 32 | 91.664 | 96.525 | 354.4 |
