# Inference benchmark: p6_improved_adamw_cos

Machine: {'processor': 'AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD', 'cpu_count': 12, 'torch_threads': 6, 'cuda_device': None, 'torch': '2.15.0+cpu'}

Latency is model-only (no image decoding, no HTTP). Accuracy is on the validation split.

| variant | size_MB | val_acc | batch | p50_ms | p95_ms | img/s |
|---|---|---|---|---|---|---|
| torch_fp32_cpu | 2.369 | 0.8014 | 1 | 0.618 | 0.767 | 1563.7 |
| torch_fp32_cpu | 2.369 | 0.8014 | 32 | 6.938 | 7.811 | 4616.9 |
| onnx_fp32_cpu | 2.369 | 0.8014 | 1 | 0.122 | 0.178 | 7662.5 |
| onnx_fp32_cpu | 2.369 | 0.8014 | 32 | 2.187 | 2.878 | 13894.2 |
