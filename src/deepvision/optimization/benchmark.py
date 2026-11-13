"""Latency and throughput measurement helpers."""

import time
from typing import Callable, Dict, List, Optional

import numpy as np


def time_callable(
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    sync: Optional[Callable[[], None]] = None,
) -> List[float]:
    """Run fn() `warmup` times (discarded), then time `iters` calls. Returns latencies in ms.

    `sync` must be torch.cuda.synchronize for GPU work, because GPU calls are asynchronous.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    latencies = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def summarize(latencies_ms: List[float], batch_size: int) -> Dict[str, float]:
    arr = np.asarray(latencies_ms)
    return {
        "batch_size": batch_size,
        "iters": int(len(arr)),
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "throughput_img_s": round(float(batch_size / (arr.mean() / 1000.0)), 1),
    }
