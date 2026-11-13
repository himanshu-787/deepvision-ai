"""Export to ONNX, quantize, and benchmark one trained model.

    python scripts/optimize_model.py --checkpoint artifacts/checkpoints/<run>_best.pt
Accuracy is measured on the VALIDATION split only. Run it for each of A, B and C
(use --skip-quantize for runs where you don't need int8).
"""

import argparse
import copy
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from deepvision.data.dataloader import get_dataloaders
from deepvision.models.loading import load_model_from_checkpoint
from deepvision.optimization.benchmark import summarize, time_callable
from deepvision.optimization.export import export_onnx, quantize_onnx_dynamic, verify_onnx
from deepvision.utils.markdown import to_markdown


@dataclass
class Variant:
    name: str
    hardware: str
    size_mb: float
    predict: Callable[[torch.Tensor], np.ndarray]          # CPU float tensor -> logits
    make_fn: Callable[[int], Callable[[], object]]         # batch size -> zero-arg callable to time
    sync: Optional[Callable[[], None]] = None


def state_dict_mb(model: torch.nn.Module) -> float:
    return sum(v.numel() * v.element_size() for v in model.state_dict().values()) / 1024**2


def torch_variant(model, device: torch.device, dtype: torch.dtype, name: str, image_size: int) -> Variant:
    m = copy.deepcopy(model).to(device=device, dtype=dtype).eval()

    @torch.inference_mode()
    def predict(x: torch.Tensor) -> np.ndarray:
        return m(x.to(device=device, dtype=dtype)).float().cpu().numpy()

    def make_fn(bs: int):
        x = torch.randn(bs, 3, image_size, image_size, device=device, dtype=dtype)

        def fn():
            with torch.inference_mode():
                m(x)
        return fn

    sync = torch.cuda.synchronize if device.type == "cuda" else None
    return Variant(name, device.type, round(state_dict_mb(m), 3), predict, make_fn, sync)


def ort_variant(path: Path, name: str, image_size: int, threads: int) -> Variant:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads                     # same CPU threads as PyTorch, for fairness
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(x: torch.Tensor) -> np.ndarray:
        return session.run(None, {input_name: x.numpy()})[0]

    def make_fn(bs: int):
        arr = np.random.randn(bs, 3, image_size, image_size).astype(np.float32)
        return lambda: session.run(None, {input_name: arr})

    return Variant(name, "cpu", round(os.path.getsize(path) / 1024**2, 3), predict, make_fn)


def accuracy(variant: Variant, loader, max_batches: Optional[int]) -> float:
    correct = seen = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        correct += int((variant.predict(x).argmax(axis=1) == y.numpy()).sum())
        seen += len(y)
    return correct / seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark and optimize a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-val-batches", type=int, default=None, help="limit accuracy checks (for quick tests)")
    p.add_argument("--skip-quantize", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    run = ckpt_path.stem.removesuffix("_best")
    model, extra = load_model_from_checkpoint(ckpt_path)
    image_size, stats = int(extra.get("image_size", 32)), extra.get("stats", "cifar")
    threads = torch.get_num_threads()
    print(f"run={run} image_size={image_size} cpu_threads={threads} cuda={torch.cuda.is_available()}")

    _, val_loader, _ = get_dataloaders(
        batch_size=args.eval_batch_size, seed=extra.get("seed", 42), augment=False,
        image_size=image_size, stats=stats,
    )

    # ---- build variants ----
    variants = [torch_variant(model, torch.device("cpu"), torch.float32, "torch_fp32_cpu", image_size)]
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        variants.append(torch_variant(model, cuda, torch.float32, "torch_fp32_cuda", image_size))
        variants.append(torch_variant(model, cuda, torch.float16, "torch_fp16_cuda", image_size))

    onnx_fp32 = Path("artifacts/onnx") / f"{run}.onnx"
    export_onnx(model, image_size, onnx_fp32)
    sample, _ = next(iter(val_loader))
    onnx_check = verify_onnx(model, onnx_fp32, sample[:16])
    print(f"ONNX vs PyTorch on a real batch: {onnx_check}")
    variants.append(ort_variant(onnx_fp32, "onnx_fp32_cpu", image_size, threads))

    if not args.skip_quantize:
        try:
            onnx_int8 = quantize_onnx_dynamic(onnx_fp32, Path("artifacts/onnx") / f"{run}_int8.onnx")
            variants.append(ort_variant(onnx_int8, "onnx_int8_cpu", image_size, threads))
        except Exception as err:   # report honestly instead of hiding a failed optimization
            print(f"!! int8 quantization/loading failed and is skipped: {type(err).__name__}: {err}")

    # ---- measure ----
    results = []
    for v in variants:
        print(f"measuring {v.name} ...")
        acc = accuracy(v, val_loader, args.max_val_batches)
        bench = []
        for bs in args.batch_sizes:
            iters = args.iters if bs <= 8 else max(10, args.iters // 4)
            if v.hardware == "cuda":
                torch.cuda.reset_peak_memory_stats()
            stats_row = summarize(time_callable(v.make_fn(bs), args.warmup, iters, v.sync), bs)
            if v.hardware == "cuda":
                stats_row["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            bench.append(stats_row)
        results.append({"name": v.name, "hardware": v.hardware, "size_mb": v.size_mb,
                        "val_accuracy": round(acc, 4), "bench": bench})

    report = {
        "run_name": run, "image_size": image_size, "split_for_accuracy": "val",
        "machine": {"processor": platform.processor(), "cpu_count": os.cpu_count(), "torch_threads": threads,
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__},
        "onnx_check": onnx_check, "variants": results,
    }
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    (Path("artifacts/metrics") / f"benchmark_{run}.json").write_text(json.dumps(report, indent=2))

    rows = []
    for r in results:
        for b in r["bench"]:
            rows.append({"variant": r["name"], "size_MB": r["size_mb"], "val_acc": r["val_accuracy"],
                         "batch": b["batch_size"], "p50_ms": b["p50_ms"], "p95_ms": b["p95_ms"],
                         "img/s": b["throughput_img_s"]})
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = Path("reports/experiments") / f"benchmark_{run}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Inference benchmark: {run}\n\nMachine: {report['machine']}\n\n"
                   "Latency is model-only (no image decoding, no HTTP). Accuracy is on the validation split.\n\n"
                   + to_markdown(df) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
