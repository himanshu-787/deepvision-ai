"""Evaluate a checkpoint on the validation or test split and write a full report.

    python scripts/evaluate.py --checkpoint artifacts/checkpoints/cnn_baseline_best.pt --split val
    python scripts/evaluate.py --checkpoint artifacts/checkpoints/cnn_baseline_best.pt --split test

Test evaluation refuses to overwrite an existing test report unless --force is given.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES
from deepvision.evaluation.metrics import (
    collect_predictions, compute_metrics, expected_calibration_error, top_confusions,
)
from deepvision.evaluation.visualize import (
    plot_confusion_matrix, plot_misclassified, plot_per_class_f1, plot_reliability,
)
from deepvision.models.factory import build_model
from deepvision.utils.device import get_device
from deepvision.utils.model_info import count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--force", action="store_true", help="allow overwriting an existing TEST report")
    return p.parse_args()


def raw_images(dataset, indices: np.ndarray) -> np.ndarray:
    """Original uint8 images (N,32,32,3) for display, for a CIFAR10 dataset or a Subset of one."""
    if isinstance(dataset, Subset):
        return dataset.dataset.data[[dataset.indices[i] for i in indices]]
    return dataset.data[indices]


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    run_name = ckpt_path.stem.removesuffix("_best")
    out_json = Path("artifacts/metrics") / f"{run_name}_{args.split}.json"
    if args.split == "test" and out_json.exists() and not args.force:
        raise SystemExit(f"{out_json} already exists. The test set is for ONE final report; "
                         "re-evaluating on it to make decisions is test-set tuning. Use --force only to repeat "
                         "the same final evaluation.")

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    extra = ckpt.get("extra", {})
    model_name = extra.get("model_name", "simple_cnn")
    model_kwargs = dict(extra.get("model_kwargs", {}))
    if "pretrained" in model_kwargs:
        model_kwargs["pretrained"] = False            # weights come from the checkpoint, no download needed
    image_size, stats = extra.get("image_size", 32), extra.get("stats", "cifar")

    model = build_model(model_name, num_classes=len(CLASS_NAMES), **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, seed=extra.get("seed", 42),   # same seed => same validation images as training
        augment=False, image_size=image_size, stats=stats,
    )
    loader = val_loader if args.split == "val" else test_loader

    y_true, y_pred, y_prob, loss = collect_predictions(model, loader, device)
    summary, report, cm = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    ece, bin_conf, bin_acc, _ = expected_calibration_error(y_true, y_pred, y_prob)
    confusions = top_confusions(cm, CLASS_NAMES)
    total, _ = count_parameters(model)
    size_mb = sum(v.numel() * v.element_size() for v in model.state_dict().values()) / 1024**2

    result = {
        "run_name": run_name, "split": args.split, "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"), "num_samples": int(len(y_true)), "loss": loss,
        **summary, "total_parameters": total, "model_size_mb": round(size_mb, 3),
        "top_confusions": confusions, "per_class": {c: report[c] for c in CLASS_NAMES},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    plots = Path("artifacts/plots")
    tag = f"{run_name}_{args.split}"
    plot_confusion_matrix(cm, CLASS_NAMES, plots / f"{tag}_confusion.png", title=f"{tag} (row-normalized)")
    plot_per_class_f1(report, CLASS_NAMES, plots / f"{tag}_f1.png", title=f"{tag}: F1 per class")
    plot_reliability(bin_conf, bin_acc, ece, plots / f"{tag}_reliability.png", title=tag)

    wrong = np.where(y_pred != y_true)[0]                 # most confident mistakes first
    conf = y_prob.max(axis=1)
    worst = wrong[np.argsort(-conf[wrong])][:16]
    if len(worst):
        plot_misclassified(raw_images(loader.dataset, worst), y_true[worst], y_pred[worst],
                           conf[worst], CLASS_NAMES, plots / f"{tag}_errors.png")

    print(f"\n{run_name} on {args.split} ({len(y_true)} images, checkpoint epoch {ckpt.get('epoch')})")
    print(f"  loss {loss:.4f} | accuracy {summary['accuracy']:.4f} | macro P/R/F1 "
          f"{summary['precision_macro']:.4f}/{summary['recall_macro']:.4f}/{summary['f1_macro']:.4f}")
    print(f"  ROC-AUC (OvR) {summary['roc_auc_macro_ovr']:.4f} | ECE {ece:.4f} | "
          f"mean confidence {summary['mean_confidence']:.4f} | size {size_mb:.2f} MB")
    print("  weakest classes by F1:",
          sorted(((c, round(report[c]["f1-score"], 3)) for c in CLASS_NAMES), key=lambda t: t[1])[:3])
    print("  top confusions (true -> predicted, count):", confusions[:3])
    print(f"saved {out_json} and plots in {plots}")


if __name__ == "__main__":
    main()
