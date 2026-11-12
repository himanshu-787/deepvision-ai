"""Config-driven experiment runner: scratch / head_only / finetune, with tracking."""

from pathlib import Path
from typing import Any, Dict

import torch.nn as nn
import yaml

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES
from deepvision.models.factory import build_model
from deepvision.tracking.tracker import Tracker, get_git_commit
from deepvision.training.finetune import two_stage_finetune
from deepvision.training.optimizer import build_optimizer
from deepvision.training.reporting import save_run_summary
from deepvision.training.scheduler import build_scheduler
from deepvision.training.trainer import fit
from deepvision.utils.config import config_hash
from deepvision.utils.device import get_device
from deepvision.utils.logger import get_logger
from deepvision.utils.seed import set_seed


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_name, seed, strategy = cfg["run_name"], cfg["seed"], cfg["strategy"]
    d, m, t, ft, tr = cfg["data"], cfg["model"], cfg["train"], cfg["finetune"], cfg["tracking"]

    log = get_logger("deepvision", Path("logs") / f"{run_name}.log")
    set_seed(seed)
    device = get_device()
    cfg_id = config_hash(cfg)
    log.info("run=%s strategy=%s device=%s config_hash=%s", run_name, strategy, device, cfg_id)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=d["batch_size"], num_workers=d["num_workers"], seed=seed,
        augment=d["augment"], image_size=d["image_size"], stats=d["stats"],
    )
    model = build_model(m["name"], num_classes=len(CLASS_NAMES), **(m.get("kwargs") or {})).to(device)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = Path("artifacts/checkpoints") / f"{run_name}_best.pt"
    extra = {
        "model_name": m["name"], "model_kwargs": m.get("kwargs") or {}, "image_size": d["image_size"],
        "stats": d["stats"], "class_names": CLASS_NAMES, "seed": seed,
        "config_hash": cfg_id, "strategy": strategy,
    }

    with Tracker(tr["experiment_name"], run_name, tr["mlflow"], tr["tensorboard"]) as tracker:
        tracker.log_params(cfg)
        tracker.set_tags({"git_commit": get_git_commit(), "config_hash": cfg_id,
                          "device": str(device), "strategy": strategy})
        callback = lambda epoch, metrics: tracker.log_metrics(metrics, step=epoch)
        common = dict(amp=t["amp"], grad_clip=t["grad_clip"], accum_steps=t["accum_steps"],
                      patience=t["patience"], extra=extra, epoch_callback=callback)

        if strategy == "scratch":
            optimizer = build_optimizer(t["optimizer"], model.parameters(), t["lr"], t["weight_decay"], t["momentum"])
            scheduler = build_scheduler(t["scheduler"], optimizer, t["epochs"])
            history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                          t["epochs"], ckpt_path, scheduler=scheduler, **common)
        elif strategy == "head_only":
            model.set_trainable(["fc"])
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = build_optimizer(t["optimizer"], params, t["lr"], t["weight_decay"], t["momentum"])
            scheduler = build_scheduler(t["scheduler"], optimizer, t["epochs"])
            history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                          t["epochs"], ckpt_path, scheduler=scheduler, **common)
        elif strategy == "finetune":
            history = two_stage_finetune(
                model, train_loader, val_loader, criterion, device,
                stages=ft["stages"], head_epochs=ft["head_epochs"], ft_epochs=ft["ft_epochs"],
                head_lr=ft["head_lr"], backbone_lr=ft["backbone_lr"], lr_decay=ft["lr_decay"],
                weight_decay=ft["weight_decay"], checkpoint_path=ckpt_path, patience=t["patience"],
                amp=t["amp"], grad_clip=t["grad_clip"], accum_steps=t["accum_steps"],
                extra=extra, epoch_callback=callback,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        summary = save_run_summary(run_name, model, history, device, config=cfg,
                                   extra_fields={"strategy": strategy, "config_hash": cfg_id})
        cfg_path = Path("artifacts/metrics") / f"{run_name}_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        tracker.log_summary({
            "best_val_acc": summary["best_val_acc"], "best_epoch": summary["best_epoch"],
            "total_train_seconds": summary["total_train_seconds"],
            "total_parameters": summary["total_parameters"],
        })
        tracker.log_artifacts([ckpt_path, cfg_path, Path("artifacts/metrics") / f"{run_name}.json",
                               Path("artifacts/plots") / f"{run_name}_curves.png"])
    log.info("done: best val acc %.4f at epoch %d", summary["best_val_acc"], summary["best_epoch"])
    return summary
