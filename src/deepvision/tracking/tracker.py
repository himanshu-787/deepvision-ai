"""One small wrapper around MLflow and TensorBoard."""

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# A local SQLite database; view it with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_URI = "sqlite:///mlflow.db"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """{'train': {'lr': 0.1}} -> {'train.lr': 0.1}"""
    out: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


class Tracker:
    """Use as a context manager so the run is always closed, even after an error."""

    def __init__(self, experiment_name: str, run_name: str, use_mlflow: bool = True, use_tensorboard: bool = True):
        self.mlflow = None
        self.writer = None
        if use_mlflow:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            self.mlflow = mlflow
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(Path("runs") / run_name))

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:480] for k, v in flatten(params).items()})

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self.mlflow:
            self.mlflow.set_tags(tags)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_summary(self, metrics: Dict[str, float]) -> None:
        if self.mlflow:
            self.mlflow.log_metrics(metrics)

    def log_artifacts(self, paths: Iterable[Path]) -> None:
        if self.mlflow:
            for path in paths:
                if Path(path).exists():
                    self.mlflow.log_artifact(str(path))

    def close(self, failed: bool = False) -> None:
        if self.writer:
            self.writer.close()
        if self.mlflow:
            self.mlflow.end_run(status="FAILED" if failed else "FINISHED")

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type: Optional[type], exc, tb) -> None:
        self.close(failed=exc_type is not None)
