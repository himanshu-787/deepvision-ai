"""Export all runs of an MLflow experiment to a CSV (best validation accuracy first)."""

import argparse
from pathlib import Path

import mlflow

from deepvision.tracking.tracker import MLFLOW_URI


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="deepvision")
    args = p.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    df = mlflow.search_runs(experiment_names=[args.experiment], order_by=["metrics.best_val_acc DESC"])
    wanted = ["tags.mlflow.runName", "tags.strategy", "tags.git_commit", "metrics.best_val_acc",
              "metrics.best_epoch", "metrics.total_train_seconds", "params.model.name",
              "params.train.optimizer", "params.train.lr", "params.data.image_size"]
    df = df[[c for c in wanted if c in df.columns]]
    out = Path("reports/experiments/mlflow_runs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
