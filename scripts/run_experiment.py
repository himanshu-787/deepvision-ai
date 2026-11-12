"""Run any experiment from a YAML config.

    python scripts/run_experiment.py --config configs/cnn.yaml
    python scripts/run_experiment.py --config configs/cnn.yaml --set train.epochs=1 --set run_name=smoke_cnn
"""

import argparse

from deepvision.training.pipeline import run_experiment
from deepvision.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Config-driven experiment runner")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="override a config value, e.g. train.lr=0.0003 (repeatable)")
    args = p.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
