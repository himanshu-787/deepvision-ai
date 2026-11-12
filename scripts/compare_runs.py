"""Table of all training runs found in artifacts/metrics. Usage: python scripts/compare_runs.py"""

import json
from pathlib import Path

import pandas as pd

from deepvision.utils.markdown import to_markdown


def main() -> None:
    rows = []
    for f in sorted(Path("artifacts/metrics").glob("*.json")):
        d = json.loads(f.read_text())
        if "history" not in d:          # skip evaluation files (*_val.json, *_test.json)
            continue
        h, cfg = d["history"], d.get("config") or d.get("args", {})
        i = max(range(len(h["val_acc"])), key=lambda k: h["val_acc"][k])
        rows.append({
            "run": d["run_name"],
            "model": cfg.get("model", cfg.get("strategy", "simple_cnn")),
            "opt": cfg.get("optimizer", "adam"),
            "lr": cfg.get("lr", "-"),
            "wd": cfg.get("weight_decay", 0.0),
            "sched": cfg.get("scheduler", "none"),
            "params": d["total_parameters"],
            "best_val_acc": round(h["val_acc"][i], 4),
            "best_epoch": i + 1,
            "train_acc@best": round(h["train_acc"][i], 4),
            "gap": round(h["train_acc"][i] - h["val_acc"][i], 4),
            "seconds": round(sum(h["epoch_seconds"])),
        })
    if not rows:
        raise SystemExit("No training runs found in artifacts/metrics.")
    df = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
    print(df.to_string(index=False))
    out = Path("reports/experiments/run_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Training run comparison (validation set)\n\n" + to_markdown(df) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
