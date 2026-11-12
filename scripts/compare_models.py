"""Final comparison table + chart for chosen runs.

    python scripts/compare_models.py --runs <model_A_run> <model_B_run> <model_C_run>
Requires training summaries and *_val.json / *_test.json evaluation reports.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deepvision.utils.markdown import to_markdown

M = Path("artifacts/metrics")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run names in order: A B C ...")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        train = json.loads((M / f"{run}.json").read_text())
        val = json.loads((M / f"{run}_val.json").read_text())
        test = json.loads((M / f"{run}_test.json").read_text())
        rows.append({
            "run": run,
            "params": train["total_parameters"],
            "trainable (end)": train["trainable_parameters"],
            "train_s": round(train["total_train_seconds"]),
            "train_loss": round(train["train_loss_at_best"], 4),
            "train_acc": round(train["train_acc_at_best"], 4),
            "val_loss": round(val["loss"], 4),
            "val_acc": round(val["accuracy"], 4),
            "test_loss": round(test["loss"], 4),
            "test_acc": round(test["accuracy"], 4),
            "test_P_macro": round(test["precision_macro"], 4),
            "test_R_macro": round(test["recall_macro"], 4),
            "test_F1_macro": round(test["f1_macro"], 4),
            "ECE": round(test["ece"], 4),
            "size_MB": test["model_size_mb"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    chosen = df.loc[df["val_acc"].idxmax(), "run"]
    note = (f"Selection rule: highest VALIDATION accuracy -> **{chosen}**. "
            "Test numbers are reported for transparency and were not used to choose.")
    print("\n" + note)

    out = Path("reports/experiments/model_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Model comparison\n\n" + to_markdown(df) + "\n\n" + note + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["test_acc"], width=0.4, label="test accuracy")
    ax.bar([i + 0.2 for i in x], df["test_F1_macro"], width=0.4, label="test macro-F1")
    ax.set_xticks(list(x)); ax.set_xticklabels(df["run"], rotation=15)
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Model comparison (test set)")
    plt.tight_layout()
    Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("artifacts/plots/model_comparison.png", dpi=120)
    print(f"saved {out} and artifacts/plots/model_comparison.png")


if __name__ == "__main__":
    main()
