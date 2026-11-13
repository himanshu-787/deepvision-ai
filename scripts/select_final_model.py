"""Choose the final model by VALIDATION accuracy and stage it for deployment.

    python scripts/select_final_model.py --runs <A> <B> <C>
    python scripts/select_final_model.py --runs <A> <B> <C> --override <run> --reason "within noise of best, 3x faster"
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from deepvision.models.loading import load_model_from_checkpoint
from deepvision.tracking.tracker import get_git_commit

M = Path("artifacts/metrics")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--override", default=None, help="choose a different run than the best-val one")
    p.add_argument("--reason", default=None, help="required with --override")
    args = p.parse_args()

    scores = {}
    for run in args.runs:
        f = M / f"{run}_val.json"
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run scripts/evaluate.py --split val for {run} first.")
        scores[run] = json.loads(f.read_text())["accuracy"]

    ranking = sorted(scores.items(), key=lambda kv: -kv[1])
    print("Validation accuracy ranking:")
    for run, acc in ranking:
        print(f"  {run:30s} {acc:.4f}")

    chosen, reason = ranking[0][0], "highest validation accuracy"
    if args.override:
        if args.override not in scores or not args.reason:
            raise SystemExit("--override must name one of --runs and needs --reason.")
        chosen, reason = args.override, args.reason

    ckpt = Path("artifacts/checkpoints") / f"{chosen}_best.pt"
    model, extra = load_model_from_checkpoint(ckpt)        # fails loudly if the checkpoint is unusable
    dest = Path("artifacts/final")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, dest / "model.pt")
    info = {
        "run_name": chosen, "selection_reason": reason, "validation_accuracy": scores[chosen],
        "architecture": extra.get("model_name"), "image_size": extra.get("image_size"),
        "config_hash": extra.get("config_hash"), "git_commit": get_git_commit(),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nFinal model: {chosen} ({reason}) -> {dest / 'model.pt'}")


if __name__ == "__main__":
    main()
