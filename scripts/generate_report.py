"""Build final documentation from REAL experiment artifacts.

    python scripts/generate_report.py --runs <A> <B> <C> [--final <run>]

Writes reports/final_report.md, MODEL_CARD.md, docs/results_table.md and refreshes the README block
between <!-- RESULTS:START --> and <!-- RESULTS:END -->.
Sections needing YOUR judgement are marked TODO(you). Nothing here is estimated: missing data -> n/a.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from deepvision.tracking.tracker import flatten
from deepvision.utils.markdown import to_markdown

M = Path("artifacts/metrics")
LABELS = "ABCDEFGH"


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Complete the earlier step that creates it (see Phases 6-11).")
    return json.loads(path.read_text())


def model_name(cfg: Dict[str, Any]) -> str:
    m = cfg.get("model")
    return m["name"] if isinstance(m, dict) else (m or "simple_cnn")


def strategy_of(train: Dict[str, Any]) -> str:
    return train.get("strategy") or train["config"].get("strategy") or "scratch"


def image_size_of(cfg: Dict[str, Any]) -> Any:
    flat = flatten(cfg)
    return flat.get("data.image_size", flat.get("image_size", 32))


def latency_ms(run: str, batch_size: int = 1, variant: str = "torch_fp32_cpu") -> Optional[float]:
    f = M / f"benchmark_{run}.json"
    if not f.exists():
        return None
    for v in json.loads(f.read_text())["variants"]:
        if v["name"] == variant:
            for b in v["bench"]:
                if b["batch_size"] == batch_size:
                    return b["p50_ms"]
    return None


def benchmark_table(run: str) -> str:
    f = M / f"benchmark_{run}.json"
    if not f.exists():
        return "_Benchmark not run for this model (python scripts/optimize_model.py)._"
    rows = []
    for v in json.loads(f.read_text())["variants"]:
        for b in v["bench"]:
            rows.append({"variant": v["name"], "size_MB": v["size_mb"], "val_acc": v["val_accuracy"],
                         "batch": b["batch_size"], "p50_ms": b["p50_ms"], "p95_ms": b["p95_ms"],
                         "img/s": b["throughput_img_s"]})
    return to_markdown(pd.DataFrame(rows))


def update_readme(block: str) -> None:
    readme = Path("README.md")
    if not readme.exists():
        return
    pattern = re.compile(r"(<!-- RESULTS:START -->)(.*?)(<!-- RESULTS:END -->)", re.S)
    text = readme.read_text(encoding="utf-8")
    if not pattern.search(text):
        print("README markers not found; README results block not updated.")
        return
    readme.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", text), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run names in order: A B C")
    p.add_argument("--final", default=None, help="final run (default: from artifacts/final/model_info.json)")
    args = p.parse_args()

    info_path = Path("artifacts/final/model_info.json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    final = args.final or info.get("run_name")
    if final not in args.runs:
        raise SystemExit("--final (or model_info.json run_name) must be one of --runs.")

    data = {r: {"train": load(M / f"{r}.json"), "val": load(M / f"{r}_val.json"),
                "test": load(M / f"{r}_test.json")} for r in args.runs}

    rows = []
    for label, r in zip(LABELS, args.runs):
        t, v, te = data[r]["train"], data[r]["val"], data[r]["test"]
        rows.append({
            "model": label, "run": r, "arch": model_name(t["config"]), "strategy": strategy_of(t),
            "img_size": image_size_of(t["config"]), "params": t["total_parameters"],
            "trainable": t["trainable_parameters"], "train_s": round(t["total_train_seconds"]),
            "train_acc": round(t["train_acc_at_best"], 4), "val_acc": round(v["accuracy"], 4),
            "test_acc": round(te["accuracy"], 4), "test_P": round(te["precision_macro"], 4),
            "test_R": round(te["recall_macro"], 4), "test_F1": round(te["f1_macro"], 4),
            "ECE": round(te["ece"], 4), "size_MB": te["model_size_mb"], "cpu_ms_bs1": latency_ms(r),
        })
    df = pd.DataFrame(rows).fillna("n/a")
    table = to_markdown(df)
    selection = (f"Selection rule: highest **validation** accuracy (with any override documented in "
                 f"`artifacts/final/model_info.json`). Final model: **{final}**"
                 + (f" ({info['selection_reason']})." if info.get("selection_reason") else "."))
    note = ("Test metrics are from one evaluation per model and were not used for selection. "
            "`cpu_ms_bs1` = median single-image latency, PyTorch fp32 on this machine's CPU, model only.")
    results_block = f"{table}\n\n{selection}\n\n{note}"
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results_table.md").write_text(results_block + "\n", encoding="utf-8")
    update_readme(results_block)

    fin = data[final]
    per_class = fin["test"]["per_class"]
    ranked = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])
    worst = ", ".join(f"{c} (F1 {m['f1-score']:.3f})" for c, m in ranked[:3])
    best = ", ".join(f"{c} (F1 {m['f1-score']:.3f})" for c, m in ranked[-3:][::-1])
    confusions = "\n".join(f"- true **{a}** predicted as **{b}**: {n} images"
                           for a, b, n in fin["test"]["top_confusions"][:5])
    notes_file = Path("reports/experiments/error_analysis.md")
    your_notes = notes_file.read_text(encoding="utf-8") if notes_file.exists() else "TODO(you): write reports/experiments/error_analysis.md"
    log_file = Path("reports/experiments/experiment_log.md")
    exp_log = log_file.read_text(encoding="utf-8") if log_file.exists() else "TODO(you): write reports/experiments/experiment_log.md"
    devices = sorted({data[r]["train"]["device"] for r in args.runs})
    hp = "\n\n".join(f"**Model {l}: `{r}`**\n\n```yaml\n{yaml.safe_dump(data[r]['train']['config'], sort_keys=False)}```"
                     for l, r in zip(LABELS, args.runs))
    fin_row = rows[args.runs.index(final)]

    report = f"""# DeepVision AI: Final Technical Report

_Generated by `scripts/generate_report.py` from the artifacts in `artifacts/metrics/`. PyTorch {torch.__version__}._

## 1. Project overview
DeepVision AI trains and compares three image classifiers on CIFAR-10 with PyTorch: a CNN trained from scratch,
an ImageNet-pretrained ResNet18 with a frozen backbone (feature extraction), and a fine-tuned ResNet18. The best model is
selected on validation data, evaluated once on the test set, benchmarked, and served through a FastAPI service in Docker.

## 2. Problem statement
Classify 32x32 colour images into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) and
measure honestly how training strategy affects accuracy, cost and calibration.

## 3. Objectives
1. Implement real training loops (forward, loss, backpropagation, optimizer updates, validation, checkpointing).
2. Compare scratch training, feature extraction and fine-tuning under a documented protocol.
3. Evaluate with leak-free splits and class-level metrics; analyse errors and calibration.
4. Track experiments reproducibly (YAML configs, seeds, MLflow, TensorBoard).
5. Deploy the selected model as a tested, containerised API.

## 4. Technical architecture
See the diagram in `README.md`. Data (CIFAR-10) -> transforms/DataLoaders -> model factory -> trainer (+ tracking) ->
evaluation -> selection (validation accuracy) -> ONNX/benchmarks -> predictor -> FastAPI -> Docker.

## 5. Dataset
CIFAR-10 (Krizhevsky, 2009): 60,000 colour images of 32x32 pixels, 10 balanced classes. We use the official 50,000 train /
10,000 test split and carve a seeded 5,000-image validation set from the training portion
(45,000 train / 5,000 validation / 10,000 test). Validation and test images are never used for training.

## 6. Data preprocessing
- Training: RandomCrop(32, padding=4) and RandomHorizontalFlip at native resolution, optional upscaling, tensor conversion, normalization.
- Validation/test: no augmentation (deterministic), same resize and normalization.
- Normalization: CIFAR-10 channel statistics for the custom CNN; ImageNet statistics for pretrained ResNet18 models.
- Input size per model: see the `img_size` column below.

## 7. Model architectures and sizes
{table}

Architectures: `simple_cnn`/`improved_cnn` = three conv blocks plus a two-layer classifier (the improved version adds BatchNorm and
Dropout); `resnet18` = torchvision ResNet18 with a new 512->10 linear head. `trainable` is the count at the END of training.

## 8. Training methodology
Cross-entropy loss, mini-batch training, per-epoch validation, and checkpointing of the epoch with the highest validation accuracy.
Reproducibility: fixed seeds, YAML-configured runs, config hash and git commit recorded per run (MLflow).

## 9. Fine-tuning methodology
Two stages: (1) train only the new head with the backbone frozen; (2) unfreeze selected backbone stages and train them with
discriminative learning rates (smaller for earlier stages) using AdamW and cosine decay. Frozen BatchNorm layers stay in eval mode.

## 10. Experimental setup
Training devices: {", ".join(devices)}. Single seed per configuration unless noted in the experiment log (variance is therefore not
fully characterised). Full run log below (Section 14).

## 11. Hyperparameters (exact configuration of each run)
{hp}

## 12. Evaluation metrics
Accuracy, macro precision/recall/F1, one-vs-rest ROC-AUC (in the JSON reports), confusion matrices, expected calibration
error (ECE, 15 bins), inference latency and model size.

## 13. Results
{results_block}

### Inference benchmark of the final model (`{final}`), validation accuracy
{benchmark_table(final)}

Latency measures the model only (no decoding, no HTTP). Hardware details are in `artifacts/metrics/benchmark_{final}.json`.

**Discussion: TODO(you).** In 8-12 lines, using ONLY the table above, answer: Which model is best on validation and on test?
How large is the gap between feature extraction and fine-tuning? Did the pretrained models beat the scratch CNN, at what extra cost
(time, size, latency)? How do train/val/test accuracy compare (overfitting)? What do ECE values say about confidence? Mention seed variance if you measured it.

## 14. Error analysis (final model: `{final}`, test set)
Weakest classes by F1: {worst}.
Strongest classes by F1: {best}.

Most frequent confusions:
{confusions}

Figures: `artifacts/plots/{final}_test_confusion.png`, `{final}_test_errors.png`, `{final}_test_reliability.png`.

### Your analysis notes
{your_notes}

### Experiment log
{exp_log}

## 15. Deployment
The selected checkpoint is staged at `artifacts/final/model.pt` (metadata in `model_info.json`: run `{fin_row['run']}`,
validation accuracy {fin_row['val_acc']}). It is loaded by `deepvision.inference.Predictor` and served by FastAPI (`api/`),
packaged by the repository `Dockerfile`; the weights are mounted at runtime, not baked into the image.

## 16. API usage
`GET /health`, `GET /ready`, `GET /classes`, `GET /model-info`, `POST /predict?top_k=3` (multipart field `file`: JPEG/PNG/WebP, default max 5 MB).
Interactive documentation at `/docs`. Example: `python scripts/predict_client.py demo_images/cat.png`.

## 17. Installation
See the README Quickstart (virtual environment, PyTorch for your hardware, `pip install -r requirements.txt`, `pip install -e .`).

## 18. Limitations
- CIFAR-10 images are tiny (32x32); real photos differ in resolution, framing and distribution, so accuracy on them is unknown and likely lower.
- The classifier always chooses one of 10 classes; out-of-distribution inputs receive confident wrong labels (`low_confidence` only partly mitigates this).
- One seed per configuration; small differences between models may be within run-to-run noise.
- Pretrained models were upscaled from 32x32, which introduces blur not present in ImageNet data.
- Latency was measured on one machine and excludes preprocessing and network time.
- Known label noise and near-duplicate images between CIFAR train and test sets have been reported in the literature (ciFAIR, Barz & Denzler).

## 19. Future improvements
Repeat runs over several seeds with confidence intervals; hyperparameter search; stronger augmentation (MixUp/CutMix, RandAugment);
modified ResNet stem for small images; temperature scaling for calibration; static int8 quantization with calibration data;
ONNX Runtime serving; input-drift monitoring; out-of-distribution detection.

## 20. References
1. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Deng, J. et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. CVPR.
4. Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
5. Ioffe, S., Szegedy, C. (2015). Batch Normalization. ICML.
6. Srivastava, N. et al. (2014). Dropout. JMLR.
7. Kingma, D., Ba, J. (2015). Adam. ICLR. Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
8. Yosinski, J. et al. (2014). How transferable are features in deep neural networks? NeurIPS.
9. Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
10. Barz, B., Denzler, J. (2020). Do We Train on Test Data? Purging CIFAR of Near-Duplicates. Journal of Imaging.
"""
    Path("reports").mkdir(exist_ok=True)
    Path("reports/final_report.md").write_text(report, encoding="utf-8")

    card = f"""# Model Card: DeepVision AI classifier (`{final}`)

## Model details
- Architecture: `{fin_row['arch']}` ({fin_row['params']:,} parameters), training strategy: {fin_row['strategy']}.
- Input: RGB image resized to 32x32, then processed to {fin_row['img_size']}x{fin_row['img_size']}; output: probabilities for 10 classes.
- Framework: PyTorch {torch.__version__}. Training run `{final}` (details and git commit in `artifacts/final/model_info.json`).
- Pretrained components: ImageNet weights from torchvision (where the architecture is `resnet18`); check the weights' terms of use before commercial use.
- Developed by: TODO(you: your name). License: see repository LICENSE.

## Intended use
Educational and portfolio demonstration of image-classification training, fine-tuning and deployment on CIFAR-10-style images.

## Out-of-scope use
Safety-critical or high-stakes decisions; classifying arbitrary real-world photos without separate validation; any use where a wrong
confident label could harm someone.

## Training and evaluation data
CIFAR-10 (50,000 training images -> 45,000 train / 5,000 validation, seeded split; 10,000 official test images, used once for the final report).
Provenance: CIFAR-10 is a labelled subset of the Tiny Images collection; consult the dataset page for its current status and terms.

## Metrics (test set, one evaluation)
| metric | value |
|---|---|
| accuracy | {fin_row['test_acc']} |
| macro precision | {fin_row['test_P']} |
| macro recall | {fin_row['test_R']} |
| macro F1 | {fin_row['test_F1']} |
| expected calibration error | {fin_row['ECE']} |
| validation accuracy (used for selection) | {fin_row['val_acc']} |

Weakest classes by F1: {worst}.

## Limitations and risks
- Trained only on 32x32 images of 10 object classes; unknown behaviour on other image types, resolutions, lighting, or objects outside the 10 classes.
- It always outputs one of the 10 classes, even for unrelated images; the API's `low_confidence` flag is a heuristic, not a guarantee.
- Reported numbers come from a single training run; they carry run-to-run uncertainty.
- The dataset has known labelling noise and near-duplicate images across train and test (Barz & Denzler, 2020).

## Ethical considerations
No personal data is used. Do not use this model to make decisions about people.
"""
    Path("MODEL_CARD.md").write_text(card, encoding="utf-8")
    print("Wrote reports/final_report.md, MODEL_CARD.md, docs/results_table.md and updated the README block.")
    print("Search for 'TODO(you)' and complete those sections yourself.")


if __name__ == "__main__":
    main()
