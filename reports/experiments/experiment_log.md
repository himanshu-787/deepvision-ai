# DeepVision AI: Experiment Log & Evidence Trail

This document logs all experiments across Phases 5 through 10, detailing the hypothesis, configuration changes, measured outcomes, and engineering decisions.

---

## Experiment 1: Phase 5 Baseline CNN
- **Run Name**: `cnn_baseline`
- **Model**: `SimpleCNN` (3 convolutional blocks with MaxPool, Linear 512, Linear 10; 620,362 params)
- **Hypothesis**: Standard mini-batch SGD with Adam ($lr=10^{-3}$, no weight decay) on raw $32 \times 32$ CIFAR-10 will establish a solid convolutional baseline (~75–80% validation accuracy).
- **Configuration**:
  - Epochs: 15
  - Optimizer: Adam ($lr=0.001$, $\beta=(0.9, 0.999)$)
  - Regularization: Standard data augmentation (RandomCrop + RandomHorizontalFlip); no dropout, no batch normalization.
- **Results**:
  - Best Val Accuracy: **80.28%** at epoch 14
  - Train Accuracy at Best: 79.91%
  - Total Training Time: 744 s
- **Observations & Symptoms**: Overfitting started appearing by epoch 12 as training loss dropped to 0.55 while validation loss plateaued at ~0.58.
- **Decision**: Accept as the Phase 5 benchmark. Address generalization and variance in Phase 6 with architectural regularization and decoupled weight decay.

---

## Experiment 2: Phase 6 Regularized CNN with Cosine Annealing (Model A)
- **Run Name**: `p6_improved_adamw_cos`
- **Model**: `ImprovedCNN` (Conv blocks with `bias=False`, `BatchNorm2d`, `Dropout(0.3)`; 620,586 params)
- **Hypothesis**: Adding Batch Normalization stabilizes internal covariate shifts, Dropout prevents co-adaptation of features, and AdamW with decoupled weight decay ($10^{-2}$) and Cosine Annealing will eliminate overfitting and improve calibration.
- **Configuration**:
  - Epochs: 15
  - Optimizer: AdamW ($lr=0.001$, weight decay = $0.01$)
  - Scheduler: CosineAnnealingLR ($T_{\max}=15$, $\eta_{\min}=0$)
- **Results**:
  - Best Val Accuracy: **80.14%** at epoch 15
  - Train Accuracy at Best: 77.85% (no overfitting; train-to-val gap is negative: -2.29%)
  - Test Accuracy: **79.63%** | Macro F1: **0.7955**
  - ECE: **0.0096** (near-perfect calibration)
  - Total Training Time: 683 s
- **Decision**: Select as **Model A** (Custom CNN champion). The model achieved near-identical accuracy to baseline while dramatically eliminating overfitting and lowering ECE to under 1%.

---

## Experiment 3: Phase 7 Transfer Learning Feature Extractor (Model B)
- **Run Name**: `r18_head`
- **Model**: ResNet-18 (Pretrained on ImageNet-1k; frozen backbone, new trainable linear head `nn.Linear(512, 10)`; 11,181,642 total params, 5,130 trainable params)
- **Hypothesis**: Reusing rich generic hierarchical visual features from ImageNet will allow fast convergence and competitive classification with minimal trainable parameters.
- **Configuration**:
  - Input resolution: $64 \times 64$ px (bicubic interpolation)
  - Normalization: ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
  - Epochs: 10
  - Optimizer: Adam ($lr=0.001$, weight decay = 0.0)
- **Results**:
  - Best Val Accuracy: **73.38%** at epoch 10
  - Test Accuracy: **72.35%** | Macro F1: **0.7211**
  - ECE: **0.0778** (over-confident)
  - Total Training Time: 1,090 s
- **Observations & Symptoms**: Accuracy plateaued at ~73% because the backbone weights are completely rigid. The domain gap between high-resolution ImageNet and low-resolution CIFAR-10 causes feature misalignment (e.g. background textures confuse ships with airplanes).
- **Decision**: Keep as **Model B** to demonstrate the baseline capabilities and limits of frozen feature extraction. Proceed to fine-tuning deep layers in Phase 8.

---

## Experiment 4: Phase 8 Discriminative Stage Fine-Tuning (Model C)
- **Run Name**: `r18_ft_l34`
- **Model**: ResNet-18 initialized from `r18_head_best.pt` with `layer3`, `layer4`, and `fc` unfrozen (10,498,570 trainable params).
- **Hypothesis**: Fine-tuning deeper task-specific convolutional stages (`layer3`, `layer4`) using differential/discriminative learning rates (slower for earlier layers, faster for the head) will adapt the model to CIFAR-10 semantics without catastrophic forgetting.
- **Configuration**:
  - Differential LRs: `fc` ($10^{-3}$), `layer4` ($10^{-4}$), `layer3` ($5 \times 10^{-5}$)
  - Optimizer: AdamW (weight decay = $10^{-4}$)
  - Epochs: 7 fine-tuning epochs ($64 \times 64$ px)
- **Results**:
  - Best Val Accuracy: **88.12%** at epoch 7
  - Test Accuracy: **87.83%** | Macro F1: **0.8780** | ROC-AUC (OvR): **0.9910**
  - ECE: **0.0241**
  - Total Training Time: 2,668 s
- **Observations**: Dramatic performance leap (+14.7% over frozen head, +8.2% over custom CNN). Confusions between ships and planes collapsed, and cat/bird detection improved by over 14 percentage points.
- **Decision**: Unanimously selected as the **champion model (Model C)** by validation accuracy.

---

## Experiment 5: Phase 10 Tracking Stack & Seed Reproducibility
- **Run Names**: `smoke_cnn` (seed 42), `cnn_seed1` (seed 1)
- **Configuration**: Config-driven execution via `configs/cnn.yaml` with MLflow and TensorBoard active.
- **Results**:
  - `smoke_cnn`: Val Acc = 59.30%, Train Loss = 1.5215, Train Acc = 43.98% (Epoch 1)
  - `cnn_seed1`: Val Acc = 57.58%, Train Loss = 1.5427, Train Acc = 43.58% (Epoch 1)
  - Variance: $\Delta \text{Val Acc} \approx 1.72\%$ across different seeds at epoch 1.
- **MLflow Tracking**: Parameters, metrics, tags (git commit `8dd78b6`, config hash, device `cpu`), and artifacts (best model checkpoint, resolved yaml config, metrics json, curves png) successfully recorded in `mlflow.db` and exported to `reports/experiments/mlflow_runs.csv`.
- **Decision**: Confirms that pipeline execution is robust, reproducible, and tracked.
