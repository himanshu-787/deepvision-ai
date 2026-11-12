# Error Analysis and Model Diagnostics

This report synthesizes diagnostic evaluations across our three core architectures on CIFAR-10:
- **Model A (`p6_improved_adamw_cos`)**: Custom `ImprovedCNN` with BatchNorm and Dropout trained from scratch with AdamW + Cosine LR scheduler.
- **Model B (`r18_head`)**: Pretrained ResNet-18 feature extractor with frozen backbone and trained linear classification head.
- **Model C (`r18_ft_l34`)**: Two-stage discriminative fine-tuning of ResNet-18 (unfreezing `layer3`, `layer4`, and `fc`).

---

## 1. Confusion Matrix Analysis

- **Model A (`p6_improved_adamw_cos`)**:
  - The dominant confusion is bidirectional animal misclassification: **`cat` $\leftrightarrow$ `dog`** (on the test set: 156 cats predicted as dogs, 150 dogs predicted as cats).
  - Secondary confusions occur within semantic superclasses: `bird` $\to$ `deer` (74) and `truck` $\to$ `automobile` (32).
- **Model B (`r18_head`)**:
  - Displays pronounced cross-category confusions caused by domain mismatch between full-resolution ImageNet pretraining and upscaled $64 \times 64$ CIFAR images without backbone adaptation.
  - Top confusions: `ship` $\to$ `airplane` (146), `deer` $\to$ `bird` (135), and `dog` $\to$ `cat` (123). Sky and water background textures mislead the frozen representation.
- **Model C (`r18_ft_l34`)**:
  - Discriminative fine-tuning almost completely resolves the background-driven errors (ship/airplane confusions drop below top ranking).
  - Residual confusions concentrate on fine-grained biological overlap: `cat` $\to$ `dog` (130) and `dog` $\to$ `cat` (100), along with minor vehicle silhouette overlap (`automobile` $\to$ `truck`: 45).

---

## 2. Per-Class F1 Score Breakdown

| Class | Model A Test F1 | Model B Test F1 | Model C Test F1 | $\Delta$ (Model C vs Model A) |
|---|---|---|---|---|
| **airplane** | 0.826 | 0.771 | **0.893** | +0.067 |
| **automobile** | 0.887 | 0.822 | **0.932** | +0.045 |
| **bird** | 0.698 | 0.634 | **0.860** | +0.162 |
| **cat** (weakest) | **0.621** | **0.599** | **0.760** | **+0.139** |
| **deer** | 0.775 | 0.617 | **0.852** | +0.077 |
| **dog** | 0.717 | 0.648 | **0.813** | +0.096 |
| **frog** | 0.861 | 0.793 | **0.916** | +0.055 |
| **horse** | 0.835 | 0.762 | **0.899** | +0.064 |
| **ship** | 0.884 | 0.781 | **0.930** | +0.046 |
| **truck** | 0.850 | 0.783 | **0.925** | +0.075 |
| **Macro Average** | **0.795** | **0.721** | **0.878** | **+0.083** |

**Observation**: `cat` is consistently the most challenging class across all models. Fine-tuning in Model C provides a +13.9 percentage point boost on `cat` and a +16.2 point boost on `bird`.

---

## 3. High-Confidence Error Gallery Analysis (`*_errors.png`)

Inspection of the top misclassified samples sorted by predicted confidence reveals:
1. **Low native resolution ambiguity**: At $32 \times 32$ native pixels, small animals with obstructed snouts or curled postures lack visible diagnostic features (e.g., small dogs sitting like cats).
2. **Context occlusion**: Animals photographed in water or tall grass are frequently misclassified based on dominant background context.
3. **Vehicle angle ambiguity**: Flatbed trucks photographed head-on share frontal geometry with sedans, causing high-confidence `automobile` predictions on true `truck` labels.

---

## 4. Model Calibration and Expected Calibration Error (ECE)

| Metric | Model A (`ImprovedCNN`) | Model B (`r18_head`) | Model C (`r18_ft_l34`) |
|---|---|---|---|
| **Test Accuracy** | 79.63% | 72.35% | **87.83%** |
| **Mean Confidence** | 79.64% | 80.13% | 90.18% |
| **ECE** | **0.0096** | 0.0778 | 0.0241 |
| **Calibration Status** | Near-perfect alignment | Strongly over-confident | Well-calibrated |

- **Model A** is exceptionally well-calibrated (ECE = 0.0096); the predicted confidence almost matches empirical accuracy.
- **Model B** suffers from over-confidence: its frozen ImageNet representations produce peaky probability distributions (average confidence 80.1%) despite only achieving 72.4% accuracy.
- **Model C** maintains robust calibration (ECE = 0.0241) while boosting average confidence to 90.2% in accordance with its high 87.8% accuracy.

---

## 5. Generalization: Train vs. Validation vs. Test

| Model | Train Acc (at Best) | Val Acc | Test Acc | Val-to-Test Gap | Overfitting Gap (Train - Val) |
|---|---|---|---|---|---|
| **Model A** | 77.85% | 80.14% | 79.63% | 0.51% | -2.29% (no overfit) |
| **Model B** | 70.82% | 73.38% | 72.35% | 1.03% | -2.56% (underfit) |
| **Model C** | 91.21% | 88.12% | 87.83% | 0.29% | +3.09% (healthy) |

All models exhibit minimal divergence between validation and test scores (<1.1%), proving that model selection on the validation split accurately reflected true test performance without data snooping or test-set overfitting.

---

## 6. Hardware, Compute and Resource Trade-Offs

| Model | Parameters | Trainable | Checkpoint Size | CPU Training Time | Test Accuracy |
|---|---|---|---|---|---|
| **Model A** | 620,586 | 620,586 | 2.37 MB | 683 s (~11 min) | 79.63% |
| **Model B** | 11,181,642 | 5,130 | 42.69 MB | 1,090 s (~18 min) | 72.35% |
| **Model C** | 11,181,642 | 10,498,570 | 42.69 MB | 2,668 s (~44 min) | **87.83%** |

- **Edge / Resource-Constrained Deployment**: Model A is 18× smaller (2.37 MB vs 42.69 MB) and 4× faster to train, while reaching ~80% accuracy.
- **High-Accuracy Deployment**: Model C delivers +8.2% test accuracy over Model A, justifying the parameter count for cloud or desktop vision pipelines.

---

## 7. Documented Limitations

1. **Resolution limitations**: Source images are natively $32 \times 32$; bicubic interpolation to $64 \times 64$ does not restore lost high-frequency spatial information.
2. **Synthetic balance**: CIFAR-10 contains an exact uniform distribution (5,000 train images per class). Real-world applications typically have imbalanced long-tailed distributions.
3. **Out-of-Distribution (OOD)**: Evaluation is strictly in-distribution; performance on noisy, corrupted (e.g., CIFAR-10-C), or natural distribution shifts remains unmeasured.
4. **Single random seed**: The primary comparison models used seed 42; variance across random initializations is assessed via Phase 10 seed reproducibility trials.
