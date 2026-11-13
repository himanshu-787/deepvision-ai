# Model Card: DeepVision AI classifier (`r18_ft_l34`)

## Model details
- Architecture: `resnet18` (11,181,642 parameters), training strategy: finetune_partial.
- Input: RGB image resized to 32x32, then processed to 64x64; output: probabilities for 10 classes.
- Framework: PyTorch 2.15.0+cpu. Training run `r18_ft_l34` (details and git commit in `artifacts/final/model_info.json`).
- Pretrained components: ImageNet weights from torchvision (where the architecture is `resnet18`); check the weights' terms of use before commercial use.
- Developed by: Himanshu (https://github.com/himanshu-787). License: see repository LICENSE.

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
| accuracy | 0.8783 |
| macro precision | 0.8782 |
| macro recall | 0.8783 |
| macro F1 | 0.878 |
| expected calibration error | 0.0241 |
| validation accuracy (used for selection) | 0.8812 |

Weakest classes by F1: cat (F1 0.760), dog (F1 0.813), deer (F1 0.852).

## Limitations and risks
- Trained only on 32x32 images of 10 object classes; unknown behaviour on other image types, resolutions, lighting, or objects outside the 10 classes.
- It always outputs one of the 10 classes, even for unrelated images; the API's `low_confidence` flag is a heuristic, not a guarantee.
- Reported numbers come from a single training run; they carry run-to-run uncertainty.
- The dataset has known labelling noise and near-duplicate images across train and test (Barz & Denzler, 2020).

## Ethical considerations
No personal data is used. Do not use this model to make decisions about people.
