| model | run | arch | strategy | img_size | params | trainable | train_s | train_acc | val_acc | test_acc | test_P | test_R | test_F1 | ECE | size_MB | cpu_ms_bs1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | p6_improved_adamw_cos | improved_cnn | scratch | 32 | 620586 | 620586 | 683 | 0.7785 | 0.8014 | 0.7963 | 0.7951 | 0.7963 | 0.7955 | 0.0096 | 2.369 | 0.618 |
| B | r18_head | resnet18 | head_only | 64 | 11181642 | 5130 | 1090 | 0.7082 | 0.7338 | 0.7235 | 0.7259 | 0.7235 | 0.7211 | 0.0778 | 42.691 | 8.393 |
| C | r18_ft_l34 | resnet18 | finetune_partial | 64 | 11181642 | 10498570 | 2668 | 0.9121 | 0.8812 | 0.8783 | 0.8782 | 0.8783 | 0.878 | 0.0241 | 42.691 | 8.671 |

Selection rule: highest **validation** accuracy (with any override documented in `artifacts/final/model_info.json`). Final model: **r18_ft_l34** (highest validation accuracy).

Test metrics are from one evaluation per model and were not used for selection. `cpu_ms_bs1` = median single-image latency, PyTorch fp32 on this machine's CPU, model only.
