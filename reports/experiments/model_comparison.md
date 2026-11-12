# Model comparison

| run | params | trainable (end) | train_s | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc | test_P_macro | test_R_macro | test_F1_macro | ECE | size_MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p6_improved_adamw_cos | 620586 | 620586 | 683 | 0.631 | 0.7785 | 0.5563 | 0.8014 | 0.5734 | 0.7963 | 0.7951 | 0.7963 | 0.7955 | 0.0096 | 2.369 |
| r18_head | 11181642 | 5130 | 1090 | 0.8287 | 0.7082 | 0.7955 | 0.7338 | 0.831 | 0.7235 | 0.7259 | 0.7235 | 0.7211 | 0.0778 | 42.691 |
| r18_ft_l34 | 11181642 | 10498570 | 2668 | 0.2518 | 0.9121 | 0.3489 | 0.8812 | 0.3728 | 0.8783 | 0.8782 | 0.8783 | 0.878 | 0.0241 | 42.691 |

Selection rule: highest VALIDATION accuracy -> **r18_ft_l34**. Test numbers are reported for transparency and were not used to choose.
