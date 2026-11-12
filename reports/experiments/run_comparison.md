# Training run comparison (validation set)

| run | model | opt | lr | wd | sched | params | best_val_acc | best_epoch | train_acc@best | gap | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r18_ft_l34 | resnet18 | adam | - | 0.0001 | none | 11181642 | 0.8812 | 7 | 0.9121 | 0.0309 | 2668 |
| cnn_baseline | simple_cnn | adam | 0.001 | 0.0 | none | 620362 | 0.8028 | 14 | 0.7991 | -0.0037 | 744 |
| p6_improved_adamw_cos | improved_cnn | adamw | 0.001 | 0.01 | cosine | 620586 | 0.8014 | 15 | 0.7785 | -0.0229 | 683 |
| r18_head | resnet18 | adam | 0.001 | 0.0 | none | 11181642 | 0.7338 | 10 | 0.7082 | -0.0256 | 1090 |
