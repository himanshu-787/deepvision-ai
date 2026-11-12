"""Phase 6 - what Dropout and BatchNorm do in train vs eval mode."""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- Dropout ----
x = torch.ones(1, 10)
drop = nn.Dropout(p=0.5)
drop.train()
print("train mode:", drop(x))   # about half zeros; survivors scaled by 1/(1-p) = 2.0
drop.eval()
print("eval mode :", drop(x))   # unchanged

# ---- BatchNorm ----
bn = nn.BatchNorm1d(3)
data = torch.randn(256, 3) * 3 + 5                      # per-feature mean ~5, std ~3
bn.train()
out = bn(data)
print("\nnormalized mean:", out.mean(0).detach().numpy().round(3))
print("normalized std :", out.std(0).detach().numpy().round(3))
print("running_mean after 1 batch  :", bn.running_mean.numpy().round(2))
for _ in range(50):
    bn(data)
print("running_mean after 51 batches:", bn.running_mean.numpy().round(2))   # drifts toward ~5

bn.eval()
single = data[:1]
print("eval mode works on a batch of 1:", bn(single).shape)   # uses running stats
