"""Phase 2.2 - autograd and computational graphs."""

import torch

# y = x^2 + 3x  ->  dy/dx = 2x + 3  -> at x=2 should be 7
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x
y.backward()
print("dy/dx at x=2:", x.grad.item())       # 7.0

# A tiny neuron by hand: pred = w*x + b, loss = (pred - target)^2
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
inp, target = 2.0, 5.0
pred = w * inp + b                           # 3.5
loss = (pred - target) ** 2                  # 2.25
loss.backward()
print("loss:", loss.item())
print("dL/dw:", w.grad.item(), "(hand calc: 2*(3.5-5)*2 = -6)")
print("dL/db:", b.grad.item(), "(hand calc: 2*(3.5-5)   = -3)")

# Gradients ACCUMULATE: this is why we need zero_grad() in every training step
w.grad.zero_()
for _ in range(2):
    ((w * inp + b - target) ** 2).backward()
print("after two backward calls without zeroing:", w.grad.item())  # -12

# Turning tracking off (used in validation / inference)
with torch.no_grad():
    out = w * inp + b
print("requires_grad inside no_grad:", out.requires_grad)          # False

# detach: cut a tensor out of the graph
print("detached:", (w * 2).detach().requires_grad)                 # False
