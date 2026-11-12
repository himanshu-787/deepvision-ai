"""Phase 6 - compare optimizers on an elongated bowl: f(x, y) = 0.5 * (x^2 + 25 y^2)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def run(make_opt, steps: int = 100):
    p = torch.tensor([3.0, 1.0], requires_grad=True)   # start point (x, y)
    opt = make_opt([p])
    losses = []
    for _ in range(steps):
        loss = 0.5 * (p[0] ** 2 + 25 * p[1] ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


runs = {
    "SGD lr=0.03": lambda ps: torch.optim.SGD(ps, lr=0.03),
    "SGD+momentum 0.9": lambda ps: torch.optim.SGD(ps, lr=0.03, momentum=0.9),
    "Adam lr=0.1": lambda ps: torch.optim.Adam(ps, lr=0.1),
}
Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
for name, make in runs.items():
    losses = run(make)
    plt.semilogy(losses, label=name)
    print(f"{name:18s} loss@10={losses[9]:.4f}  loss@50={losses[49]:.5f}  loss@100={losses[99]:.6f}")
plt.xlabel("step"); plt.ylabel("loss (log scale)"); plt.legend()
plt.savefig("artifacts/plots/phase6_optimizers_toy.png", dpi=120)
plt.close()
